import argparse
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import PretrainDataset
from model.MokioModel import MokioMindConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser(description="MokioMind dense pretraining only")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="pretrain")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/sample_pretrain.jsonl",
    )
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MokioMind-Pretrain")
    return parser


def train_epoch(
    epoch,
    loader,
    iters,
    args,
    model,
    optimizer,
    scaler,
    autocast_ctx,
    lm_config,
    start_step=0,
    wandb=None,
):
    start_time = time.time()

    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = output.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            eta_min = spend_time / max(step, 1) * max(iters - step, 0) / 60
            Logger(
                f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} "
                f"loss {current_loss:.6f} lr {lr:.8f} eta {eta_min:.1f}m"
            )
            if wandb:
                wandb.log({"loss": current_loss, "lr": lr, "eta_min": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            weight_path = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth"
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({k: v.half() for k, v in state_dict.items()}, weight_path)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()


def main(cli_args=None):
    parser = build_parser()
    args = parser.parse_args(cli_args)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_seq_len,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        flash_attention="cuda" in args.device,
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb.init(
            project=args.wandb_project,
            name=f"pretrain-h{args.hidden_size}-l{args.num_hidden_layers}",
            id=wandb_id,
            resume=resume,
        )

    model, tokenizer = init_model(
        lm_config,
        from_weight=args.from_weight,
        device=args.device,
        save_dir=args.save_dir,
    )
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in args.device and args.dtype == "float16"))

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=("cuda" in args.device),
            )
            Logger(
                f"resume epoch {epoch + 1}/{args.epochs}, skip first {start_step} steps"
            )
            train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                args,
                model,
                optimizer,
                scaler,
                autocast_ctx,
                lm_config,
                start_step=start_step,
                wandb=wandb,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=("cuda" in args.device),
            )
            train_epoch(
                epoch,
                loader,
                len(loader),
                args,
                model,
                optimizer,
                scaler,
                autocast_ctx,
                lm_config,
                wandb=wandb,
            )


if __name__ == "__main__":
    main()
