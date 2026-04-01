import argparse

import torch

from model.MokioModel import MokioMindConfig
from trainer.trainer_utils import init_model


def build_parser():
    parser = argparse.ArgumentParser(description="Inference for MokioMind pretrain weights")
    parser.add_argument("--save_dir", type=str, default="./out")
    parser.add_argument("--weight", type=str, default="pretrain")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=None)
    return parser


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[1] >= model.config.max_position_embeddings:
        raise ValueError("Prompt is longer than max_seq_len; increase --max_seq_len.")

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(input_ids, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < values[:, [-1]]] = float("-inf")
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        if input_ids.shape[1] >= model.config.max_position_embeddings:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    args = build_parser().parse_args()
    config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_seq_len,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        flash_attention="cuda" in args.device,
    )

    model, tokenizer = init_model(
        config,
        from_weight=args.weight,
        save_dir=args.save_dir,
        device=args.device,
    )

    if args.prompt is not None:
        print(
            generate_text(
                model,
                tokenizer,
                args.prompt,
                args.device,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
            )
        )
        return

    while True:
        prompt = input("Prompt> ").strip()
        if not prompt:
            break
        print(
            generate_text(
                model,
                tokenizer,
                prompt,
                args.device,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
            )
        )
        print()


if __name__ == "__main__":
    main()
