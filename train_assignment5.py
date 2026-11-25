import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from app.transformer_model import FORMAT_PREFIX, GPT, load_checkpoint, tokenize


CORPUS_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
CORPUS_MIRROR = "https://gutenberg.pglaf.org/1/1/8/1184/1184-0.txt"
CORPUS_CACHE = Path("data/monte_cristo.txt")
BASE_CONFIG = {
    "max_len": 128,
    "vocab_size": 10_000,
    "embed_dim": 256,
    "num_heads": 4,
    "ff_dim": 1024,
}
TRAIN_PROMPTS = (
    ("he", "she", "then", "but", "when", "and"),
    ("at dawn", "on deck", "that morning", "they said", "after dinner", "meanwhile he"),
    ("in the harbor", "he looked up", "out at sea", "they went aboard"),
    ("when the sun rose", "as the ship sailed", "he opened the door"),
)
TEST_PROMPTS = (
    "they", "now", "perhaps", "there",
    "one day", "at midnight", "he thought", "she turned",
    "behind the door", "as he spoke", "they walked slowly", "in the distance",
    "when they came ashore", "at the next port", "he looked at her", "out in the street",
)


def download_corpus() -> str:
    errors = []
    for url in (CORPUS_URL, CORPUS_MIRROR):
        try:
            with urlopen(url, timeout=12) as response:
                return response.read().decode("utf-8-sig")
        except (URLError, TimeoutError) as error:
            errors.append(str(error))
    raise ValueError(
        f"Could not download Module 9's corpus ({'; '.join(errors)}). "
        f"Download {CORPUS_MIRROR} and pass the local file path with --corpus."
    )


def read_corpus(path: Path | None) -> list[str]:
    if path is not None and not path.is_file():
        raise ValueError(f"Corpus file not found: {path}. Pass the path to an existing .txt file.")
    corpus_file = CORPUS_CACHE if path is None else path
    needs_download = not corpus_file.is_file()
    text = download_corpus() if needs_download else corpus_file.read_text(encoding="utf-8-sig")
    start = text.find("Chapter 1")
    end = text.rfind("Chapter 5")
    if 0 <= start < end:
        words = tokenize(text[start:end])
    else:
        words = tokenize(text)
    if len(words) <= 30 or not set(FORMAT_PREFIX).issubset(words):
        raise ValueError("The corpus must be the Module 9 text and contain 'the' and 'captain'.")
    if needs_download:
        corpus_file.parent.mkdir(parents=True, exist_ok=True)
        corpus_file.write_text(text, encoding="utf-8")
    return words


def train_base(
    words: list[str], steps: int, batch_size: int, device: torch.device
) -> tuple[GPT, dict[int, str]]:
    frequent_words = Counter(words).most_common(BASE_CONFIG["vocab_size"] - 2)
    index_to_word = {0: "<PAD>", 1: "<UNK>"}
    index_to_word.update(
        (index, word) for index, (word, _) in enumerate(frequent_words, start=2)
    )
    word_to_index = {word: index for index, word in index_to_word.items()}
    encoded = torch.tensor([word_to_index.get(word, 1) for word in words], device=device)
    offsets = torch.arange(31, device=device)
    model = GPT(**BASE_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, steps + 1):
        starts = torch.randint(encoded.numel() - 30, (batch_size, 1), device=device)
        windows = encoded[starts + offsets]
        logits = model(windows[:, :-1])
        loss = F.cross_entropy(logits.transpose(1, 2), windows[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Base step {step}/{steps} | loss {loss.item():.3f}", flush=True)
    return model.eval(), index_to_word


def rollout(
    model: GPT, contexts: torch.Tensor, target: torch.Tensor, vocabulary_size: int
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Sample two generated words and record each action and its log probability."""
    actions, log_probs, states, logits_history = [], [], [], []
    for _ in FORMAT_PREFIX:
        logits = model.next_word_logits(contexts, vocabulary_size)
        policy = Categorical(logits=logits)
        action = policy.sample()
        word_ids = action + 2
        states.append(contexts)
        logits_history.append(logits)
        actions.append(word_ids)
        log_probs.append(policy.log_prob(action))
        contexts = torch.cat((contexts, word_ids[:, None]), dim=1)
    rewards = torch.stack(actions, dim=1).eq(target).float()
    return torch.stack(log_probs, dim=1), rewards, states, logits_history


@torch.no_grad()
def measure_prefix(
    model: GPT, word_to_index: dict[str, int], target: torch.Tensor, samples: int
) -> dict:
    matched_words = torch.zeros(len(FORMAT_PREFIX), device=target.device)
    complete_matches = 0
    greedy_matches = 0
    for prompt in TEST_PROMPTS:
        ids = [word_to_index.get(word, 1) for word in tokenize(prompt)]
        context = torch.tensor([ids], device=target.device)
        _, rewards, _, _ = rollout(model, context.repeat(samples, 1), target, len(word_to_index))
        matched_words += rewards.sum(dim=0)
        complete_matches += rewards.bool().all(dim=1).sum().item()
        generated = []
        for _ in FORMAT_PREFIX:
            word_id = model.next_word_logits(context, len(word_to_index)).argmax(dim=1) + 2
            generated.append(word_id)
            context = torch.cat((context, word_id[:, None]), dim=1)
        greedy_matches += int(torch.stack(generated, dim=1).eq(target).all().item())
    total = len(TEST_PROMPTS) * samples
    return {
        "exact_matches": complete_matches,
        "samples": total,
        "exact_pass_rate": complete_matches / total,
        "word_pass_rates": (matched_words / total).tolist(),
        "greedy_matches": greedy_matches,
        "greedy_prompts": len(TEST_PROMPTS),
    }


def continuation_kl(
    policy: GPT, reference: GPT, context: torch.Tensor, vocabulary_size: int
) -> torch.Tensor:
    """Keep the distribution after the requested prefix close to the base model."""
    losses = []
    for _ in range(3):
        with torch.no_grad():
            reference_probs = reference.next_word_logits(context, vocabulary_size).softmax(dim=-1)
            next_word = torch.multinomial(reference_probs, 1) + 2
        logits = policy.next_word_logits(context, vocabulary_size)
        losses.append(F.kl_div(logits.log_softmax(dim=-1), reference_probs, reduction="batchmean"))
        context = torch.cat((context, next_word), dim=1)
    return torch.stack(losses).mean()


def post_train(
    model: GPT, index_to_word: dict[int, str], steps: int, eval_samples: int
) -> tuple[GPT, dict]:
    device = next(model.parameters()).device
    word_to_index = {word: index for index, word in index_to_word.items()}
    target = torch.tensor([word_to_index[word] for word in FORMAT_PREFIX], device=device)
    policy = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    train_contexts = [
        torch.tensor(
            [[word_to_index.get(word, 1) for word in tokenize(prompt)] for prompt in prompts],
            device=device,
        ).repeat_interleave(8, dim=0)
        for prompts in TRAIN_PROMPTS
    ]

    before = measure_prefix(policy, word_to_index, target, eval_samples)
    for step in range(1, steps + 1):
        contexts = train_contexts[(step - 1) % len(train_contexts)]
        log_probs, rewards, states, logits = rollout(policy, contexts, target, len(word_to_index))
        # Each action gets the rewards received at that word and later words.
        returns = torch.stack((rewards.sum(dim=1), rewards[:, 1]), dim=1)
        advantages = returns - returns.mean(dim=0)
        policy_loss = -(log_probs * advantages).mean()
        with torch.no_grad():
            reference_probs = [
                model.next_word_logits(state, len(word_to_index)).softmax(dim=-1)
                for state in states
            ]
        kl_loss = sum(
            F.kl_div(output.log_softmax(dim=-1), reference, reduction="batchmean")
            for output, reference in zip(logits, reference_probs)
        ) / len(FORMAT_PREFIX)
        prefix_context = torch.cat((contexts, target.expand(contexts.size(0), -1)), dim=1)
        body_kl = continuation_kl(policy, model, prefix_context, len(word_to_index))
        loss = policy_loss + 0.02 * kl_loss + 0.15 * body_kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            batch_pass_rate = rewards.bool().all(dim=1).float().mean().item()
            print(f"RL step {step}/{steps} | batch exact match {batch_pass_rate:.1%}", flush=True)

    after = measure_prefix(policy, word_to_index, target, eval_samples)
    print(f"Held-out exact match: {before['exact_pass_rate']:.1%} → {after['exact_pass_rate']:.1%}")
    return policy, {"before": before, "after": after, "test_prompts": list(TEST_PROMPTS)}


def save_checkpoint(
    path: Path, model: GPT, index_to_word: dict[int, str], config: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "index_to_word": index_to_word,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Module 9 GPT and RL-post-train its format.")
    parser.add_argument("--corpus", type=Path, help="Path to a local copy of the Module 9 text.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("data/module9_gpt.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/assignment5_rl.pt"))
    parser.add_argument("--base-steps", type=int, default=400)
    parser.add_argument("--base-batch-size", type=int, default=16)
    parser.add_argument("--rl-steps", type=int, default=300)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.base_steps, args.base_batch_size, args.rl_steps, args.eval_samples) < 1:
        parser.error("Step counts, batch size, and sample count must be positive.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(args.device)
    if args.base_checkpoint.is_file():
        base, vocab, config = load_checkpoint(args.base_checkpoint)
        base = base.to(device)
    else:
        try:
            words = read_corpus(args.corpus)
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        print(f"Training Module 9 GPT on {len(words):,} corpus words using {device}", flush=True)
        base, vocab = train_base(words, args.base_steps, args.base_batch_size, device)
        config = BASE_CONFIG
        save_checkpoint(args.base_checkpoint, base, vocab, config)

    torch.manual_seed(args.seed)
    policy, metrics = post_train(base, vocab, args.rl_steps, args.eval_samples)
    save_checkpoint(args.checkpoint, policy, vocab, config)
    metrics.update(prefix=" ".join(FORMAT_PREFIX), seed=args.seed, rl_steps=args.rl_steps)
    args.checkpoint.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.checkpoint}")


if __name__ == "__main__":
    main()
