import re
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn


FORMAT_PREFIX = ("the", "captain")
CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts" / "assignment5_rl.pt"


def tokenize(text: str) -> list[str]:
    """Use the Module 9 word tokenizer."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().split()


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        return self.token_emb(tokens) + self.pos_emb(positions)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, ff_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.dropout_1 = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(0.1),
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        attention, _ = self.attn(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
        tokens = self.ln_1(tokens + self.dropout_1(attention))
        return self.ln_2(tokens + self.ffn(tokens))


class GPT(nn.Module):
    """Module 9's one-block GPT, with matching checkpoint parameter names."""

    def __init__(
        self, max_len: int, vocab_size: int, embed_dim: int, num_heads: int, ff_dim: int
    ):
        super().__init__()
        self.max_len = max_len
        self.embed = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        self.transformer = TransformerBlock(num_heads, embed_dim, ff_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.transformer(self.embed(tokens)))

    def next_word_logits(self, tokens: torch.Tensor, vocabulary_size: int) -> torch.Tensor:
        # The notebook reserves IDs 0 and 1 for PAD and UNK. Valid words start at 2.
        return self(tokens[:, -self.max_len:])[:, -1, 2:vocabulary_size]


def load_checkpoint(path: Path) -> tuple[GPT, dict[int, str], dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    config.pop("key_dim", None)  # The notebook accepts this unused argument.
    model = GPT(**config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["index_to_word"], config


class TransformerGenerator:
    def __init__(self, model: GPT, index_to_word: dict[int, str]):
        self.model = model
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in index_to_word.items()}

    @torch.inference_mode()
    def generate(
        self, prompt: str, max_new_words: int, temperature: float, seed: int | None
    ) -> list[str]:
        words = tokenize(prompt)
        if not words:
            raise ValueError("The prompt must contain at least one letter or digit.")
        ids = [self.word_to_index.get(word, 1) for word in words[-self.model.max_len:]]
        context = torch.tensor([ids], dtype=torch.long)
        rng = None if seed is None else torch.Generator().manual_seed(seed)
        generated = []
        for _ in range(max_new_words):
            logits = self.model.next_word_logits(context, len(self.index_to_word)).squeeze(0)
            if temperature == 0:
                action = logits.argmax().item()
            else:
                probs = (logits / max(temperature, 1e-4)).softmax(dim=-1)
                action = torch.multinomial(probs, 1, generator=rng).item()
            word_id = action + 2
            generated.append(self.index_to_word[word_id])
            context = torch.cat((context, torch.tensor([[word_id]])), dim=1)
        return generated


@lru_cache(maxsize=1)
def load_transformer_generator() -> TransformerGenerator:
    model, index_to_word, _ = load_checkpoint(CHECKPOINT)
    return TransformerGenerator(model, index_to_word)
