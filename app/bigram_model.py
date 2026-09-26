from collections import Counter, defaultdict
import random
import re


class BigramModel:
    def __init__(self, corpus: list[str]):
        """Create a bigram model from a list of sentences."""
        text = " ".join(corpus)
        self.bigram_probabilities = self._analyze_bigrams(text)

    def _simple_tokenizer(self, text: str) -> list[str]:
        """Convert text to lowercase and split it into words."""
        return re.findall(r"\b\w+\b", text.lower())

    def _analyze_bigrams(self, text: str):
        """Calculate the possible next words and their probabilities."""
        words = self._simple_tokenizer(text)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probabilities = defaultdict(dict)

        for (word1, word2), count in bigram_counts.items():
            probability = count / unigram_counts[word1]
            bigram_probabilities[word1][word2] = probability

        return bigram_probabilities

    def generate_text(self, start_word: str, length: int = 20) -> str:
        """Generate text beginning with the requested word."""
        if length <= 0:
            return ""

        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probabilities.get(current_word)

            if not next_words:
                break

            next_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values()),
                k=1,
            )[0]

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)