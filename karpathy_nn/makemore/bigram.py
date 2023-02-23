import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn.functional as F


class CountingBigram:
    def __init__(self, words: list[str]) -> None:
        (
            self.co_occurrence_matrix,
            self.integer_to_string,
            self.string_to_integer,
        ) = self._calculate_co_occurrences(words)
        self.probability_matrix = self._get_probability_matrix(
            self.co_occurrence_matrix
        )

    def visualize_co_occurrences(self) -> None:
        """Visualizes the given co-occurrence matrix with labels given by
        the ``self.integer_to_string`` dict.

        """
        plt.figure(figsize=(16, 16))
        plt.imshow(self.co_occurrence_matrix, cmap="Blues")

        for i in range(self.co_occurrence_matrix.shape[0]):
            for j in range(self.co_occurrence_matrix.shape[1]):
                bigram_string = self.integer_to_string[i] + self.integer_to_string[j]

                # ha = horizontal alignment
                # va = vertical alignment
                plt.text(j, i, bigram_string, ha="center", va="bottom", color="gray")
                plt.text(
                    j,
                    i,
                    self.co_occurrence_matrix[i, j].item(),
                    ha="center",
                    va="top",
                    color="gray",
                )
        plt.axis("off")
        plt.show()

    def _calculate_co_occurrences(
        words: list[str],
    ) -> tuple[Tensor, dict[int, str], dict[str, int]]:
        """Returns the co-occurrence matrix and the conversion dicts for convenience."""
        chars = sorted(list(set("".join(words))))
        string_to_integer = {
            string: integer + 1 for integer, string in enumerate(chars)
        }
        string_to_integer["."] = 0

        integer_to_string = {
            integer: string for string, integer in string_to_integer.items()
        }

        co_occurrence_matrix = torch.zeros(
            (len(chars) + 1, len(chars) + 1), dtype=torch.int32
        )

        for word in words:
            tokens = ["."] + list(word) + ["."]

            for token1, token2 in zip(tokens, tokens[1:]):
                idx1, idx2 = string_to_integer[token1], string_to_integer[token2]
                co_occurrence_matrix[idx1, idx2] += 1

        return co_occurrence_matrix, integer_to_string, string_to_integer

    def _get_probability_matrix(co_occurrence_matrix: Tensor) -> Tensor:
        """``co_occurrence_matrix`` contains the unnormalized probabilities (counts).
        This function returns a ``probability_matrix``, which is a row-wise normalization
        of the ``co_occurrence_matrix``, thereby it contains the conditional probabilities
        p(next character | current character). The row index corresponds
        to the current character, the column index corresponds to the next character.

        """
        return co_occurrence_matrix / co_occurrence_matrix.sum(dim=1, keepdims=True)

    def generate_names(self, num_names: int) -> list[str]:
        """Generates novel names based on the conditional probabilities
        p(next character | current character).

        Args:
            num_names: Number of names to generate.
            probability_matrix: The probability matrix, which is a row-wise normalization of
                the co-occurrence matrix. It contains the conditional probabilities
                p(next character | current character).
            integer_to_string: Dictionary that translates between the indices of the tokens
                and the tokens themselves.

        Returns:
            A list of generated names.

        """
        g_cpu = torch.Generator().manual_seed(2147483647)

        out_list = []

        for _ in range(num_names):
            gen_list = []
            idx = 0
            while True:
                unnormalized_probability_vector = self.probability_matrix[idx]
                idx = torch.multinomial(
                    input=unnormalized_probability_vector,
                    num_samples=1,
                    replacement=True,
                    generator=g_cpu,
                ).item()

                if idx == 0:
                    break
                gen_list.append(self.integer_to_string[idx])

            out_list.append("".join(gen_list))

        return out_list


class LearnedBigram:
    def __init__(self, words: list[str]) -> None:
        self.xs, self.ys, self.num_tokens = self._get_dataset(words)
        self.g = torch.Generator().manual_seed(2147483647)
        self.W = torch.randn(
            self.num_tokens, self.num_tokens, generator=self.g, requires_grad=True
        )

        self._train()

    def _train(self, num_epochs: int = 500) -> None:
        for epoch in range(num_epochs):
            # Forward pass
            xenc = F.one_hot(self.xs, num_classes=self.num_tokens).float()
            logits = xenc @ self.W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdims=True)
            loss = (
                -probs[torch.arange(len(self.xs)), self.ys].log().mean()
                + 0.01 * (self.W**2).mean()
            )
            print(f"Loss at epoch {epoch}: {loss.item()}")

            # Backward pass
            self.W.grad = None
            loss.backward()

            # Update
            learning_rate = 50  # !!!
            self.W.data -= learning_rate * self.W.grad

    def _get_dataset(words: list[str]) -> tuple[Tensor, Tensor, int]:
        chars = sorted(list(set("".join(words))))
        string_to_integer = {
            string: integer + 1 for integer, string in enumerate(chars)
        }
        string_to_integer["."] = 0

        xs, ys = [], []

        for word in words:
            tokens = ["."] + list(word) + ["."]

            for token1, token2 in zip(tokens, tokens[1:]):
                idx1, idx2 = string_to_integer[token1], string_to_integer[token2]

                xs.append(idx1)
                ys.append(idx2)

        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        return xs, ys, len(string_to_integer)

    def generate_names(self, num_names: int) -> list[str]:
        """Generates novel names based on the conditional probabilities
        p(next character | current character).

        Args:
            num_names: Number of names to generate.

        Returns:
            A list of generated names.

        """
        g_cpu = torch.Generator().manual_seed(2147483647)

        out_list = []

        for _ in range(num_names):
            gen_list = []
            idx = 0
            while True:
                xenc = F.one_hot(torch.tensor([idx]), num_classes=self.num_tokens).float()
                logits = xenc @ self.W
                counts = logits.exp()
                prob = counts / counts.sum(dim=1, keepdims=True)

                idx = torch.multinomial(prob, num_samples=1, replacement=True, generator=g_cpu).item()

                if idx == 0:
                    break
                gen_list.append(self.integer_to_string[idx])

            out_list.append("".join(gen_list))

        return out_list
