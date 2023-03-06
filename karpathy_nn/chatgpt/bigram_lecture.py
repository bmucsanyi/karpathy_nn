from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from karpathy_nn.makemore.data.load_data import load_shakespeare

# Hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8  # What is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"  # Run on GPU if you have one
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = load_shakespeare()

# Here are all the unique characters that occur in this text
characters = sorted(list(set(text)))
num_tokens = len(characters)
# Create a mapping from characters to integers
string_to_integer = {character: integer for integer, character in enumerate(characters)}
integer_to_string = {integer: character for integer, character in enumerate(characters)}

encode = lambda string: [
    string_to_integer[character] for character in string
]  # Encoder: take a string, output a list of integers
decode = lambda int_list: "".join(
    [integer_to_string[integer] for integer in int_list]
)  # Decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # First 90% will be train, the rest will be val
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split: str) -> tuple[Tensor, Tensor]:
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# PyTorch doesn't build a computational graph, can be much more efficient
@torch.no_grad()
def estimate_loss() -> dict[str, Tensor]:
    out = {}
    model.eval()  # Right now this doesn't do anything
    # If we had, e.g., BatchNorm or Dropout, this would make a difference.
    # It's always good to switch between the modes.
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  # Much less noisy than a single batch
    model.train()
    return out


# Super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, num_tokens: int) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_tokens, num_tokens)

    def forward(
        self, idx: Tensor, targets: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + k)
        return idx


model = BigramLanguageModel(num_tokens)
model = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
