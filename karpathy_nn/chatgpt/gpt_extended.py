import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch import Tensor

from typing import Optional

from karpathy_nn.makemore.data.load_data import load_shakespeare

# Hyperparameters
batch_size = 64  # How many independent sequences will we process in parallel?
block_size = 256  # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
dim_embedding = 384
num_heads = 6
num_layers = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

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
n = int(0.9 * len(data))  # First 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split: str) -> tuple[Tensor, Tensor]:
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss() -> dict[str, Tensor]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(Module):
    """One head of self-attention."""

    def __init__(self, head_size: int) -> None:
        super().__init__()

        # Typically people don't use biases
        # These are the linear transformations we are going to apply to all our nodes
        self.key = nn.Linear(dim_embedding, head_size, bias=False)
        self.query = nn.Linear(dim_embedding, head_size, bias=False)
        self.value = nn.Linear(dim_embedding, head_size, bias=False)

        # tril is not a parameter of the model. In PyTorch naming conventions, this
        # is called a buffer. We have to assign it to the module using register_buffer.
        # If we have parameters in your model which should be saved and restored in the
        # state_dict but not trained by the optimizer, we should register them as
        # buffers. Buffers won't be returned in model.parameters(), so that the optimizer
        # won't update them.
        # Another reason is that all buffers and parameters will be pushed to the device,
        # if called on the parent model. (model.cuda())
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Input of size (batch, time-step, channels)
        # Output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Compute attention scores ("affinities")
        weights = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # Use scaled self-attention
        )  # (B, T, head_size) @ (B, head_size, T) = (B, T, T)

        # This makes sure that the future doesn't communicate to the past.
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")  # Makes it a decoder block
        )  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B,T, head_size)

        # Aggregate the values
        out = weights @ v  # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out


class MultiHeadAttention(Module):
    """Implements multiple heads of self-attention in parallel."""

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, dim_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, dim_embedding: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_embedding, 4 * dim_embedding),
            nn.ReLU(),
            nn.Linear(4 * dim_embedding, dim_embedding),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, dim_embedding: int, num_heads: int) -> None:
        # dim_embedding: embedding dimension
        # num_heads: the number of heads we'd like
        super().__init__()
        head_size = dim_embedding // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(dim_embedding)
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class GPTLanguageModel(Module):
    def __init__(self) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_tokens, dim_embedding)
        self.position_embedding_table = nn.Embedding(block_size, dim_embedding)
        self.blocks = nn.Sequential(
            *[Block(dim_embedding, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(dim_embedding)  # final layer norm
        self.lm_head = nn.Linear(dim_embedding, num_tokens)

        # Better init, not covered in the original GPT video but important,
        # will be covered in a follow-up video.
        self.apply(self._init_weights)

    def _init_weights(self, module: Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_embedding = self.token_embedding_table(idx)  # (B, T, C)
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = token_embedding + positional_embedding  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, num_tokens)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(parameter.numel() for parameter in model.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist())).close()
