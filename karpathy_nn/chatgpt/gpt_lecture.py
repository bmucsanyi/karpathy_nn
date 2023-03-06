from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from karpathy_nn.makemore.data.load_data import load_shakespeare

# Hyperparameters

# Original setting
# batch_size = 32  # How many independent sequences will we process in parallel?
# block_size = 8  # What is the maximum context length for predictions?
# max_iters = 5000  # Increased because the learning rate is lower
# eval_interval = 300
# learning_rate = 1e-3  # Self-attention can't tolerate very high learning rates
# device = "cuda" if torch.cuda.is_available() else "cpu"  # Run on GPU if you have one
# eval_iters = 200
# dim_embedding = 32
# num_heads = 4
# num_layers = 3

# Upscaled setting -- achieves a validation loss of 1.48, which is quite a bit of an
# improvement from 2.06, just by scaling up the neural net.
# Takes 15 minutes on an A100. It is much-much more English-like, but still non-sensical
# if we read it. This is, of course, just a "small" Transformer trained on the character
# level for 1 million characters of Shakespeare. It blabbers in a Shakespeare-like
# manner, but doesn't make sense at this scale.
batch_size = 64  # How many independent sequences will we process in parallel?

# Crucial to choose it large for proper English generation. Otherwise, the network
# simply cannot know what would be a sensible next word. Now we have 256 characters of
# context to predict the 257th character.
block_size = 256  # What is the maximum context length for predictions?
max_iters = 5000  # Increased because the learning rate is lower
eval_interval = 500

# We brought down the learning rate a little bit because our neural network is now much
# bigger.
learning_rate = 3e-4  # Self-attention can't tolerate very high learning rates
device = "cuda" if torch.cuda.is_available() else "cpu"  # Run on GPU if you have one
eval_iters = 200

# Every head is 64-dimensional, as opposed to 8 we had previously.
# This is a standard.
dim_embedding = 384
num_heads = 6

# At every forward pass, there is a 20% chance for each neuron to be turned off at
# respective parts of the network.
num_layers = 6
dropout = 0.2
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

        # "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".
        # It randomly shuts off some subset of neurons in the forward pass (so also
        # in the backward pass because there won't be gradient through these neurons).
        # As the mask for dropping out neurons is changed for every single forward pass,
        # we end up training an ensemble of sub-networks. At test time, everything is
        # fully enabled and all of the sub-networks are merged into a single model.
        # It's just used as a regularization technique. Andrej added it because we scale
        # up the model quite a bit and he was concerned about overfitting.

        # Use dropout when calculating the affinities, after the softmax.
        # We can completely zero out the contribution of a random previous token.
        # This randomly prevents a node to communicate with the current node.
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

        # Needed for skip-connections to project back to the input dimensionality
        # (i.e., into the residual pathway).
        # (B, T, num_heads * head_size) -> (B, T, dim_embedding)
        self.proj = nn.Linear(head_size * num_heads, dim_embedding)

        # We add dropout right before the connection back into the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Concatenate over channel dimension
        out = torch.cat(
            [head(x) for head in self.heads], dim=-1
        )  # (B, T, num_heads * head_size)
        out = self.dropout(self.proj(out))  # (B, T, dim_embedding)
        return out


# Per-node-level computation: feed-forward MLP.
# Note that not only are the batches processed separately,
# but also the tokens. Only the self-attention communicates
# across tokens. Once the tokens gathered all the data,
# they think about that data individually. This is
# what FeedForward is doing.
class FeedForward(Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, dim_embedding: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Grow the feed-forward block that's on the side of the residual pathway
            nn.Linear(dim_embedding, 4 * dim_embedding),
            nn.ReLU(),
            # Projection going back to the residual pathway
            nn.Linear(4 * dim_embedding, dim_embedding),
            # We add dropout right before the connection back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# This is what's replicated in the Transformer N times.
# The only difference is the cross-attention part.
# The Block intersperses communication and computation.
# The communication is done using multi-headed self-attention,
# and the computation is done using a per-token feed-forward network.
class Block(Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, dim_embedding: int, num_heads: int) -> None:
        # dim_embedding: embedding dimension
        # num_heads: the number of heads we'd like -- the group size in group convolution
        super().__init__()
        head_size = dim_embedding // num_heads  # 32 // 4 = 8
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(dim_embedding)
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

    def forward(self, x: Tensor) -> Tensor:
        # We're interspersing communication and feed-forward many-many times.
        # This doesn't give a very good result alone.
        # The reason is that we're starting to get a pretty deep neural network.
        # Deep neural networks suffer from optimization issues in the vanilla case.
        # This is what we're slightly starting to run into.
        # Luckily, the Transformer paper resolves these difficulties.
        # x = self.self_attention(x)
        # x = self.feed_forward(x)
        # return x

        # There are two optimizations that dramatically help with the depth of these
        # networks and make sure that the networks remain optimizable.
        # (1): Skip-connections (residual connections). They come from the
        # "Deep Residual Learning for Image Recognition" ResNet paper from 2015.
        # This sets up a residual pathway in the computational graph that is forked
        # from the complicated transformations we perform on the input.
        # The two pathways are merged via addition.
        # We know that addition distributes the incoming gradient equally to both
        # of its branches. The gradients of the loss "hop" through every addition
        # node all the way to the forked input *directly*. They also go through the
        # other complicated branch. The gradient "super-highway" goes from the
        # supervision all the way to the input unimpeded.
        # The complicated branch is usually initialized in the beginning such that
        # they contribute very-very little (if anything) to the residual pathway.
        # It's almost like they're not there. During the optimization, they come online
        # over time and start to contribute. But at initialization, we can go directly
        # from supervision to input with an unimpeded gradient. That dramatically helps
        # with the optimization.

        # Fork off, do some computation, come back -- residual connection
        # x = x + self.self_attention(x)

        # Fork off, do some computation, come back -- residual connection
        # x = x + self.feed_forward(x)

        # (2): LayerNorm. It's implemented in PyTorch, it's a paper that came out in
        # 2016.
        # Pseudocode:
        # Layer normalization pseudocode
        # def layer_norm(x, gamma, beta, epsilon):
        #     # Calculate layer statistics
        #     layer_mean = mean(x, axis=1, keepdims=True)
        #     layer_std = std(x, axis=1, keepdims=True)
        #     # Normalize layer
        #     x_hat = (x - layer_mean) / (layer_std + epsilon)
        #     # Scale and shift
        #     out = gamma * x_hat + beta
        #     return out
        # BatchNorm made sure that across the batch dimension, any individual neuron's
        # pre-activations had a standard Gaussian distribution.
        # It normalizes across the rows of the pre-activations, for each output neuron
        # independently. The rows are not normalized: we are normalizing the columns
        # to 0 mean and unit variance.
        # LayerNorm simply normalizes the rows instead of the columns. I.e., for each
        # sample independently, the pre-activations of the output neuron follow a
        # standard Gaussian distribution. We simply have to change dim=0 to dim=1
        # in our BatchNorm implementation. Now the columns are not normalized anymore.
        # Because our computation now doesn't span across examples, we can delete
        # all of the buffer stuff, as we can always apply it, even for a single example.
        # We also have no distinction between training and test time.
        # It only has to keep track of gamma and beta: these parameters scale and
        # offset on the feature dimension, just like in BatchNorm.

        # Very few details about the Transformer have changed since it has been released,
        # but LayerNorm is a part that slightly departs from the original paper.
        # In the paper, Add & Norm is applied *after* the transformation
        # (communication or feed-forward). But now it's more common to apply the
        # LayerNorm *before* the transforation. This is called the pre-norm formulation,
        # and that's the one we are going to implement as well.

        # Layer Norm is applied immediately on x, before the transformation.
        # The E and Var are calculated over the dim_embedding = 32 dimensions.
        # The B and T dimensions act as batch dimensions for the transformation --
        # for each *token* in each sample, this is calculated separately and
        # independently. This makes each token a sample from a standard Gaussian at the
        # beginning of training. However, we have the trainable gamma and beta parameters
        # that allow the LayerNorm to output features that are not standard Gaussian.
        # We need two separate layer norm because they have trainable parameters.
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


# Super simple bigram model
class GPTLanguageModel(nn.Module):
    # We're not going to implement an encoder in this case. (We don't have
    # cross-attentions to an encoder.)
    def __init__(self) -> None:
        super().__init__()
        # Embedding table for 32D embeddings for each token
        self.token_embedding_table = nn.Embedding(num_tokens, dim_embedding)

        # Each position from 0 to block_size - 1 will get its own embedding vector
        self.position_embedding_table = nn.Embedding(block_size, dim_embedding)

        # I.e., 4 heads of 8-dimensional self-attention.
        # Instead of having only one communication channel with head_size = 32,
        # we have four communication channels in parallel and each one of these
        # are smaller correspondingly. As we have four commmunication channels,
        # we want 8D self-attention. This gives us a 32D concatenated representation.
        # This is kind of like a group convolution (e.g., in AlexNet).
        # Instead of having one large convolution, we do convolution in groups, which is
        # synonimous to multi-headed self-attention.
        # It helps to have multiple independent communication channels because the tokens
        # have a lot to talk about. :) (2.4 -> 2.27)
        # E.g., they might want to find consonants, wovels, wovels just from certain
        # positions, etc. Gathering lots of different outputs and concatenating them
        # helps a lot.
        # self.self_attention_heads = MultiHeadAttention(4, dim_embedding // 4)
        self.language_modeling_head = nn.Linear(dim_embedding, num_tokens)

        # self.feed_forward = FeedForward(dim_embedding=dim_embedding)

        # We create sequential applications of blocks.

        # (2.27 -> 2.09 with the skip-connections and hidden increase)
        # Our network is starting to be big enough that our train loss is getting ahead
        # of the validation loss, so we're starting to see a little bit of overfitting.
        # Now we start to see some sensible words in the generated text.
        # It's starting to look like English. :)

        # (2.09 -> 2.06 with the LayerNorm additions)
        # They would help even more if we had a bigger and deeper network.
        # At this stage we have a pretty complete Transformer according to the original
        # paper. It's a decoder-only Transformer. The major pieces are in place, so
        # we can try simply scaling it up.
        self.blocks = nn.Sequential(
            *[
                Block(dim_embedding=dim_embedding, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        # There should typically be a LayerNorm here too, at the end of
        # the Transformer and right before the final linear layer
        # that decodes into the vocabulary.
        self.final_layer_norm = nn.LayerNorm(dim_embedding)

    def forward(
        self, idx: Tensor, targets: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        # So far we've taken the indices and we've encoded them based on the identity
        # of the tokens inside idx.
        # The thing that people often do is to not just encode the identity of the tokens
        # but also their position.

        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_embeddings = self.token_embedding_table(idx)  # (B, T, dim_embedding)
        positional_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)  # 0 .. T - 1
        )  # (T, dim_embedding)
        x = token_embeddings + positional_embeddings  # (B, T, dim_embedding)

        # The simplest way to plug in a (decoder) self-attention head.
        # Apply multi-headed self-attention.
        # x = self.self_attention_heads(x)  # (B, T, dim_embedding)

        # The below was only true for the bigram model:
        # x not only holds the token identities, but also the positions at which these
        # tokens occur. This is currently not that useful (because we simply have a
        # bigram model, it doesn't matter whether a token is at the 5th or 1st position),
        # but in the self-attention block it will start to matter.

        # Previously, the multi-headed self-attention did the communication between
        # the nodes, but we went way too fast to calculating the logits (with just a
        # single linear layer). So the tokens looked at each other, but didn't really
        # have a lot of time to think about what they have found from the other tokens.
        # This FeedForward object helps "thinking".
        # It is applied after we self-attend. (2.27 -> 2.24)
        # x = self.feed_forward(x)  # (B, T, dim_embedding)

        x = self.blocks(x)  # (B, T, dim_embedding)

        # We will now intersperse communication with computation. That's also what
        # the Transformer does: the communication-computation blocks are replicated
        # N times.

        # x goes into the decoder language modeling head to create the logits
        logits = self.language_modeling_head(x)  # (B, T, num_tokens)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        # Because we're using positional embeddings, we can never have more than
        # block_size inputs coming in, as if idx >= block_size then the
        # position_embedding_table is going to run out of scope, as it only has
        # embeddings for up to block_size.

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + k)
        return idx


model = GPTLanguageModel()
model = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
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
