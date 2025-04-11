from networkx import number_weakly_connected_components
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=64, num_heads=8):
    super().__init__()
    self.num_heads = num_heads
    self.d_head = d_model // num_heads

    self.W_qkv = nn.Linear(d_model, 3 * d_model)
    self.out = nn.Linear(d_model, d_model)

  def forward(self, x):
    B, T, C = x.shape
    qkv = self.W_qkv(x)

    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
    k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
    v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
    attn_weights = torch.softmax(attn_scores, dim=-1)

    out = attn_weights @ v
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.out(out)


class PositionalEncoding(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    position = torch.arange(512).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(1, 512, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe)

  def forward(self, x):
    x = x + self.pe[:, : x.size(1)]
    return x


class FeedForward(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
    )

  def forward(self, x):
    return self.net(x)


class TransformerBlock(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.attn = MultiHeadAttention(d_model, num_heads)
    self.ffn = FeedForward(d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.ffn(self.norm2(x))
    return x


class MiniTransformer(nn.Module):
  def __init__(self, vocab_size, d_model=64, num_layers=3, num_heads=4):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, d_model)
    self.pos_enc = PositionalEncoding(d_model)
    self.blocks = nn.Sequential(
      *[TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
    )
    self.fc = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    x = self.embed(x)
    x = self.pos_enc(x)
    x = self.blocks(x)
    logits = self.fc(x)
    return logits


# text = "hello world"
# vocab = sorted(list(set(text)))
# vocab_size = len(vocab)
#
# stoi = {ch: i for i, ch in enumerate(vocab)}
#
# inputs = [stoi[ch] for ch in text[:-1]]
# targets = [stoi[ch] for ch in text[1:]]
#
# model = MiniTransformer(vocab_size)
# optimizer = torch.optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()
#
# for epoch in range(100):
#   logits = model(torch.tensor([inputs]))
#   loss = criterion(logits.view(-1, vocab_size), torch.tensor(targets))
#
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#
#   if epoch % 10 == 0:
#     print(f"Epoch {epoch}, Loss:{loss.item():.4f}")
