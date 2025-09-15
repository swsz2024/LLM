import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 4  
context_length = 16  
d_model = 64  

num_blocks = 8  
num_heads = 4  
learning_rate = 1e-3  
dropout = 0.1  
max_iters = 5000  # epoch
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Load training data
with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Using TikToken (Same as GPT3) to tokenize the source text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1  
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  

# Split train and validation
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]


# Define Feed Forward Network
# FFN：先升维，再降维，再dropout
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# Define Scaled Dot Product Attention
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # Lower triangular mask
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Time steps(current context_length), Channels(dimensions)
        assert T <= self.context_length
        assert C == self.d_model
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        # Scaled dot product attention: Q @ K^T / sqrt(d_k)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply masked attention
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 按行做softmax
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)

        # Apply dot product attention: weights @ V
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # concat拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks  # transformer blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # Set up token embedding look-up table
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model)

        # Run all the transformer blocks
        # Different from original paper, here we add a final layer norm after all the blocks
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        """
        Set up position embedding look-up table
        following the same approach as the original Transformer paper (Sin and Cos functions)
        """
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        # position需要的参数：列（按列的奇和偶分配不同的函数）、行（行数的顺序代表每个token的顺序）
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)  # 行，每个token
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # 列
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)

        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # The "logits" are the output values of our model before applying softmax
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            # 训练
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)  # farward前向传播
            # 这里需要注意：logits维度为(Batch,Squence,class) class是词汇表的长度
            # 取squence=-1即取句子的最后一个词，预测下一个词
            logits_last_timestep = logits[:, -1, :]

            # logits_last_timestep维度为(Batch,class) ，即每句话的最后一个词的预测概率
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            # 这里用torch.multinomial是为了增加多样性，进行概率采样预测
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Initialize the model
model = TransformerLanguageModel()
model = model.to(device)


# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model state dictionary
torch.save(model.state_dict(), 'model-ckpt.pt')

# Generate
model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')
