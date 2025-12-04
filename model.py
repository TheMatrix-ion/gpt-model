import os
import torch
import torch.nn as nn
from torch.nn import functional as F


os.chdir('E:/tools/Microsoft VS Code/workspace')

# hyperparameters
block_size = 8  # 每次看多少个字
batch_size = 64 # 一批训练几组
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-2
max_iters = 3000
eval_interval = 300

# read data
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 字符级编码
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 创建idx到字符的映射
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]          # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# splits ds
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # 
    data = train_data if split == 'train' else val_data
    # 随机起点的范围必须在 0 到 数据总长度 - 窗口大小 之间
    # size 是batch size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 从data中拿到block size个连续的数据，并且堆叠起来，构造一个x，同理y
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # 评估阶段无需梯度
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 继承简单的二元语言模型
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding：是一个巨大的数字表格，当你给它一个索引（比如数字 5），它就去表格的第 5 行，把那一行所有的数字拿出来给你
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            # 构造符合cross_entropy的输入
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 调用 forward：把当前所有的上下文 idx 扔给模型
            logits, loss = self(idx)
            # 仅仅关注最后一个时间步
            logits = logits[:, -1, :]
            # softmax
            probs = F.softmax(logits, dim=-1)
            # torch.multinomial 按照概率进行抽样。概率大的字容易被抽中，但概率小的字也有机会
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))       
