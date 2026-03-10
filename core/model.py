"""
model.py - 定义模型结构和前向传播逻辑

实现一个简化版的 Decoder-only Transformer (GPT 架构)
参数量: 1~5M (可配置)

Q:
    1. 为什么需要 Masked Attention?
    2. Multi-Head 的作用是什么?
    3. 残差连接为什么能解决梯度消失?
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# 从config模块导入全局配置
from config.config import block_size, n_embd, n_head, n_layer, dropout, device


### 单头自注意力机制 (Head)
class Head(nn.Module):
    """
    ### 单头自注意力机制 (Single Head of Self-Attention)

    Q: 为什么需要 tril 掩码?
    A: 防止模型在训练时 “偷看” 未来的 token，保证因果性 (Causal Language Modeling)  

    Q: 为什么除以 head_size ** -0.5?
    A: 缩放点击注意力 (Scaled Dot-Product), 防止 Softmax 梯度消失
    """
    
    def __init__(self, head_size: int): 
        super().__init__()
        # 线性变换: 输入 -> 查询、键、值 (Q K V)
        """
            Q: 搜索的问题 -> "我想找什么"
            K: 每本书的标签/索引 -> "这本书是关于什么的"
            V: 书的实际内容 -> "这本书具体写了什么"

            注意力计算过程:
                1. 你用 Q (搜索问题) 去匹配所有书的 K (标签)
                2. 计算匹配度分数 -> 哪些书最相关
                3. 根据匹配度, 加权读取 V (书的内容), 得到最终答案
            
            # 输入 x: [B, T, C] = [批次, 序列长度, 嵌入维度]
            # 例如: [64, 256, 128] = 64 个句子, 每个句子 256 词, 每个词 128 维向量

            # n_embd (Embedding Dimension)
                每个 token 被表示为一个 n_embd 维的向量 (例如 128 维)
                例如: n_embd = 128,
                "cat" -> [0.1, 0.3, ..., 0.05] (128 维向量)
                作用：控制模型的表达能力和参数量，过小可能欠拟合, 模型越弱，但训练速度快，过大可能过拟合, 模型越强，但训练慢，参数量大
            
            # head_size (每个注意力头的维度)
                多头注意力将 n_embd 分成多个头，每个头处理一部分信息
                例如 n_embd = 128, n_head = 4, 则 head_size = n_embd // n_head = 32

                每个头处理32维度，4个并行处理，最后拼接回128维度
                作用：允许模型从不同的子空间学习不同类型的关系，增强模型的表达能力
                    - 头1 可能专注于语法关系 (主谓宾)
                    - 头2 可能专注于语义关系 (同义词、上下位词)
                    - 头3 可能专注于长距离依赖 (跨句子关系)
                    - 头4 可能专注于局部关系 (相邻词的关系)

            # bias=False: 因为我们不需要偏置项，注意力机制主要依赖于输入的线性变换结果，添加偏置可能会引入不必要的参数和计算
                为什么Transformer的线性层通常不使用偏置项?
                A:  因为Transformer中的线性层主要用于生成查询、键和值，这些线性变换的输出会被后续的注意力机制处理，添加偏置项可能会引入不必要的参数和计算，且在实践中并没有显著提升性能，因此通常设置 bias=False 来简化模型。
                    - LayerNorm 已经提供了偏移能力
                    - 减少参数量
                    - 实验发现效果相当或更好
        """
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 注册因果掩码 (buffer 不会被优化器更新)
        # tril: 下三角矩阵，确保位置 i 只能看到位置 <= i 的信息
        self.tril: torch.Tensor  # 占位符，实际值在 __init__ 中注册
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自注意力机制(self-Attention)
        
        句子: "I love AI" (3个Token)
        B=1(1个样本), T=3(序列长度, 3个时间步, Token=3), C=4(嵌入维度, 每个Token被表示为4维向量)
        """
        # 获取输入维度
        # x.shape = [1, 3, 4]
        # B=1(batch), T=3(序列长度), C=4(每个token的向量维度)
        B, T, C = x.shape
        # 计算Q, K, V
        """
            Q: 为什么需要 Q、K、V 三个不同变换?
            A: 让模型学习不同的表示空间，Q 用于查询，K 用于匹配，V 用于输出。分开变换增加模型的表达能力。
                - Q (查询): 代表当前 token 的信息，告诉模型我们想要什么样的信息 “我在找什么信息”
                - K (键): 代表所有 token 的信息，告诉模型每个 token 有什么样的信息 “我有什么信息”
                - V (值): 代表所有 token 的实际内容，告诉模型每个 token 具体是什么 “我的实际内容是什么”
                类比数据库查询：Q是搜索词，K是索引，V是实际数据
        """
        k = self.key(x)     # [1, 3, 4] 通过线性层变换
        q = self.query(x)   # [1, 3, 4] 通过线性层变换
        v = self.value(x)   # [1, 3, 4] 通过线性层变换

        # 计算注意力分数
        """
        q.shape = [1, 3, 4] (batch, seq_len, head_size), k.transpose(-2, -1).shape = [1, 4, 3]
        wei.shape = [1, 3, 3]
        规则: 矩阵乘法 [M, N] @ [N, P] -> [M, P]
        A = [[1, 2, 3],
             [4, 5, 6]]  # A.shape = [2, 3]
        B = [[7, 8],
             [9, 10],
             [11, 12]]  # B.shape = [3, 2]
        A @ B = [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
                 [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]  # 结果.shape = [2, 2]

        PyTorch中，当张量维度≥3时，@运算会：
            1. 保持前面的batch维度不变(要求匹配或可广播)
            2. 对最后两个维度执行矩阵乘法
            q = torch.randn(1, 3, 4)
            k = torch.randn(1, 3, 4)

            转置
            k_T = k.transpose(-2, -1)  # [1, 4, 3]
            wei = q @ k_T  # [1, 3, 4] @ [1, 4, 3] -> [1, 3, 3]

        物理意义: 计算每个 token 与其它token的相关性
        wei[0, i, j] = token i 对 token j 的关注程度

            I         love          AI
           
    I      [0.8        0.5         0.2]  <- "I" 最关注自己, 其次是 "love", 最不关注 "AI"
    love   [0.3        0.9         0.4]  <- "love" 最关注自己, 其次是 "AI", 最不关注 "I"
    AI     [0.1        0.6         0.8]  <- "AI" 最关注自己, 其次是 "love", 最不关注 "I"
        
            Q: 为什么要对注意力分数进行缩放 (C ** -0.5)?
            A: 缩放点击(Scaled Dot-Product)
               - 当C很大时， q@k的值会很大，softmax梯度会消失
               - 除以sqrt(C)，保持方差稳定，训练更稳定
        """
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        """
        应用因果掩码 (Causal Masking) 防止偷看未来信息
        tril 是下三角矩阵
        [[1, 0, 0],       [1,      0, 0]
         [1, 1, 0],  ->   [-inf,   1, 0]
         [1, 1, 1]]       [-inf, -inf, 1]

        # 掩码后
                I      love  AI
         I     [0.8,  -inf, -inf]
         love  [0.3,   0.9,  -inf]
         AI    [0.1,   0.6,   0.8]

            Q: 为什么推理时不需要掩码?
            A: 推理是逐步生成的，每次只生成一个 token，模型只能看到已经生成的 token，因此不需要掩码。
        """
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        """
        -inf 经过 softmax 变成 0，其它值归一化为概率（每行和为1）
        例如:
               I       love   AI
        I    [[0.8,  -inf, -inf],       [[0.999, 0, 0],
        love [0.3,   0.9,  -inf],  ->   [0.047, 0.953, 0],
        AI   [0.1,   0.6,   0.8]]       [0.015, 0.665, 0.32]]

         Q: 为什么要在注意力权重上应用 Dropout?
         A: 增强模型的泛化能力，防止过拟合。Dropout 会随机丢弃一些注意力连接，让模型学会不依赖特定的 token。
        """
        wei = F.softmax(wei, dim=-1)
        """
        Dropout 正则化
        训练时随机丢弃一些注意力连接，增强模型的鲁棒性和泛化能力，防止过拟合
        推理时dropout自动关闭
        """
        wei = self.dropout(wei)
        """
        wei.shape = [1, 3, 3], v.shape = [1, 3, 4]
        out.shape = [1, 3, 4]

        物理意义：每个Token的输出 = 所有token的value的加权和，权重由注意力分数决定
        out[0] = 1.0 * v[0] + 0.0 * v[1] + 0.0 * v[2]  <- "I"的输出
        out[1] = 0.4 * v[0] + 0.6 * v[1] + 0.0 * v[2]  <- "love"的输出
        out[2] = 0.2 * v[0] + 0.3 * v[1] + 0.5 * v[2]  <- "AI"的输出
        """
        out = wei @ v
        return out
    

### 多头注意力机制 (Multi-Head Attention)
class MultiHeadAttention(nn.Module):
    """
    ### 多头注意力机制 (Multi-Head Attention)

    Q: 为什么要使用多头注意力?
    A: 允许模型从不同的子空间学习不同类型的关系，增强模型的表达能力。
        - 头1 可能专注于语法关系 (主谓宾)
        - 头2 可能专注于语义关系 (同义词、上下位词)
        - 头3 可能专注于长距离依赖 (跨句子关系)
        - 头4 可能专注于局部关系 (相邻词的关系)
    """
    def __init__(self, num_heads: int, head_size: int):

        """
        输入x: [B, T, n_embd] = [批次, 序列长度, 嵌入维度] = [64, 256, 128]
        n_head = 4, head_size = n_embd // n_head = 32

        step 1. 列表推导式
        [h(x) for h in self.heads]
        结果: [head0_out, head1_out, head2_out, head3_out]
        每个head_out形状: [64, 256, 32]

        step 2. torch.cat 拼接
        torch.cat([head0_out, head1_out, head2_out, head3_out], dim=-1)
        在最后一个维度拼接: 32 + 32 + 32 + 32 = 128
        结果: [B, T, n_embd] = [64, 256, 128]

        step 3. 线性投影
        self.proj(out)
        Linear(128, 128) 形状不变 [64, 256, 128]
        作用: 让不同 Head 的信息充分融合

        step 4. Dropout
        self.dropout(out)
            训练时随机丢弃一些连接，增强模型的鲁棒性和泛化能力，防止过拟合
            推理时dropout自动关闭
        """
        super().__init__()
        # 创建多个 Head 实例
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 投影层 将多头输出拼接后投影回 n_embd 维度
        self.proj = nn.Linear(n_embd, n_embd)
        # Dropout 正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)


    """
    参数量计算:
        假设n_embd=128, n_head=4, head_size=32
        每个Head的参数量:
            - Key: 128 -> 32 = 128 * 32 = 4096
            - Query: 128 -> 32 = 128 * 32 = 4096
            - Value: 128 -> 32 = 128 * 32 = 4096
            单个Head总参数量 = 4096 * 3 = 12288
        4个Head总参数量 = 12288 * 4 = 49152
        投影层参数量:
            - 输入维度: n_embd = 128
            - 输出维度: n_embd = 128
            参数量 = 128 * 128 = 16384
        总计 = 49152 + 16384 = 65536 (约 65K 参数)

        Q: 如果我想增加模型容量，应该增加n_head还是n_embd?
        A: 优先增加n_embd，因为它直接增加每个token的表示能力，增加n_head可以让模型学习更多类型的关系，但每个head的维度会变小，可能导致信息丢失。增加n_embd同时增加head_size，可以保持每个head的信息量。
            参数量与n_embd^2成正比，而与n_head只是线性增加。但n_head增加可以让模型关注更多不同特征。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行计算所有 head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 线性投影 + Dropout
        out = self.dropout(self.proj(out))
        return out
    
# 前馈网络 (Feed-Forward Network)
class FeedForward(nn.Module):
    """位置前馈网络 (Feed-Forward Network)"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            # 升维 128 -> 512
            nn.Linear(n_embd, 4 * n_embd),
            # 激活函数: 引入非线性
            nn.ReLU(),
            # 降维: 512 -> 128
            nn.Linear(4 * n_embd, n_embd),
            # 正则化 Dropout
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播: 输入 -> 升维 -> 激活 -> 降维 -> Dropout -> 输出
        return self.net(x)


# Transformer Block (包含 Multi-Head Attention 和 Feed-Forward Network)
class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

### GPT 模型主体
class GPT(nn.Module):
    """简化版 GPT 模型"""

    def __init__(self, vocab_size: int):
        """
        vocab_size = 65      # 字符级词表大小
        block_size = 256     # 上下文长度
        n_embd = 128         # 嵌入维度
        n_head = 4           # 注意力头数
        n_layer = 4          # Transformer 层数
        """
        super().__init__()
        """
        输入: [2, 10]               输出: [2, 10, 128]
        ┌──────────┐               ┌───────────────────┐
        │ 46 43 50 │               │ [v₁] [v₂] [v₃]... │
        │ 12 34 56 │   ──Embed──→  │ [v₁] [v₂] [v₃]... │
        └──────────┘               └───────────────────┘
           token IDs                  稠密向量

        参数量: 65 * 128 = 8320 (约 8K 参数)
        将离散的token ID映射为 连续的稠密向量
        """
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        """
        输入： [T] 的位置索引 (0, 1, 2, ..., block_size-1)
        输出： [T, n_embd] 的位置嵌入向量 [T, 128]
        参数量: block_size * n_embd = 256 * 128 = 32768 (约 32K 参数)
        作用：注入位置信息(Transformer本身没有序列顺序概念)，让模型知道每个token在序列中的位置

            Q: 为什么要位置编码?
            A: 自注意力机制是排列不变的(Permutation Invariant),
                猫吃鱼 和 鱼吃猫 的注意力计算结果相同
                所以要位置编码区分:
                    token_emb[0] + pos_emb[0] -> "猫" 在啊位置0
                    token_emb[1] + pos_emb[1] -> "吃" 在位置1

            Q: 为什么工业模型 (如 Llama， Qwen) 不用这种学习式位置编码?
            A: 学习式位置编码无法外推到比训练时更长的序列，RoPE(旋转位置编码)可以支持长度外推。

            Q: 位置编码可以加到每一层吗?
            A: 原始Transformer只在第一层加，但有些变体(如GPT-J)在每层都加
        """
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        """
        输入: [B, T, n_embd] -> 输出: [B, T, n_embd]  形状不变
        参数量: n_layer * (MultiHeadAttention + FeedForward) ≈ 4 * (65K + 131K) = 784K 参数
        作用: 模型的核心计算单元，提取特征
        Block 1: [B, T, 128] ──> [B, T, 128]
        Block 2: [B, T, 128] ──> [B, T, 128]
        Block 3: [B, T, 128] ──> [B, T, 128]
        Block 4: [B, T, 128] ──> [B, T, 128]

            Q: 为什么所有层的输入输出维度相同?
            A: 因为残差连接要求x + f(x)维度一致

            Q: 增加n_layer和增加n_embd哪个更有效?
            A: 取决于任务。深层网络提取更抽象特征，宽层网络捕获更多信息。通常两者平衡(Scaling Law)
        """
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        """
        输入: [B, T, n_embd] -> 输出: [B, T, n_embd]  形状不变
        参数量: 2 * n_embd = 256 (约 0.25K 参数)
        作用: 对模型输出进行归一化处理，提高训练稳定性和性能

            Q: 为什么最后还需要LayerNorm?
            A: 经过4层Block后，激活值可能分布不均
                LayerNorm将其标准化为 均值=0， 方差=1
                有助于LM Head更稳定地预测
        """
        self.ln_f = nn.LayerNorm(n_embd)
        """
        输入: [B, T, n_embd] -> 输出: [B, T, 65] 每个位置对每个token的预测分数
        参数量: 128 * 65 = 8320 (约 8K 参数)
        作用: 将隐藏状态映射回词表空间，用于预测下一个token

        输入: [2, 10, 128]              输出: [2, 10, 65]
        ┌───────────────────┐           ┌─────────────────────┐
        │ [v₁] [v₂] [v₃]... │           │ [p₁] [p₂] [p₃]...   │
        │ [v₁] [v₂] [v₃]... │  ──LM──→  │ [p₁] [p₂] [p₃]...   │
        └───────────────────┘           └─────────────────────┘
        隐藏状态                        每个 token 的概率分数
        """
        self.lm_head = nn.Linear(n_embd, vocab_size)
        """
        权重初始化
        作用: 递归地对所有子模块应用 _init_weights方法
        初始化策略: 正态分布 N(0, 0.02)
        为什么重要: 好的初始化可以加速收敛, 避免梯度消失/爆炸
        """
        self.apply(self._init_weights)

        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model initialized with {params/1e6:.2f}M parameters.")
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            """
            线性层:
            - 权重: 从正态分布 N(0, 0.02) 初始化
            - 偏置: 初始化为0
             这种初始化策略在Transformer中被广泛使用，能够保持训练的稳定性
            """
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            """
            嵌入层:
            - 权重: 从正态分布 N(0, 0.02) 初始化
                这种初始化策略能够让嵌入向量在训练初期具有适当的随机性，促进模型学习更丰富的表示
                注意: 位置嵌入和 token 嵌入都使用相同的初始化策略
                也可以使用均匀分布或其他策略，但正态分布是Transformer中常见的选择
                例如: nn.init.uniform_(module.weight, a=-0.1, b=0.1) 也可以，但正态分布更常见
                这种初始化策略在Transformer中被广泛使用，能够保持训练的稳定性
                也可以使用均匀分布或其他策略，但正态分布是Transformer中常见的选择
                例如: nn.init.uniform_(module.weight, a=-0.1, b=0.1) 也可以，但正态分布更常见
                这种初始化策略在Transformer中被广泛使用，能够保持训练的稳定性
            """
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: [B, T] 输入 token IDs (Batch, Time)
            targets: [B, T] 目标 token IDs (可选, 用于计算loss)
        Returns:
            logits: [B, T, V] 或 [B*T, V] 预测分数 (V = vocab_size)
            loss: 标量或 None
        """
        
        """
        获取输入维度
            B = Batch Size (批次大小, 如64)
            T = Time Steps (序列长度, 如256)

            Q: 为什么需要记录 B 和 T?
            A: 后续reshape和位置编码需要用到
        """
        B, T = idx.shape

        """
        输入: idx [B, T]
        输出: tok_emb [B, T, n_embd]

        物理意义: 将每个token ID 映射为稠密向量
        例如: ID=5 的字符 'a' -> [0.1, -0.3, 0.8, ...] (n_embd维向量)
            Q: embedding层有多少参数
            A: vocab_size * n_embd = 65 * 128 = 8320 (约 8K 参数)
        """
        tok_emb = self.token_embedding_table(idx)

        """
        位置编码 Positional Embedding
        输入: [0, 1, 2, ..., T-1] 位置索引
        输出: pos_emb [T, n_embd]

        物理意义: 注入位置信息
        为什么需要位置编码? 没有RNN的时序性, 需要显示告诉模型"第1个词, 第2个词, ..."

            Q: 为什么这里不用 RoPE?
            A: RoPE更先进, 但实现复杂 
        """
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        
        """
        Embedding 相加
        输入: tok_emb [B, T, n_embd] + pos_emb [T, n_embd] -> 输出: x [B, T, n_embd]

        广播机制: pos_emb 会自动复制到每个batch
        物理意义: 每个token的向量 = 语义信息 + 位置信息
        """
        x = tok_emb + pos_emb  # [B, T, n_embd]

        """
        输入: x [B, T, n_embd]
        输出: x [B, T, n_embd]

        物理意义: 核心计算层
        每个block包含:
            1. Multi-Head self-Attention: 捕捉词与词的关系
            2. Feed-Forward Network: 非线性变换
            3. LayerNorm + Residual: 稳定训练

            Q: 为什么输入输出维度相同?
            A: 残差连接要求维度一致, 便于信息流动和梯度传播
        """
        x = self.blocks(x)

        """
        输入: x [B, T, n_embd]
        输出: x [B, T, n_embd]

        物理意义: 归一化, 使分布稳定，便于后续线性层预测
        Q: 为什么最后还要 LayerNorm?
        A: Pre-LN 架构的标准做法，让输出分布更稳定
        """
        x = self.ln_f(x)

        """
        输入: x [B, T, n_embd]
        输出: logits [B, T, vocab_size]

        物理意义: 将n_embd维向量映射回词表空间，预测下一个token的分数
        每个分数代表"下一个token是该词的概率" (未归一化)

            Q: logits 和 probabilities 的区别?
            A: logits 是原始分数, probabilities 是 softmax 后的概率
        """
        logits = self.lm_head(x)

        """
        计算Loss
        """
        if targets is None:
            # 推理模式，只需要Logits，不计算Loss
            loss = None
            
        else:
            # 训练模式

            """
            展平logits
            输入: [B, T, V]
            输出: [B*T, V]
            为什么? cross_entropy要求输入是二维的 (N, C), 其中N是样本数, C是类别数
            """
            logits = logits.view(B * T, -1)

            """
            展平targets
            输入: [B, T]
            输出: [B*T]
            为什么? 每个位置都有一个目标token
            """
            targets = targets.view(B * T)

            """
            计算交叉熵损失
            输入: logits [B*T, V], targets [B*T]
            输出: loss 标量

            物理意义: 衡量预测分布与真实分布的差异
                    值越小，模型预测越准确
                Q: 为什么用CrossEntropy 而不是 MSE?
                A: 分类任务标准损失，对概率分布更敏感
            """
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        自回归生成: 根据输入序列，逐个预测新 token

        Args:
            idx: [B, T] 初始输入序列 (如 "Hello" 的 token IDs)
            max_new_tokens: 需要生成的新 token 数量
        Returns:
            idx: [B, T + max_new_tokens] 完整生成序列
        """
        # 逐个生成新token
        for _ in range(max_new_tokens):

            """
            截取上下文 (避免位置编码越界)
            输入: idx [B, T_current] (T_current 随生成增长)
            输出: idx_cond [B, min(T_current, block_size)]

            为什么需要?
                1. 位置编码只定义了 block_size 个位置
                2. 超出范围会报错
                3. 模型只需要看最近的上下文即可

                Q: 如果序列超过block_size怎么办?
                A: 滑动窗口, 只保留最近 block_size 个 token
            """
            idx_cond = idx[:, -block_size:]

            """
            前向传播 获取预测分数

            输入: idx_cond [B, T_cond]
            输出: logits [B, T_cond, vocab_size]

            为什么 targets=None?
            推理时没有标准答案, 不需要计算 loss
                
                Q: 推理和训练的forward有什么区别?
                A: 推理不需要计算Loss, 节省计算资源
            """
            logits, _ = self(idx_cond, None) 

            """
            只取最后一个时间步的预测

            输入: logits [B, T_cond, V]
            输出: logits [B, V]

            为什么只取最后一个?
            我们只关心"下一个token" 是什么, 不需要前面位置的预测
            例如：输入 “Hello”， 只预测第5个位置的下一个词

                Q: 为什么不使用所有位置的 logits?
                A: 自回归生成是逐个 token 进行的，只需预测下一步
            """
            logits = logits[:, -1, :]
            
            """
            转换为概率分布
            输入: logits [B, V]
            输出: probs [B, V]

            物理意义:
                logits 是原始分数，softmax后变成概率分布
                所有词的概率之和 = 1
                例如：[0.01, 0.02, 0.85, 0.01, ...] 表示第三个词的概率最高

                Q: logits和probs的区别?
                A: logits 是未归一化的分数, probs是归一化的概率
            """
            probs = F.softmax(logits, dim=-1)

            """
            采样下一个token
            输入: probs [B, V]
            输出: idx_next [B, 1]

            物理意义:
                根据概率分布随机采样一个token
                高概率的词更可能被选中，但低概率词也有机会

                Q: 为什么不用argmax而用multinomial?
                A: argmax 总是选概率最高的，输出单一重复，采样增加多样性，让生成更自然
            """
            idx_next = torch.multinomial(probs, num_samples=1)

            """
            拼接到序列末尾
            输入: idx [B, T_current], idx_next [B, 1]
            输出: idx [B, T_current + 1]

            物理意义:
                将生成的 token 添加到序列末尾
                下一次循环时，模型可以看到整个新生成的token

                Q: 为什么dim=1?
                A: dim=0时batch维度, dim=1时时间维度，我们要在时间维度上拼接新token
            """
            idx = torch.cat((idx, idx_next), dim=1)

        """
        返回完整序列
            输出: idx [B, T_initial + max_new_tokens]

        生成过程:
            初始输入: "Hello" -> idx = [H, e, l, l, o] [B=1, T=5]
        -> 
            第一次循环:
                idx_cond = [H, e, l, l, o]
                logits = 模型预测下一个词的概率分布
                probs = softmax(logits)
                idx_next = 采样得到的下一个词 (如 " " 空格)
                idx = [H, e, l, l, o, " "] [B=1, T=6]
        ->
            第二次循环:
                idx_cond = [e, l, l, o, " "] (只保留最近的 block_size 个 token)
                logits = 模型预测下一个词的概率分布
                probs = softmax(logits)
                idx_next = 采样得到的下一个词 (如 "W")
                idx = [H, e, l, l, o, " ", "W"] [B=1, T=7]

            ...
                最终输出: [H, e, l, l, o, " ", "W", "o", "r", "l", "d"] [B=1, T=T+max_new_tokens]
        """
        return idx
    

### 测试
if __name__ == "__main__":
    """
        Q: 为什么用 if __name__ == "__main__": ?
        A:
            1. 防止被其它模块 import 时自动执行测试代码
            2. 区分 作为库使用 和 作为脚本运行
            3. Python的惯用写法，确保测试代码只在直接运行时执行
    """
    from core.data import SimpleTokenizer, prepare_data

    # 从 config 导入 vocab_size 需要先初始化 tokenizer
    """
        # 为什么需要先初始化 tokenizer？
        # 因为 vocab_size 取决于数据中的唯一字符数
        # 字符级分词：65 (莎士比亚数据集)
        # BPE 分词：50257 (GPT-2) 等

        Q: vocab_size 对模型有什么影响
        A: 决定 Embedding层大小，影响参数量和表达能力
    """
    text = prepare_data()
    tokenizer = SimpleTokenizer(text)
    vocab_size = tokenizer.vocab_size

    print(f"Building model with vocab_size={vocab_size}...")

    """
    初始化模型

    为什么调用 model.to(device)??
    确保模型参数和数据在同一设备 (CPU/GPU)

        Q: 模型初始化后的参数是随机的吗?
        A: 是的，模型参数通过 _init_weights 方法随机初始化，通常使用正态分布 N(0, 0.02)。这有助于训练的稳定性和收敛速度。
    """
    model = GPT(vocab_size)
    model.to(device)

    # 前向传播测试 (targets=None 可以看到原始 logits 形状)
    """
    为什么logits是[B, T, vocab_size]而不是[B*T, vocab_size]? 3D的
    [B, T, vocab_size] 是模型的原始输出形状，表示每个位置对每个词的预测分数
    """
    test_input = torch.randint(0, vocab_size, (2, 10), device=device)
    logits, loss = model(test_input, targets=None)

    print(f"Input shape: {test_input.shape}")
    print(f"Logits shape: {logits.shape}")  # 这里会显示 [2, 10, 65]

    # 再测试带 loss 的情况
    """
    targets = test_input  -> 计算loss

    为什么targets也是 test_input?
    测试时随便用一个形状的Tensor即可
    真实训练时 targets 是输入右移一位的结果
    """
    logits_with_loss, loss = model(test_input, test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # 生成测试
    """
    切换到评估模式

        Q: model.eval() 和 model.train() 有什么区别?
        A: Dropout 行为不同: train 时随机丢弃, eval时不丢弃
           LayerNorm 行为不同: train 时用 batch 统计, eval时用全局统计
           推理时必须用 eval(), 否则结果不稳定
    """
    model.eval()
    """
    起始序列: "Hello" -> [46, 43, 50, 50, 53] (根据 tokenizer 的 string_to_int 映射)
    形状：[B=1, T=5]

        Q: 为什么是二维张量[1, 5]而不是一维[5]?
        A: 模型输入要求是 [B, T] 形状，即使只有一个序列也需要保持批次维度
    """
    start = torch.tensor([[tokenizer.string_to_int['H'],
                           tokenizer.string_to_int['e'],
                           tokenizer.string_to_int['l'],
                           tokenizer.string_to_int['l'],
                           tokenizer.string_to_int['o']]], device=device)
    """
    生成20个新token
    输出形状: [B=1, T=5+20=25]

        Q: generate 能批量生成吗?
        A: 可以，B>1时每个样本独立生成
    """
    generated = model.generate(start, max_new_tokens=20)
    print(f"Generated: {tokenizer.decode(generated[0].tolist())}")

    print("Model test completed successfully.")
