"""
config.py - 全局超参数配置

统一管理数据、模型、训练相关的所有超参数
面试考点：为什么要把配置单独管理？
答：便于实验追踪、避免硬编码、支持超参数搜索、代码可维护性高
"""

# ==========================================
# 1. 设备配置
# ==========================================
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==========================================
# 2. 数据相关配置
# ==========================================
block_size = 256      # 上下文窗口长度 (模型一次能看到多少 token)
batch_size = 64       # 每个 batch 的样本数
train_split = 0.9     # 训练集占比

# ==========================================
# 3. 模型架构配置 (控制模型大小)
# ==========================================
n_embd = 128          # 嵌入维度 (Embedding Dimension)
n_head = 4            # 注意力头数 (Number of Attention Heads)
n_layer = 4           # Transformer 层数 (Number of Blocks)
dropout = 0.2         # Dropout 概率

# 参数量估算 (~1-5M)
# - Embedding: vocab_size * n_embd ≈ 65 * 128 = 8,320
# - Attention: 4 * n_embd² * n_head ≈ 4 * 128² * 4 = 262,144 (每层)
# - FFN: 8 * n_embd² ≈ 8 * 128² = 131,072 (每层)
# - 总计 ≈ (262k + 131k) * 4 层 + Embedding ≈ 1.6M

# ==========================================
# 4. 训练相关配置 (后续 train.py 使用)
# ==========================================
learning_rate = 3e-4     # 学习率
max_iters = 5000         # 最大迭代次数
eval_interval = 500      # 评估间隔
eval_iters = 200         # 评估时的 batch 数
warmup_iters = 100       # 学习率预热步数

# ==========================================
# 5. 模型规模预设 (方便快速切换)
# ==========================================
# 取消注释以下任一配置来快速切换模型大小

# Nano (约 1M 参数，快速测试)
# n_embd, n_head, n_layer = 64, 2, 2

# Small (约 5M 参数，推荐)
# n_embd, n_head, n_layer = 128, 4, 4

# Medium (约 20M 参数，需要更多显存)
# n_embd, n_head, n_layer = 256, 8, 6