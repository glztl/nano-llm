# ✌️ Tiny LLM

从零手写实现一个迷你大语言模型，深入学习 Transformer 架构与 PyTorch 框架原理。

每一段代码都配有详细注释与解读，涵盖从输入到输出的矩阵变换示例，并穿插常见 Q&A，帮助你扫清理论与实践障碍。

本项目以低维度、可控规模为切入点，逐步剖析 LLM 的核心机制。无需高端 GPU，普通 CPU 也能完整体验训练与推理流程，轻松感受大模型的魅力。

适合 AI/NLP 学习者、工程师、面试准备者，助你夯实底层原理、提升工程能力，快速上手大语言模型开发之路。

> **项目目标**：通过从零实现一个完整的 LLM，深入理解大模型的底层原理，为面试和实际工作打下坚实基础。

---

## 📋 目录

- [特性](#-特性)
- [模型规格](#-模型规格)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [生成示例](#-生成示例)
- [训练进度](#-训练进度)
- [面试考点](#-面试考点)
- [后续计划](#-后续计划)

---

## ✨ 特性

- 🔧 **从零实现**：不依赖 HuggingFace Transformers，纯 PyTorch 手写 Transformer 架构
- 📚 **完整流程**：涵盖数据处理、模型构建、训练循环、推理生成全流程
- 🎯 **面试导向**：代码注释包含面试高频考点，适合求职准备
- 🚀 **工程规范**：使用 UV 包管理、Git 版本控制、VS Code 配置统一
- 📊 **可视化训练**：实时显示训练进度、Loss 变化、生成样本

---

## 📊 模型规格

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **参数量** | 0.84M | 轻量级，适合学习 |
| **架构** | Decoder-only Transformer | GPT 风格 |
| **词表大小** | 65 | 字符级分词 |
| **上下文长度** | 256 tokens | 一次处理的最大序列 |
| **嵌入维度** | 128 | `n_embd` |
| **注意力头数** | 4 | `n_head` |
| **Transformer 层数** | 4 | `n_layer` |
| **训练数据** | Shakespeare (1.1M 字符) | 莎士比亚全集 |
| **训练步数** | 5000 iters | 约 15 分钟 (CPU) |
| **优化器** | AdamW | β=(0.9, 0.95), weight_decay=1e-1 |
| **学习率调度** | Warmup + Cosine Decay | 预热 100 步 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- UV (包管理工具)

### 安装

```bash
# 克隆项目
git clone https://github.com/glztl/tiny-llm.git
cd tiny-llm

# 安装依赖 (使用 UV)
uv sync

### 模型训练
 开始训练
    uv run python train.py
```
 训练完成后会生成:
 - best_model.pth (最佳验证集模型)
 - final_model.pth (最终模型)


### 推理生成
 - 批量生成示例
   ```bash
    uv run python generate.py best_model.pth
   ```

 - 交互式对话
   ```bash
    uv run python generate.py best_model.pth
   ```
 - 选择输出模式

### 自定义配置
- 编辑 config.py 调整超参数：

- 增大模型 (需要更多显存)
  ```bash
    n_embd = 256    # 嵌入维度
    n_head = 8      # 注意力头数
    n_layer = 6     # Transformer 层数
  ```
    
- 增加训练步数
  ```bash
    max_iters = 10000
  ```

### 项目结构

tiny_llm/
├── .vscode/                 # VS Code 项目配置
│   ├── settings.json        # 编辑器设置
│   └── extensions.json      # 调试配置
├── config/                  # 配置模块
│   └── config.py            # 全局超参数配置
├── core/                    # 核心代码
│   ├── data.py              # 数据加载与字符级分词
│   ├── model.py             # Transformer 模型架构
│   ├── train.py             # 训练循环 (含学习率调度)
│   └── generate.py          # 推理生成 (支持温度采样)
├── docs/                    # 文档
│   └── basic.md             # 基础说明文档
├── .gitignore               # Git 忽略配置
├── README.md                # 项目说明
├── pyproject.toml           # 项目依赖配置
└── uv.lock                  # 依赖锁定文件

### 生成示例
- 训练前 (随机权重)
  ```bash
   输入: Hello
   输出: HellomJrZO,n!URnAYw&sEmK
  ```

- 训练后 (5000 步)
  ```bash
   输入: ROMEO:
   输出: As look and that the king: that, thou noble country
         The barned lamster, the crage to of the no hear...
   
   输入: To be or not to be,
   输出: As look shad is not discrent a stongue,
         And court of that he desire of three,
         You that God gracesti...
   
   输入: Once upon a time
   输出: is love so did in them,
         Which would the bend for my eyears.
         KING RICHARD III:
         What's how they hear...
   ```

### 📈 训练进度

| 阶段         | Train Loss | Val Loss | 生成质量     |
|--------------|------------|----------|--------------|
| 初始 (0 步)  | 4.20       | 4.20     | 随机乱码     |
| 早期 (500 步)| 3.00       | 3.10     | 有字符结构   |
| 中期 (2000 步)| 2.00      | 2.20     | 有单词       |
| 后期 (5000 步)| 1.50      | 1.60     | 有语义       |


### 面试考点
本项目涵盖以下面试高频知识点：
- **Transformer 架构**
  - 自注意力机制 (Self-Attention) 原理与实现
  - 多头注意力 (Multi-Head Attention) 的作用
  - 因果掩码 (Causal Mask) 的必要性
  - 位置编码 (Positional Encoding) 的实现
  - 残差连接 (Residual Connection) 的作用
  - 层归一化 (LayerNorm) 的位置与作用
- **训练技巧**
  - 学习率调度 (Warmup + Cosine Decay)
  - 梯度裁剪 (Gradient Clipping)
  - AdamW 优化器 vs Adam
  - Dropout 正则化
  - 混合精度训练 (AMP)
- **推理优化**
  - 自回归生成 (Autoregressive Generation)
  - Temperature 采样控制
  - Top-K / Top-P 采样
  - KV Cache 优化 (后续实现)
- **工程实践**
  - PyTorch 张量形状变换
  - 数据加载与预处理
  - 模型保存与加载 (state_dict)
  - Git 版本控制规范
  - 项目配置管理

🔜 后续计划

| 功能               | 优先级 | 状态     |
|--------------------|--------|----------|
| BPE 分词升级       | ⭐⭐⭐   | ⏳ 待实现 |
| RoPE 位置编码      | ⭐⭐⭐   | ⏳ 待实现 |
| SwiGLU 激活函数    | ⭐⭐    | ⏳ 待实现 |
| 增大模型规模 (10M+)| ⭐⭐⭐   | ⏳ 待实现 |
| LoRA 微调          | ⭐⭐⭐⭐  | ⏳ 待实现 |
| 多 GPU 训练        | ⭐⭐    | ⏳ 待实现 |
| 量化部署 (GGUF)    | ⭐⭐    | ⏳ 待实现 |
| Web 界面演示       | ⭐     | ⏳ 待实现 |

### 学习资源

- [Attention Is All You Need - Transformer 原论文](https://arxiv.org/abs/1706.03762)
- [nanoGPT - Andrej Karpathy 的简化实现](https://github.com/karpathy/nanoGPT)
- [The Illustrated Transformer - 可视化讲解](https://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Course - NLP 与大模型课程](https://huggingface.co/learn/nlp-course/)

### 许可证
MIT License

<div align="center">

如果你觉得这个项目有帮助，欢迎点个 ⭐Star 支持！<br/>
Made with ✌ by glztl

</div>