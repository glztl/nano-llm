# Tiny LLM

从零手写实现一个迷你大语言模型，用于学习 Transformer 架构与 PyTorch 底层原理。

## 目标
1. 练习 Python 基本功
2. 熟悉 PyTorch 框架
3. 理解大模型底层思路
4. 面试准备

## 环境
- Python 3.10+
- PyTorch
- UV (包管理)

## 项目结构
- `data.py`: 数据加载与分词
- `model.py`: 模型架构定义
- `train.py`: 训练循环
- `generate.py`: 推理生成

## 使用方法

```bash
uv run data.py