import torch
import requests
import os

# 从 config 导入配置 (统一配置管理)
from config.config import block_size, batch_size, train_split, device

"""
    1. 超参数配置

    Q: 为什么要把超参数单独管理
    A: 方便实验追踪，避免硬编码，便于调整模型大小和训练细节
"""
block_size = block_size  # 模型一次能看到的最大上下文长度 (Context Window)
batch_size = batch_size  # 每次梯度更新使用的样本数
train_split = train_split  # 训练集占比
device = device


"""
    2. 数据下载与准备
"""


def prepare_data() -> str:
    """
    下载数据并进行字符级分词

    Q: 字符级分词 vs 词级分词 vs BPE的优缺点?
    """

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # 缓存数据，避免每次运行都重新下载
    if not os.path.exists("input.txt"):
        print("Downloading data...")
        text = requests.get(url).text
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()

    print(f"Data length: {len(text)} characters")
    return text


"""
    3. 分词器 (Tokenizer)

    原始文本: 'hello'
    ->
    [encode]
    ->
    数字列表: [1, 0, 2, 2, 3]
    ->
    模型训练/推理: 模型只认识数字
    ->
    [decode]
    ->
    恢复文本: 'hello'

    Q: 字符级分词的优缺点
    A: 
        优点：
            1. 此表极小(65)个字符，内存占用低
            2. 不会出现OOV(未登录词), 任何字符都能编码
            3. 实现简单，无需外部库
        缺点：
            1. 序列很长，hello要5个token，计算慢 
            2. 语义信息弱，模型需要更深才能理解词义
            3. 无法利用子词信息，如unhappy 拆解成 un + happy

        工业界方案： BPE（Byte Pair Encoding）、WordPiece、SentencePiece，平衡词表大小与语义表达。

    Q:  为什么需要两个映射字典 (string_to_int 和 int_to_string)
    A:  编码时：查 string_to_int，O(1) 时间将字符转 ID
        解码时：查 int_to_string，O(1) 时间将 ID 转字符
    Q:  能不能只用一个字典反向查找？
    A:  可以，但反向查找需要遍历字典，时间复杂度 O(n)，效率低。空间换时间是常见优化策略。 

    Q:  如果遇到训练时没见过的字符怎么办？
    A:  字符级分词：不会出现 OOV，因为词表包含所有出现过的字符。但如果是全新字符（如表情符号），会报错 KeyError
        工业级方案：
            添加 <UNK>（未知 token）占位
            使用字节级编码（如 GPT-2 的 BPE），任何字符都能拆成字节
            def encode(self, s: str) -> list[int]:
                return [self.string_to_int.get(c, self.unk_token) for c in s]
""" 
class SimpleTokenizer:
    def __init__(self, text: str):
        # 构建词表: 所有出现的唯一字符
        """
            去重并按照 ASCII 码排序
            
            h e l l o
            {'h', 'e', 'l', 'o'}
            {'e', 'h', 'l', 'o'}
        """
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # 建立映射关系 (String <-> Int)
        """
            {'e': 0, 'h': 1, 'l': 2, 'o': 3}
            {0: 'e', 1: 'h', 2: 'l', 3: 'o'}
        """
        self.string_to_int = {ch: i for i, ch in enumerate(self.chars)}  # String to Int
        self.int_to_string = {i: ch for i, ch in enumerate(self.chars)}  # Int to String

    def encode(self, s: str) -> list[int]:
        """
            字符串 -> 数字列表

            输入: 'hello' ->
            h->1, e->0, l->2, l->2, o->3
            输出: [1, 0, 2, 2, 3]

            return [self.string_to_int[c] for c in s] 
         ==> 
            result[]
            for c in s:
                result.append(self.string_to_int(c))

            return result

            Q: 为什么返回值是 List[int] 而不是 torch.Tensor
            A: 分词器只负责"文本 -> 数学"的映射，不依赖框架。转 Tensor 是数据加载阶段的事情，这样设计解耦更好
        """
        return [self.string_to_int[c] for c in s]

    def decode(self, indices: list[int]) -> str:
        """
            数字列表 -> 字符串

            输入: [1, 0, 2, 2, 3]
            遍历: 1->'h', 0->'e', 2->'l', 2->'l', 3->'o'
            拼接: join(['h', 'e', 'l', 'l', 'o'])
            输出: 'hello'

            Q: 为什么用 "".join 而不是用 '+' 拼接字符串
            A: 字符串在 Python 中是不可变的，'+' 的话，每次都会创建新对象，时间复杂度是O(n^2), join()一次性分配内存
               时间复杂度 O(n)，效率更高
        """
        return "".join([self.int_to_string[i] for i in indices])


"""
    4. 数据集构建 (Dataset & DataLoader)

    原始文本 (string)
    ->
    tokenizer.encode() -> 数字列表 (list[int], [int, int, int, ...])
    ->
    torch.tensor() -> Tensor (torch.LongTensor([int, int, int, ...]))
    ->
    train/val 划分 -> 训练集/验证集 (Tensor)
    ->
    get_batch() -> 随机采样一个 batch 的输入 (X) 和目标 (Y) [batch_size, block_size]
    ->
    模型输入 x, 目标 y
"""
class TextDataset:
    def __init__(self, text, tokenizer, block_size, split="train") -> None:
        """
            text: 原始文本数据
                Q: 为什么输入原始文本而不是已经编码的数字列表?
                A: 由类内部负责划分 train/val
            tokenizer: 分词器实例

            block_size: 模型上下文窗口大小
                决定模型一次看多少字符，影响显存和训练速度

            split: "train" 或 "val"，决定划分训练集还是验证集
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        # 将文本转换为 Tensor
        """
            Q: 为什么 dtype 是 torch.long?
            A: 因为这是索引数据，不是浮点数，long节省内存且符合 Embedding 输入要求

            语义正确：这是字符 ID（如 46, 43, 50...），不是连续数值，不能用浮点数。
            Embedding 要求：nn.Embedding 的输入必须是 LongTensor。
            内存效率：如果用 float32 存 ID，浪费内存且无意义。

            torch.long (int64) 用于索引、类别、ID， 8字节/元素 
            torch.float (float32) 用于模型权重、梯度， 4字节/元素
            torch.float (float16) 用于混合精度训练， 2字节/元素
        """
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        """
            划分训练集/验证集

            Q: 为什么先划分再采样？
            A: 确保训练集和验证集不重叠，避免数据泄露 (Data Leakage)
               重叠后：
                模型"见过"答案，验证 Loss 虚低
                无法检测过拟合，你以为模型泛化好，实际是死记硬背

        """
        n = int(train_split * len(data))
        if split == "train":
            self.data = data[:n]
        else:  # "val"
            self.data = data[n:]
            
        print(f"Dataset [{split}] size: {len(self.data)} tokens")

    def get_batch(self, batch_size):
        """
        随机采样一个 batch 的数据

            Q: 为什么输入 X 和 目标 Y 是错位一位的?
            A: 因为任务是 Next Token Prediction, 输入 "Hello"， 期望输出 "ello"

            具体来说，输入序列是 [H, e, l, l, o]，目标序列是 [e, l, l, o, <eos>]，模型需要学习在给定输入的情况下预测下一个字符。

        torch.randint的参数解释：
            torch.randint(
            low=0,                              # 默认从 0 开始
            high=len(self.data) - self.block_size,  # 最大起始索引
            size=(batch_size,)                  # 生成 batch_size 个随机数
        )
        """
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))

        # 高效堆叠 Tensor, 避免 Python 循环
        """
            Q: 为什么 X 和 Y 错位一位？

            A: 
                输入 X:  [B, O, L, I, N, G, ...]   # 模型看到的
                目标 Y:  [O, L, I, N, G, <eos>]   # 模型需要预测的
               
                输入x          目标y           任务
                'B'            'O'             预测 'B' 后的下一个字符是 'O'
                'O'            'L'             预测 'O' 后的下一个字符是 'L'
                'L'            'I'             预测 'L' 后的下一个字符是 'I'
                'I'            'N'             预测 'I' 后的下一个字符是 'N'

                自监督学习的本质是让模型根据过去预测未来，Y 是 X 向右平移一位的结果
        """
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])

        """
            Q: .to(device) 的作用是什么？
            A: 将 Tensor 移动到指定设备（CPU 或 GPU），确保数据和模型在同一设备上进行计算，避免跨设备通信导致的性能问题。

            # 常见错误
            model = GPT().to('cuda')   # 模型在 GPU
            x = torch.randn(64, 256)   # 数据在 CPU
            output = model(x)          # Expected all tensors to be on the same device

            # 正确做法
            x = x.to('cuda')           # 数据也移到 GPU
            output = model(x)

            Q: 频繁调用 .to(device) 会影响性能吗？
            A: 频繁调用 .to(device) 会导致性能下降，因为每次调用都会涉及数据复制和设备同步。
               建议在数据加载阶段一次性将数据移到目标设备，避免在训练循环中频繁调用。
        """
        return x.to(device), y.to(device)


"""
    5. 测试
"""
if __name__ == "__main__":
    print(f"Using device: {device}")

    # 1. 数据准备
    text = prepare_data()

    # 2. 初始化分词器
    tokenizer = SimpleTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 3. 测试编解码
    test_str = "hello"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    print(f"Encode test: {test_str} -> {encoded}")
    print(f"Decode test: {encoded} -> {decoded}")
    assert test_str == decoded, "Tokenizer failed!"

    # 4. 构建数据集
    train_dataset = TextDataset(text, tokenizer, block_size, split="train")
    val_dataset = TextDataset(text, tokenizer, block_size, split="val")

    # 5. 获取一个 batch 测试
    xb, yb = train_dataset.get_batch(batch_size)
    print(f"Input shape: {xb.shape}, Target shape: {yb.shape}")
    print(f"First input sequence: {tokenizer.decode(xb[0].tolist())}")
    print(f"First target sequence: {tokenizer.decode(yb[0].tolist())}")