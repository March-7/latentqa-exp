#!/usr/bin/env python3
"""
数据集测试脚本 - 展示LatentQA数据集的工作原理
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lit.utils.dataset_utils import (
    LatentQADataset, 
    DataCollatorForLatentQA,
    get_model_config_name,
    NUM_READ_TOKENS_TO_SHIFT,
    NUM_WRITE_TOKENS_TO_SHIFT,
    PAD_TOKEN_IDS
)
from lit.utils.infra_utils import get_tokenizer

def print_separator(title):
    """打印分隔符"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def load_sample_data():
    """加载示例数据"""
    data_dir = "./data/train"
    
    # 加载少量示例数据
    with open(f"{data_dir}/qa.json", "r") as f:
        qa_data = json.load(f)
    
    with open(f"{data_dir}/control.json", "r") as f:
        control_data = json.load(f)
    
    with open(f"{data_dir}/stimulus.json", "r") as f:
        stimulus_data = json.load(f)
    
    with open(f"{data_dir}/stimulus_completion.json", "r") as f:
        stimulus_completion_data = json.load(f)
    
    return {
        'qa': qa_data,
        'control': control_data,
        'stimulus': stimulus_data,
        'stimulus_completion': stimulus_completion_data
    }

def test_model_config_mapping():
    """测试模型名称映射"""
    print_separator("模型名称映射测试")
    
    test_paths = [
        "/data1/ckx/hf-checkpoints/meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "/some/other/path/Meta-Llama-3-8B-Instruct"
    ]
    
    for path in test_paths:
        mapped_name = get_model_config_name(path)
        print(f"原始路径: {path}")
        print(f"映射名称: {mapped_name}")
        print(f"PAD Token ID: {PAD_TOKEN_IDS[mapped_name]}")
        print("-" * 50)

def test_tokenizer():
    """测试tokenizer"""
    print_separator("Tokenizer测试")
    
    # 使用一个较小的模型进行测试
    model_name = "/data1/ckx/hf-checkpoints/meta-llama/Llama-3.1-8B-Instruct"
    
    try:
        tokenizer = get_tokenizer(model_name)
        print(f"✓ 成功加载tokenizer: {model_name}")
        print(f"  - PAD Token ID: {tokenizer.pad_token_id}")
        print(f"  - EOS Token ID: {tokenizer.eos_token_id}")
        print(f"  - 词汇表大小: {len(tokenizer)}")
        
        # 测试tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        print(f"  - 测试文本: '{test_text}'")
        print(f"  - Token IDs: {tokens}")
        print(f"  - 解码结果: '{tokenizer.decode(tokens)}'")
        
    except Exception as e:
        print(f"✗ 加载tokenizer失败: {e}")
        print("  这是因为模型文件不在本地，但映射逻辑是正确的")

def test_dataset_structure():
    """测试数据集结构"""
    print_separator("数据集结构测试")
    
    # 加载示例数据
    data = load_sample_data()
    
    # 展示数据结构
    print("1. QA数据结构:")
    qa_sample_key = list(data['qa'].keys())[0]
    qa_sample = data['qa'][qa_sample_key][0]
    print(f"   - 类别: {qa_sample_key}")
    print(f"   - 问题: {qa_sample[0]}")
    print(f"   - 答案: {qa_sample[1]}")
    
    print("\n2. Control数据结构:")
    control_sample = data['control'][0]
    print(f"   - 控制指令: {control_sample['control_user']}")
    print(f"   - 标签: {control_sample['label']}")
    
    print("\n3. Stimulus数据结构:")
    stimulus_sample = data['stimulus'][0]
    print(f"   - 控制指令: {stimulus_sample['control_user']}")
    print(f"   - 用户输入: {stimulus_sample['stimulus_user']}")
    print(f"   - 标签: {stimulus_sample['label']}")

def test_dataset_creation():
    """测试数据集创建"""
    print_separator("数据集创建测试")
    
    # 加载示例数据
    data = load_sample_data()
    
    # 创建模拟tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 128010
            self.eos_token_id = 128001
            self.name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
        
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, chat_template=None):
            # 简单的模板实现
            result = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    result += f"<|user|>{content}<|end|>"
                elif role == 'assistant':
                    result += f"<|assistant|>{content}<|end|>"
            
            if add_generation_prompt:
                result += "<|assistant|>"
            
            return result if not tokenize else self.encode(result)
        
        def encode(self, text):
            # 简单的编码实现
            return [i % 1000 for i in range(len(text.split()))]
        
        def __call__(self, text, **kwargs):
            if isinstance(text, str):
                tokens = self.encode(text)
                return {
                    'input_ids': torch.tensor(tokens),
                    'attention_mask': torch.ones(len(tokens))
                }
            else:
                # 处理batch
                max_len = max(len(self.encode(t)) for t in text) if text else 1
                batch_tokens = []
                batch_masks = []
                for t in text:
                    tokens = self.encode(t)
                    # 填充到相同长度
                    padded_tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
                    mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
                    batch_tokens.append(padded_tokens)
                    batch_masks.append(mask)
                
                # 创建返回对象，模拟tokenizer输出格式
                result = type('TokenizerOutput', (), {})()
                result.input_ids = torch.tensor(batch_tokens)
                result.attention_mask = torch.tensor(batch_masks)
                return result
    
    tokenizer = MockTokenizer()
    
    # 创建数据集
    try:
        # 准备数据格式：每个数据项是 [data_dict, id_tuples]
        # 将列表转换为字典格式
        data_system = [{i: item for i, item in enumerate(data['stimulus_completion'][:2])}, [(0, 0, 0) for _ in range(2)]]
        data_stimulus_completion = [{i: item for i, item in enumerate(data['stimulus_completion'][:2])}, [(1, 0, 0) for _ in range(2)]]
        data_stimulus = [{i: item for i, item in enumerate(data['stimulus'][:2])}, [(2, 0, 0) for _ in range(2)]]
        data_control = [{i: item for i, item in enumerate(data['control'][:2])}, [(3, 0, 0) for _ in range(2)]]
        
        dataset = LatentQADataset(
            tokenizer=tokenizer,
            data_system=data_system,
            data_stimulus_completion=data_stimulus_completion,
            data_stimulus=data_stimulus,
            data_control=data_control,
            qa_data=data['qa'],
            add_thought_tokens=False
        )
        
        print(f"✓ 成功创建数据集，包含 {len(dataset)} 个样本")
        
        # 获取一个样本
        sample = dataset[0]
        print("\n样本结构:")
        print(f"  - read_prompt: {sample['read_prompt'][:100]}...")
        print(f"  - dialog长度: {len(sample['dialog'])}")
        print(f"  - mask_type: {sample['mask_type']}")
        
        # 展示对话
        print("\n对话内容:")
        for i, msg in enumerate(sample['dialog']):
            print(f"    {i+1}. [{msg['role']}]: {msg['content'][:50]}...")
        
    except Exception as e:
        print(f"✗ 创建数据集失败: {e}")
        import traceback
        traceback.print_exc()

def test_data_collator():
    """测试数据整理器"""
    print_separator("数据整理器测试")
    
    # 创建模拟样本
    mock_samples = [
        {
            'read_prompt': '<|user|>Hello<|end|><|assistant|>',
            'dialog': [
                {'role': 'user', 'content': 'What is AI?'},
                {'role': 'assistant', 'content': 'AI is...'}
            ],
            'mask_type': 'user'
        },
        {
            'read_prompt': '<|user|>Hi<|end|><|assistant|>',
            'dialog': [
                {'role': 'user', 'content': 'How are you?'},
                {'role': 'assistant', 'content': 'I am fine...'}
            ],
            'mask_type': 'user'
        }
    ]
    
    # 直接加载真实的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # 确保tokenizer有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据整理器
    collator = DataCollatorForLatentQA(
        tokenizer=tokenizer,
        generate=False,
        modify_chat_template=False
    )
    
    try:
        # 整理数据
        batch = collator(mock_samples)
        
        print("✓ 成功整理数据")
        print(f"  - Batch keys: {list(batch.keys())}")
        
        if 'tokenized_read' in batch:
            read_shape = batch['tokenized_read']['input_ids'].shape
            print(f"  - 读取数据形状: {read_shape}")
        
        if 'tokenized_write' in batch:
            write_shape = batch['tokenized_write']['input_ids'].shape
            print(f"  - 写入数据形状: {write_shape}")
        
        if 'verb_mask' in batch:
            mask_shape = batch['verb_mask'].shape
            print(f"  - 掩码形状: {mask_shape}")
            print(f"  - 掩码示例: {batch['verb_mask'][0][:10].tolist()}")
        
    except Exception as e:
        print(f"✗ 数据整理失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 LatentQA 数据集测试脚本")
    print("此脚本将帮助你理解数据集的工作原理")
    
    # 运行各项测试
    test_model_config_mapping()
    test_tokenizer()
    test_dataset_structure()
    test_dataset_creation()
    test_data_collator()
    
    print_separator("测试完成")
    print("✅ 所有测试已完成！")
    print("\n📝 关键概念总结:")
    print("1. 模型名称映射：将本地路径映射到标准配置")
    print("2. 数据集结构：包含读取prompt、对话和掩码类型")
    print("3. 数据整理：将原始数据转换为模型输入格式")
    print("4. 激活掩码：控制哪些token的激活被用于训练")
    print("\n🔗 更多信息请查看 dataset_utils.py 中的详细实现")

if __name__ == "__main__":
    main()