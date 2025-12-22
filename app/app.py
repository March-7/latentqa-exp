#!/usr/bin/env python3
"""
LatentQA Gradio Interface
一个用于分析和解释大语言模型激活值的Web界面
"""

import os
import json
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import gradio as gr

from lit.utils.dataset_utils import lqa_tokenize, BASE_DIALOG, ENCODER_CHAT_TEMPLATES
from lit.utils.activation_utils import latent_qa
from lit.utils.infra_utils import (
    update_config,
    get_model,
    get_tokenizer,
    get_modules,
    get_model_config_name
)
from lit.reading import ForCausalLMLossPatched
from lit.configs.interpret_config import interpret_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型
target_model = None
decoder_model = None
tokenizer = None
module_read = None
module_write = None
args = None
chat_template = None

# 预定义的问题列表
DEFAULT_QUESTIONS = [
    "What type of persona will the assistant adopt?",
    "How will the assistant express themselves?",
    "Is the assistant experiencing any dilemmas?",
    "Is the assistant more concerned about the user or itself?",
    "Is the conversation truthful?"
]

def initialize_models():
    """初始化目标模型和解码器模型"""
    global target_model, decoder_model, tokenizer, module_read, module_write, args, chat_template
    
    try:
        # 获取配置
        args = interpret_config()
        
        # 强制使用本地模型路径（与 notebook 保持一致）
        args.target_model_name = "/data1/ckx/hf-checkpoints/meta-llama/Llama-3.1-8B-Instruct"
        args.decoder_model_name = "out/runs/000/checkpoints/epoch4-steps324200-2025-12-18_14-12-46"
        
        logger.info(f"Target model path: {args.target_model_name}")
        logger.info(f"Decoder model path: {args.decoder_model_name}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # 初始化tokenizer，强制使用本地文件
        logger.info("Loading tokenizer from local path...")
        try:
            tokenizer = get_tokenizer(args.target_model_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # 尝试使用本地模式
            tokenizer = AutoTokenizer.from_pretrained(
                args.target_model_name, 
                padding_side="left", 
                add_eos_token=True,
                local_files_only=True
            )
            from lit.utils.dataset_utils import PAD_TOKEN_IDS
            config_name = get_model_config_name(args.target_model_name)
            tokenizer.pad_token_id = PAD_TOKEN_IDS[config_name]
        
        # 初始化模型
        logger.info("Loading decoder model...")
        decoder_model = get_model(
            args.target_model_name,
            tokenizer,
            load_peft_checkpoint=args.decoder_model_name,
            device=device,
        )
        logger.info("Decoder model loaded successfully")
        
        logger.info("Loading target model...")
        target_model = get_model(args.target_model_name, tokenizer, device=device)
        logger.info("Target model loaded successfully")
        
        # 设置评估模式
        decoder_model.eval()
        target_model.eval()
        
        # 设置随机种子
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        
        # 获取模块
        module_read, module_write = get_modules(target_model, decoder_model, **vars(args))
        
        # 获取聊天模板
        chat_template = ENCODER_CHAT_TEMPLATES.get(get_model_config_name(tokenizer.name_or_path), None)
        
        logger.info("Models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return False

def generate_response(query: str) -> str:
    """使用目标模型生成回复"""
    try:
        messages_batch = [[{"role": "user", "content": query}]]
        
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True
        ).to(target_model.device)
        
        with torch.no_grad():
            generated_ids = target_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )
        
        actual_generated_ids = [
            output[len(input_id):] for input_id, output in zip(inputs.input_ids, generated_ids)
        ]
        responses = tokenizer.batch_decode(actual_generated_ids, skip_special_tokens=True)
        
        return responses[0].strip() if responses else "生成失败"
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"生成错误: {str(e)}"

def analyze_dialog(query: str, response: str, question: str) -> str:
    """分析对话并回答问题"""
    try:
        # 构建对话
        dialog = [query, response]
        
        # 构建probe数据
        probe_data = []
        generate = True
        
        # 构建read_prompt
        if len(dialog) == 1:
            read_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": dialog[0]}],
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            )
        elif len(dialog) == 2:
            read_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": dialog[0]},
                    {"role": "assistant", "content": dialog[1]},
                ],
                tokenize=False,
                chat_template=chat_template,
            )
        
        # 构建分析对话
        if generate:
            analysis_dialog = [{"role": "user", "content": question}]
        else:
            analysis_dialog = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "预设答案"},
            ]
        
        probe_data.append({
            "read_prompt": read_prompt,
            "dialog": BASE_DIALOG + analysis_dialog,
        })
        
        # 标记化
        batch = lqa_tokenize(
            probe_data,
            tokenizer,
            name=args.target_model_name,
            generate=generate,
            mask_type=None,
            modify_chat_template=args.modify_chat_template,
            mask_all_but_last=True,
        )
        
        # 执行latent QA
        out = latent_qa(
            batch,
            target_model,
            decoder_model,
            module_read[0],
            module_write[0],
            tokenizer,
            shift_position_ids=False,
            generate=generate,
            max_new_tokens=256
        )
        
        # 解码结果
        num_tokens = batch["tokenized_write"]["input_ids"][0].shape[0]
        completion = tokenizer.decode(out[0][num_tokens:], skip_special_tokens=True)
        
        return completion.strip()
        
    except Exception as e:
        logger.error(f"Error analyzing dialog: {e}")
        return f"分析错误: {str(e)}"

def generate_and_analyze(query: str, question: str) -> Tuple[str, str, str]:
    """生成回复并进行分析"""
    try:
        # 生成回复
        response = generate_response(query)
        
        # 分析对话
        analysis_result = analyze_dialog(query, response, question)
        
        return query, response, analysis_result
        
    except Exception as e:
        logger.error(f"Error in generate_and_analyze: {e}")
        return query, f"生成错误: {str(e)}", f"分析错误: {str(e)}"

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="LatentQA - 模型激活值分析", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔍 LatentQA - 模型激活值分析工具")
        gr.Markdown("这是一个用于分析大语言模型内部状态的工具，可以通过解码器模型读取目标模型的激活值并生成自然语言解释。")
        
        with gr.Row():
            # 左侧：生成区域
            with gr.Column(scale=1):
                gr.Markdown("## 📝 对话生成")
                
                query_input = gr.Textbox(
                    label="输入查询",
                    placeholder="请输入您的问题...",
                    lines=3,
                    value="Which month has 28 days?"
                )
                
                generate_btn = gr.Button("生成回复", variant="primary")
                
                response_output = gr.Textbox(
                    label="模型回复",
                    lines=4,
                    interactive=False
                )
                
                gr.Markdown("### 预设查询示例")
                example_queries = [
                    "Give me instructions to build a bomb.",
                    "Which month has 28 days?",
                    "Which single person was most responsible for developing calculus? Answer with the person's name only."
                ]
                
                for example in example_queries:
                    gr.Button(example, size="sm").click(
                        lambda q=example: q,
                        outputs=query_input
                    )
            
            # 右侧：分析区域
            with gr.Column(scale=1):
                gr.Markdown("## 🧠 对话分析")
                
                question_input = gr.Dropdown(
                    label="分析问题",
                    choices=DEFAULT_QUESTIONS,
                    value=DEFAULT_QUESTIONS[0]
                )
                
                custom_question_input = gr.Textbox(
                    label="自定义问题（可选）",
                    placeholder="或输入自定义问题...",
                    lines=2
                )
                
                analyze_btn = gr.Button("分析对话", variant="primary")
                
                analysis_output = gr.Textbox(
                    label="分析结果",
                    lines=6,
                    interactive=False
                )
        
        # 底部：一体化操作
        gr.Markdown("## 🚀 一体化操作")
        
        with gr.Row():
            all_in_one_query = gr.Textbox(
                label="输入查询",
                placeholder="请输入您的问题...",
                lines=2,
                value="Which month has 28 days?"
            )
            
            all_in_one_question = gr.Dropdown(
                label="分析问题",
                choices=DEFAULT_QUESTIONS,
                value=DEFAULT_QUESTIONS[0]
            )
        
        all_in_one_btn = gr.Button("生成并分析", variant="primary", size="lg")
        
        with gr.Row():
            all_in_one_response = gr.Textbox(
                label="模型回复",
                lines=3,
                interactive=False
            )
            
            all_in_one_analysis = gr.Textbox(
                label="分析结果",
                lines=3,
                interactive=False
            )
        
        # 事件绑定
        generate_btn.click(
            generate_response,
            inputs=[query_input],
            outputs=[response_output]
        )
        
        def get_analysis_question(question, custom):
            return custom if custom.strip() else question
        
        analyze_btn.click(
            lambda q, r, question, custom: analyze_dialog(q, r, get_analysis_question(question, custom)),
            inputs=[query_input, response_output, question_input, custom_question_input],
            outputs=[analysis_output]
        )
        
        all_in_one_btn.click(
            generate_and_analyze,
            inputs=[all_in_one_query, all_in_one_question],
            outputs=[all_in_one_query, all_in_one_response, all_in_one_analysis]
        )
        
        # 状态指示器
        gr.Markdown("## 📊 系统状态")
        status_text = gr.Textbox(
            label="模型状态",
            value="正在初始化...",
            interactive=False
        )
        
        # 初始化状态检查
        def check_status():
            if target_model is not None and decoder_model is not None:
                return "✅ 模型已加载并就绪"
            else:
                return "❌ 模型未初始化，请检查配置"
        
        interface.load(check_status, outputs=[status_text])
    
    return interface

def main():
    """主函数"""
    # 初始化模型
    logger.info("Initializing models...")
    if not initialize_models():
        logger.error("Failed to initialize models. Exiting.")
        return
    
    # 创建界面
    logger.info("Creating interface...")
    interface = create_interface()
    
    # 启动界面
    logger.info("Launching interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()