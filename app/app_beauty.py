#!/usr/bin/env python3
"""
LatentQA Gradio Interface
一个用于分析和解释大语言模型激活值的Web界面 - Chat Layout Refactor
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

# -----------------------------------------------------------------------------
# 导入部分保持不变 (依赖 lit 库)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 全局变量与初始化 (保持不变)
# -----------------------------------------------------------------------------
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
    "Is the assistant experiencing any dilemmas?",
    "Is the conversation truthful?",
    "Why did the model choose this answer?",
    "Is the assistant deceiving the user?"
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
        
        # 初始化tokenizer
        logger.info("Loading tokenizer from local path...")
        try:
            tokenizer = get_tokenizer(args.target_model_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
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
        
        logger.info("Loading target model...")
        target_model = get_model(args.target_model_name, tokenizer, device=device)
        
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

# -----------------------------------------------------------------------------
# 核心逻辑函数 (保持不变)
# -----------------------------------------------------------------------------
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
        dialog = [query, response]
        generate = True
        probe_data = []
        
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
        
        analysis_dialog = [{"role": "user", "content": question}]
        
        probe_data.append({
            "read_prompt": read_prompt,
            "dialog": BASE_DIALOG + analysis_dialog,
        })
        
        batch = lqa_tokenize(
            probe_data,
            tokenizer,
            name=args.target_model_name,
            generate=generate,
            mask_type=None,
            modify_chat_template=args.modify_chat_template,
            mask_all_but_last=True,
        )
        
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
        
        num_tokens = batch["tokenized_write"]["input_ids"][0].shape[0]
        completion = tokenizer.decode(out[0][num_tokens:], skip_special_tokens=True)
        
        return completion.strip()
        
    except Exception as e:
        logger.error(f"Error analyzing dialog: {e}")
        return f"分析错误: {str(e)}"

# -----------------------------------------------------------------------------
# 新的 UI 交互逻辑函数
# -----------------------------------------------------------------------------

def user_submit_query(user_message, history):
    """左侧：用户发送消息"""
    return "", history + [{"role": "user", "content": user_message}]

def bot_generate_response(history):
    """左侧：模型生成回复"""
    if not history:
        return history, "", ""
    
    query = history[-1]["content"] if history[-1]["role"] == "user" else ""
    response = generate_response(query)
    
    # 更新历史记录 - 添加assistant回复
    history.append({"role": "assistant", "content": response})
    
    # 返回: 更新的左侧历史, 更新后的query状态, 更新后的response状态
    return history, query, response

def trigger_decoder_context(query, response, decoder_history):
    """左侧生成完毕后，自动向右侧添加一个上下文指示气泡"""
    # 模拟图片中蓝色的 [Llama Activations from "..."]
    # 我们用 Markdown 加粗来模拟一种特殊的系统状态提示
    display_text = f"**[Llama Activations from \"{query[:30]}...\"]**"
    
    # 添加到右侧聊天记录中，作为系统消息
    if not query:
        return decoder_history
        
    return decoder_history + [{"role": "system", "content": display_text}]

def decoder_ask_question(question, history, current_query, current_response):
    """右侧：用户提问分析"""
    if not current_query or not current_response:
        return "", history + [{"role": "user", "content": question}, {"role": "assistant", "content": "⚠️ 请先在左侧生成模型对话，再进行分析。"}]
    
    # 添加用户问题到历史
    history = history + [{"role": "user", "content": question}]
    return "", history

def decoder_generate_answer(history, current_query, current_response):
    """右侧：解码器生成解释"""
    if not history:
        return history
    
    # 找到最后一条用户消息
    last_user_message = None
    last_message_index = -1
    for i in range(len(history) - 1, -1, -1):
        # 检查消息格式，处理新旧格式兼容
        if isinstance(history[i], dict) and history[i].get("role") == "user":
            last_user_message = history[i]["content"]
            last_message_index = i
            break
        elif isinstance(history[i], list) and len(history[i]) >= 1:
            # 旧格式 [user_message, bot_response]
            last_user_message = history[i][0]
            last_message_index = i
            break
    
    if not last_user_message:
        return history
    
    # 检查类型并处理
    if not isinstance(last_user_message, str):
        logger.warning(f"last_user_message is not a string: {type(last_user_message)} - {last_user_message}")
        return history
    
    # 如果是上下文指示条（我们之前添加的 fake message），跳过
    if last_user_message.startswith("**[Llama Activations"):
        return history

    # 检查是否已经有答案（下一条消息）
    if (last_message_index + 1 < len(history) and 
        history[last_message_index + 1]["role"] == "assistant"):
        return history

    if not current_query or not current_response:
        # 添加错误消息
        history.insert(last_message_index + 1, {"role": "assistant", "content": "Context missing."})
        return history

    # 调用核心分析函数
    analysis_result = analyze_dialog(current_query, current_response, last_user_message)
    # 添加分析结果
    history.insert(last_message_index + 1, {"role": "assistant", "content": analysis_result})
    return history

def clear_all():
    """清空所有状态"""
    return [], "", "", ""

# 自定义 CSS 以匹配图片风格 (蓝色标题，圆角边框)
custom_css = """
.chat-header {
    font-weight: bold;
    font-size: 1.2em;
    margin-bottom: 5px;
    color: #2563eb; /* Blue color similar to image */
}
.chat-subtitle {
    font-size: 0.9em;
    color: #6b7280;
    margin-bottom: 10px;
}
.contain { display: flex; flex-direction: column; height: 100%; }
"""

# -----------------------------------------------------------------------------
# 界面构建 (重构版)
# -----------------------------------------------------------------------------
def create_interface():

    with gr.Blocks(title="LatentQA Interface") as interface:
        
        # 状态变量，用于在左右两侧传递上下文
        state_query = gr.State("")
        state_response = gr.State("")

        with gr.Row(equal_height=True):
            
            # --- 左侧：Model Chat ---
            with gr.Column(variant="panel"):
                gr.HTML("""
                <div class="chat-header">Model Chat (Llama-3.1-8B-Instruct)</div>
                <div class="chat-subtitle">Start by sending a message to Llama!</div>
                """)
                
                # Chatbot 组件
                model_chatbot = gr.Chatbot(
                    label="",
                    height=600
                )                
                # 输入区域
                with gr.Row():
                    msg_input = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message...",
                        container=False,
                        scale=8
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
                    # regenerate_btn = gr.Button("Regenerate", variant="secondary", size="sm") # 可选

            # --- 右侧：Decoder Chat ---
            with gr.Column(variant="panel"):
                gr.HTML("""
                <div class="chat-header">Decoder Chat</div>
                <div class="chat-subtitle">Ask the decoder about Llama's internal state</div>
                """)
                
                decoder_chatbot = gr.Chatbot(
                    label="",
                    height=600,
                )
                
                # 预设问题快捷栏
                with gr.Accordion("Quick Questions", open=False):
                    example_questions = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        label="Click to ask",
                        samples=[[q] for q in DEFAULT_QUESTIONS]
                    )

                # 输入区域
                with gr.Row():
                    decoder_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask about the internal state...",
                        container=False,
                        scale=8
                    )
                    decoder_send_btn = gr.Button("Ask", variant="primary", scale=1, elem_id="green-btn")
                
                # 为了美观，给右侧按钮加点不一样的颜色（可选，通过CSS）
                
        # -----------------------------------------------------------------------------
        # 事件绑定
        # -----------------------------------------------------------------------------
        
        # 1. 左侧对话流程
        # 用户回车或点击发送
        submit_event = msg_input.submit(
            user_submit_query, 
            [msg_input, model_chatbot], 
            [msg_input, model_chatbot]
        ).then(
            bot_generate_response,
            [model_chatbot],
            [model_chatbot, state_query, state_response]
        ).then(
            # 左侧生成完后，触发右侧显示上下文提示
            trigger_decoder_context,
            [state_query, state_response, decoder_chatbot],
            [decoder_chatbot]
        )
        
        send_btn.click(
            user_submit_query, 
            [msg_input, model_chatbot], 
            [msg_input, model_chatbot]
        ).then(
            bot_generate_response,
            [model_chatbot],
            [model_chatbot, state_query, state_response]
        ).then(
            trigger_decoder_context,
            [state_query, state_response, decoder_chatbot],
            [decoder_chatbot]
        )

        # 2. 右侧对话流程
        # 快速提问点击
        example_questions.click(
            lambda x: x[0],
            inputs=[example_questions],
            outputs=[decoder_input]
        )

        # 提交分析问题
        decoder_submit_event = decoder_input.submit(
            decoder_ask_question,
            [decoder_input, decoder_chatbot, state_query, state_response],
            [decoder_input, decoder_chatbot]
        ).then(
            decoder_generate_answer,
            [decoder_chatbot, state_query, state_response],
            [decoder_chatbot]
        )
        
        decoder_send_btn.click(
            decoder_ask_question,
            [decoder_input, decoder_chatbot, state_query, state_response],
            [decoder_input, decoder_chatbot]
        ).then(
            decoder_generate_answer,
            [decoder_chatbot, state_query, state_response],
            [decoder_chatbot]
        )

        # 3. 清除功能
        clear_btn.click(
            clear_all,
            outputs=[model_chatbot, decoder_chatbot, state_query, state_response]
        )
        
        # 状态检查
        status_text = gr.Textbox(visible=False)
        def check_status():
            if target_model is not None and decoder_model is not None:
                return "Ready"
            return "Loading..."
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
        debug=True,
        allowed_paths=["."],
        theme=gr.themes.Soft(),
        css=custom_css
    )
if __name__ == "__main__":
    main()