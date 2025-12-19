import random
from itertools import islice
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from lit.utils.infra_utils import get_model_config_name

###################################
###### Tokens and formatting ######
###################################

IGNORE_IDX = -100

NUM_READ_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 1,
    "meta-llama/Llama-3.1-8B-Instruct": 1,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 2,
    "google/gemma-3-4b-it": 1,
    "Qwen/Qwen3-4B-Instruct-2507": 0,
}

# Magic numbers that are the length of the user tag + BOS token
NUM_WRITE_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 5,
    "meta-llama/Llama-3.1-8B-Instruct": 5,
    "meta-llama/Meta-Llama-3-70B-Instruct": 5,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 2,
    "google/gemma-3-4b-it": 4,
    "Qwen/Qwen3-4B-Instruct-2507": 3,
}

PAD_TOKEN_IDS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 128010,
    "meta-llama/Llama-3.1-8B-Instruct": 128010,
    "meta-llama/Meta-Llama-3-70B-Instruct": 128010,
    "mistralai/Ministral-8B-Instruct-2410": 999,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 128010,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 151643,
    "google/gemma-3-4b-it": 0,
    "Qwen/Qwen3-4B-Instruct-2507": 151643,
}

# Magic numbers that correspond to the token idxs of the chat format for the models
# <bos>, <user>, <assistant>, <assistant_with_reflection>
CHAT_FORMAT_TOKENS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        torch.tensor([128006, 9125, 128007, 271]),
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        torch.tensor([128006, 9125, 128007, 271]),
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Meta-Llama-3-70B-Instruct": (
        torch.tensor([128006, 9125, 128007, 271]),
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "mistralai/Ministral-8B-Instruct-2410": (
        None,
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    "mistralai/Mistral-Small-24B-Instruct-2501": (
        None,
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    "google/gemma-3-4b-it": (
        None,
        torch.tensor([105, 2364, 107]),
        torch.tensor([105, 4368, 107]),
        torch.tensor([105, 48409, 107]),
    ),
    "Qwen/Qwen3-4B-Instruct-2507": (
        torch.tensor([151644, 8948, 198]),
        torch.tensor([151644, 872, 198]),
        torch.tensor([151644, 77091, 198]),
        torch.tensor([151644, 34913, 198]),
    )
}

# Reasoning models need to encode their thought tokens
ENCODER_CHAT_TEMPLATES = {
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
}
DECODER_CHAT_TEMPLATES = {
    "Qwen/Qwen3-4B-Instruct-2507": "{%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + 'reflect' + '\\n' + content }}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>reflect\\n' }}\n{%- endif %}",
    "google/gemma-3-4b-it": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"reflect\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>reflect\n'}}\n{%- endif -%}\n",
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "meta-llama/Meta-Llama-3-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Meta-Llama-3-70B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
}

# Dialog formats for the dataset
BASE_DIALOG = [
    {
        "role": "assistant",
        "content": "Sure, I've analyzed the assistant.",
    }
]

################################
###### Activation masking ######
################################


def mask_inputs(
    input_ids,
    tokenizer_name,
    mask_type=None,
    shift_start=False,
    mask_all_but_last=False,
    modify_chat_template=False,
):
    sys_tokens, start_tokens, end_tokens_default, end_tokens_modify = CHAT_FORMAT_TOKENS[
        get_model_config_name(tokenizer_name)
    ]
    end_tokens = end_tokens_modify if modify_chat_template else end_tokens_default
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(batch_size):
        sys_idx = []
        start_idx = []
        end_idx = []
        for i in range(seq_len):
            if torch.equal(input_ids[b][i : i + len(sys_tokens)], sys_tokens):
                sys_idx.append(i)
            if torch.equal(input_ids[b][i : i + len(start_tokens)], start_tokens):
                start_idx.append(i)
            if torch.equal(input_ids[b][i : i + len(end_tokens)], end_tokens):
                end_idx.append(i)
        #    工作原理：
        #    - 遍历序列的每个位置 i
        #    - 检查从位置 i 开始的子序列是否等于目标 token 序列
        #    - 如果匹配，记录起始位置 i
        
        # print(f"Batch {b} - sys_idx: {sys_idx}, start_idx: {start_idx}, end_idx: {end_idx}")

        if mask_type is None:
            if len(start_idx) != len(end_idx):
                # Data is improperly formatted so mask everything and skip this item
                mask[b][:] = True
                continue

            if mask_all_but_last:
                mask[b][: end_idx[-1] + len(end_tokens)] = True
            else:
                for i, (start, end) in enumerate(zip(start_idx, end_idx)):
                    if shift_start and i == 0:
                        mask[b][start - 1 : end + len(end_tokens)] = True
                    else:
                        mask[b][start : end + len(end_tokens)] = True
                # 掩码所有的 user 的话：“<|start_header_id|>user<|end_header_id|>
                # 
                # xxx<|eot_id|><|start_header_id|>assistant<|end_header_id|>“

        elif mask_type[b] == "user":
            if len(start_idx) == 1:
                continue
            mask[b][start_idx[0] : start_idx[1]] = True
            # 仅掩码第一个turn
        elif mask_type[b] == "system":
            if len(sys_idx) == 0:
                raise ValueError("No system message found to mask! This is usually because the model does not have a specific system turn, but your dataset does. You should not use the dataset with system turns for this particular model")
            mask[b][sys_idx[0] : start_idx[0]] = True
            # 仅掩码系统消息
        else:
            raise ValueError(f"Invalid verb mask: {mask_type[b]}")
    
    # 打印完整的 input_ids 字符串，mask 为 True 的部分用蓝色标识
    if hasattr(mask_inputs, '_debug_print') and mask_inputs._debug_print:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # ANSI 颜色代码
            BLUE = '\033[94m'
            RESET = '\033[0m'
            
            print("\n" + "="*60)
            print("🔍 MASK DEBUG INFO")
            print("="*60)
            
            for b in range(batch_size):
                print(f"\n📦 Batch {b}:")
                
                # 解码整个序列
                full_text = tokenizer.decode(input_ids[b].tolist())
                
                # 获取 mask 为 True 的位置
                true_indices = torch.where(mask[b])[0].tolist()
                
                if true_indices:
                    print(f"🎯 Mask True positions: {true_indices}")
                    
                    # 构建带颜色的字符串
                    colored_parts = []
                    current_pos = 0
                    
                    for idx in range(len(input_ids[b])):
                        if mask[b][idx]:
                            # 当前位置需要着色
                            token_text = tokenizer.decode([input_ids[b][idx].item()])
                            colored_parts.append(f"{BLUE}{token_text}{RESET}")
                        else:
                            # 普通文本
                            token_text = tokenizer.decode([input_ids[b][idx].item()])
                            colored_parts.append(token_text)
                    
                    colored_text = ''.join(colored_parts)
                    print(f"🔤 Full sequence with mask:")
                    print(f"   {colored_text}")
                    
                    # 统计信息
                    total_tokens = len(input_ids[b])
                    masked_tokens = len(true_indices)
                    print(f"📊 Stats: {masked_tokens}/{total_tokens} tokens masked ({masked_tokens/total_tokens*100:.1f}%)")
                else:
                    print("⚪ No True positions in mask")
                    print(f"🔤 Full sequence:")
                    print(f"   {full_text}")
            
            print("\n" + "="*60)
            print("🏁 END DEBUG INFO")
            print("="*60 + "\n")
        except Exception as e:
            print(f"❌ Debug print failed: {e}")
    
    return mask


def lqa_tokenize(
    batch,
    tokenizer,
    name=None,
    generate=False,
    mask_type=None,
    mask_all_but_last=False,
    modify_chat_template=False,
):
    '''
    Args:
        batch (list): A list of dictionaries, each containing 'read_prompt' and 'dialog
            keys.  
            {
                    "read_prompt": item["read_prompt"],
                    "dialog": item["dialog"],
                    "label": item["dialog"][-1]["content"],
            }

    tokenized_batch = {
        "tokenized_read": tokenized_read,      # 目标模型输入的tokenized结果
       "read_lengths": read_lengths,          # 读取序列的有效长度
        "tokenized_write": tokenized_write,    # 解码器模型输入的tokenized结果
        "write_lengths": write_lengths,        # 写入序列的有效长度
       "verb_lengths": verb_lengths,          # (可选) 
    #  verb部分的长度，仅当mask_type不为None时
    }
    '''
    name = tokenizer.name_or_path if name is None else name
    TOKENIZER_HAS_BOS = any(
        [
            m in name.lower()
            for m in ["gemma", "mistral", "llama-3", "deepseek-r1-distill"]
        ]
    )

    # Tokenize read inputs
    tokenized_read = tokenizer(
        [item["read_prompt"] for item in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch = {"tokenized_read": tokenized_read}

    # Compute length of read input and maybe add verb_lengths
    read_lengths = torch.sum(tokenized_read.attention_mask, dim=1)
    # Exclude BOS token if present
    tokenized_batch["read_lengths"] = read_lengths - (1 if TOKENIZER_HAS_BOS else 0)

    if mask_type is not None:
        verb_mask = mask_inputs(tokenized_read.input_ids, name, mask_type=mask_type)
        verb_lengths = torch.sum(verb_mask, dim=1)
        pad_lengths = read_lengths - verb_lengths
        tokenized_batch["verb_lengths"] = verb_lengths
    else:
        pad_lengths = read_lengths

    # Tokenize dialog inputs
    queries = []
    for i in range(len(pad_lengths)):
        query = [
            {
                "role": "user",
                "content": "? " * (pad_lengths[i] - NUM_READ_TOKENS_TO_SHIFT[get_model_config_name(name)]),
            }
        ]
        query += batch[i]["dialog"]
        queries.append(
            tokenizer.apply_chat_template(
                query,
                tokenize=False,
                add_generation_prompt=generate,
                chat_template=(
                    DECODER_CHAT_TEMPLATES[get_model_config_name(name)] if modify_chat_template else None
                ),
            )
        )
    tokenized_write = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch["tokenized_write"] = tokenized_write

    # Compute length of write input
    write_lengths = torch.sum(tokenized_write.attention_mask, dim=1)
    tokenized_batch["write_lengths"] = write_lengths - NUM_WRITE_TOKENS_TO_SHIFT[get_model_config_name(name)]

    # Add labels for training
    if not generate:
        user_inputs_mask = mask_inputs(
            tokenized_write.input_ids,
            name,
            mask_type=None,
            shift_start=TOKENIZER_HAS_BOS,
            mask_all_but_last=mask_all_but_last,
            modify_chat_template=modify_chat_template,
        )
        assert tokenizer.padding_side == "left"
        tokenized_write["labels"] = tokenized_write.input_ids.clone()
        mask = (tokenized_write.attention_mask == 0) | user_inputs_mask
        tokenized_write["labels"][mask] = IGNORE_IDX
    return tokenized_batch


###########################
####### Dataloading #######
###########################


class LatentQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_system,
        data_stimulus_completion,
        data_stimulus,
        data_control,
        qa_data,
        add_thought_tokens=False,
    ):
        self.tokenizer = tokenizer
        self.data = [
            data_system[0],
            data_stimulus_completion[0],
            data_stimulus[0],
            data_control[0],
        ]
        self.id_tuples = [
            data_system[1],
            data_stimulus_completion[1],
            data_stimulus[1],
            data_control[1],
        ]
        self.labels = [
            list(data_system[0].keys()),
            list(data_stimulus_completion[0].keys()),
            list(data_stimulus[0].keys()),
            list(data_control[0].keys()),
        ]
        self.qa_data = qa_data
        self.add_thought_tokens = add_thought_tokens
        self.chat_template = ENCODER_CHAT_TEMPLATES.get(get_model_config_name(tokenizer.name_or_path), None)
        self.lengths = []
        for idx in range(self.__len__()):
            behavior, qa = self.get_behavior_qa(idx)
            self.lengths.append(
                sum([len(s) for s in behavior]) + sum([len(q) for q in qa])
            )

    def get_behavior_qa(self, idx):
        if idx < len(self.id_tuples[0]):
            j = 0
        elif idx < len(self.id_tuples[0]) + len(self.id_tuples[1]):
            j = 1
            idx -= len(self.id_tuples[0])
        elif idx < len(self.id_tuples[0]) + len(self.id_tuples[1]) + len(
            self.id_tuples[2]
        ):
            j = 2
            idx -= len(self.id_tuples[0]) + len(self.id_tuples[1])
        else:
            j = 3
            idx -= (
                len(self.id_tuples[0]) + len(self.id_tuples[1]) + len(self.id_tuples[2])
            )
        label_idx, data_idx, qa_idx = self.id_tuples[j][idx]
        label = self.labels[j][label_idx]
        return self.data[j][label][data_idx], self.qa_data[label][qa_idx]

    def __len__(self):
        return sum([len(id_tuples) for id_tuples in self.id_tuples])

    def __getitem__(self, idx):
        behavior, qa = self.get_behavior_qa(idx)
        (
            system,
            control_user,
            control_thought,
            control_model,
            stimulus_user,
            stimulus_thought,
            stimulus_model,
        ) = behavior
        if system != "":
            assert control_user == control_model == stimulus_model == ""
            read_prompt = [
                {"role": "system", "content": system},
                {"role": "user", "content": stimulus_user},
            ]
            add_generation_prompt = True
            mask_type = "system"
        elif control_model == "":
            assert stimulus_user == stimulus_model == ""
            read_prompt = [{"role": "user", "content": control_user}]
            add_generation_prompt = True
            mask_type = "user"
        elif stimulus_model == "":
            read_prompt = [
                {"role": "user", "content": control_user},
                {"role": "assistant", "content": control_model},
                {"role": "user", "content": stimulus_user},
            ]
            add_generation_prompt = True
            mask_type = "user"
        else:
            if self.add_thought_tokens:
                read_prompt = [
                    {"role": "user", "content": control_user},
                    {"role": "assistant", "content": control_model},
                    {"role": "user", "content": stimulus_user},
                    {
                        "role": "assistant",
                        "content": f"<think>\n{stimulus_thought}\n</think>\n\n{stimulus_model}",
                    },
                ]
                add_generation_prompt = False
                mask_type = "user"
            else:
                read_prompt = [
                    {"role": "user", "content": control_user},
                    {"role": "assistant", "content": control_model},
                    {"role": "user", "content": stimulus_user},
                    {"role": "assistant", "content": stimulus_model},
                ]
                add_generation_prompt = False
                mask_type = "user"
        read_prompt = self.tokenizer.apply_chat_template(
            read_prompt,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template,
        )
        qa_dialog = [
            {"role": "user", "content": qa[0]},
            {"role": "assistant", "content": qa[1]},
        ]
        return {
            "read_prompt": read_prompt,
            "dialog": BASE_DIALOG + qa_dialog,
            "mask_type": mask_type,
        }


class DataCollatorForLatentQA:
    def __init__(
        self,
        tokenizer,
        generate=False,
        mask_all_but_last=False,
        nudge_persona=False,
        modify_chat_template=False,
    ):
        self.tokenizer = tokenizer
        self.generate = generate
        self.mask_all_but_last = mask_all_but_last
        self.nudge = "Base your answers on my instructions. " if nudge_persona else ""
        self.modify_chat_template = modify_chat_template

    def __call__(self, batch):
        formatted_batch = []
        mask_type = []
        for item in batch:
            formatted_batch.append(
                {
                    "read_prompt": item["read_prompt"],
                    "dialog": item["dialog"],
                    "label": item["dialog"][-1]["content"],
                }
            )
            mask_type.append(item["mask_type"])
        return lqa_tokenize(
            formatted_batch,
            self.tokenizer,
            mask_type=mask_type,
            generate=self.generate,
            mask_all_but_last=self.mask_all_but_last,
            modify_chat_template=self.modify_chat_template,
        )


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True
    ) -> None:
        self.lengths = data_source.lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[: len(ids) // self.batch_size * self.batch_size]

        batches = [
            ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (
                len(self.lengths) % self.batch_size > 0
            )


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        data_source,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


def get_batch_sampler(dataset, train_config, mode):
    return LengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        drop_last=False,
        shuffle=(mode == "train"),
    )


def get_dist_batch_sampler(dataset, train_config, mode):
    if dist.is_initialized():
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
    else:
        num_replicas = 1
        rank = 0
    
    return DistributedLengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=(mode == "train"),
        seed=train_config.seed,
    )


def get_dataset(train_config, tokenizer, train=True):
    FILTER = train_config.filter.split("-")
    qa_path = train_config.train_qa if train else train_config.eval_qa
    with open(qa_path, "r") as f:
        qa_data = json.load(f)
        NUM_QA = max([len(qa_data[label]) for label in qa_data])
        assert NUM_QA == min([len(qa_data[label]) for label in qa_data])

    def build_data_and_idx(path):
        # Get data
        data = defaultdict(list)
        if path == "":
            return data, []
        with open(path, "r") as f:
            raw_data = json.load(f)
            for item in raw_data:
                if item["label"].split("-")[0] in FILTER:
                    continue
                # continue 实现了黑名单过滤机制，FILTER 中指定的类型会被排除，其他类型会被保留
                data[item["label"]].append(
                    (
                        item.get("system", ""),
                        item.get("control_user", ""),
                        item.get("control_thought", ""),
                        item.get("control_model", ""),
                        item.get("stimulus_user", ""),
                        item.get("stimulus_thought", ""),
                        item.get("stimulus_model", ""),
                    )
                )
        # Get id tuples
        NUM_BEHAVIORS = max([len(data[label]) for label in data])
        assert NUM_BEHAVIORS == min([len(data[label]) for label in data])
        id_tuples = range(len(data) * NUM_BEHAVIORS * NUM_QA)
        if train_config.train_percent == 1 or not train:
            id_tuples = list(id_tuples)
        else:
            id_tuples = random.sample(
                id_tuples, int(len(id_tuples) * train_config.train_percent)
            )
        for i in range(len(id_tuples)):
            label_idx = id_tuples[i] // (NUM_BEHAVIORS * NUM_QA)
            data_idx = (id_tuples[i] // NUM_QA) % NUM_BEHAVIORS
            qa_idx = id_tuples[i] % NUM_QA
            id_tuples[i] = (label_idx, data_idx, qa_idx)
        return data, id_tuples

    p0 = train_config.train_system if train else train_config.eval_system
    p1 = (
        train_config.train_stimulus_completion
        if train
        else train_config.eval_stimulus_completion
    )
    p2 = train_config.train_stimulus if train else train_config.eval_stimulus
    p3 = train_config.train_control if train else train_config.eval_control
    data_system = build_data_and_idx(p0)
    data_stimulus_completion = build_data_and_idx(p1)
    data_stimulus = build_data_and_idx(p2)
    data_control = build_data_and_idx(p3)

    return LatentQADataset(
        tokenizer,
        data_system,
        data_stimulus_completion,
        data_stimulus,
        data_control,
        qa_data,
        add_thought_tokens=train_config.add_thought_tokens,
    )


def get_dataloaders(train_config, tokenizer):
    dataset_train = get_dataset(train_config, tokenizer, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=DataCollatorForLatentQA(
            tokenizer,
            mask_all_but_last=False,
            nudge_persona=train_config.nudge_persona,
            modify_chat_template=train_config.modify_chat_template,
        ),
        batch_sampler=get_dist_batch_sampler(dataset_train, train_config, "train"),
    )
    if train_config.eval_ppl:
        dataset_eval = get_dataset(train_config, tokenizer, train=False)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=DataCollatorForLatentQA(
                tokenizer,
                mask_all_but_last=False,
                nudge_persona=train_config.nudge_persona,
                modify_chat_template=train_config.modify_chat_template,
            ),
            batch_sampler=get_dist_batch_sampler(dataset_eval, train_config, "val"),
        )
        return train_dataloader, eval_dataloader

    return train_dataloader, None


def print_dataset_samples(dataloader, tokenizer, num_samples=2, rank=0):
    if rank != 0:
        return
    
    # ANSI 颜色代码
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    
    print("\n" + "="*80)
    print("数据集样本调试信息")
    print("="*80)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_samples:
            break
        
        print(f"\n{'='*80}")
        print(f"批次 {batch_idx + 1}:")
        print(f"{'='*80}")
        
        batch_size = batch["tokenized_read"]["input_ids"].shape[0]
        print(f"批次大小: {batch_size}")
        
        for sample_idx in range(min(2, batch_size)):
            print(f"\n{'-'*80}")
            print(f"样本 {sample_idx + 1}:")
            print(f"{'-'*80}")
            
            print(f"\n{BLUE}[目标模型输入 (Read)]{ENDC}")
            read_input_ids = batch["tokenized_read"]["input_ids"][sample_idx]
            read_attention_mask = batch["tokenized_read"]["attention_mask"][sample_idx]
            read_length = batch["read_lengths"][sample_idx].item()
            
            print(f"  序列长度: {read_input_ids.shape[0]}")
            print(f"  Read Lengths: {read_length}")
            print(f"  有效 token 数: {torch.sum(read_attention_mask).item()}")
            print(f"  Input IDs 形状: {read_input_ids.shape}")
            
            # 打印 Input IDs
            print(f"{GREEN}  Input IDs:{ENDC}")
            ids_list = read_input_ids.tolist()
            for i in range(0, len(ids_list), 20):
                chunk = ids_list[i:i+20]
                ids_str = " ".join(f"{id:6d}" for id in chunk)
                print(f"    {ids_str}")
            
            # 打印 Attention Mask
            print(f"{GREEN}  Attention Mask:{ENDC}")
            mask_list = read_attention_mask.tolist()
            for i in range(0, len(mask_list), 20):
                chunk = mask_list[i:i+20]
                mask_str = " ".join(f"{mask:6d}" for mask in chunk)
                print(f"    {mask_str}")
            
            # 打印解码文本
            valid_tokens = read_input_ids[read_attention_mask.bool()]
            decoded_text = tokenizer.decode(valid_tokens, skip_special_tokens=False)
            print(f"{GREEN}  解码文本:{ENDC}")
            for line in decoded_text.split('\n'):
                print(f"    {line}")
            
            if "verb_lengths" in batch:
                verb_length = batch["verb_lengths"][sample_idx].item()
                print(f"  Verb Lengths: {verb_length}")
                print(f"  Pad Lengths: {read_length - verb_length}")
            
            print(f"\n{BLUE}[解码器模型输入 (Write)]{ENDC}")
            write_input_ids = batch["tokenized_write"]["input_ids"][sample_idx]
            write_attention_mask = batch["tokenized_write"]["attention_mask"][sample_idx]
            write_length = batch["write_lengths"][sample_idx].item()
            
            print(f"  序列长度: {write_input_ids.shape[0]}")
            print(f"  Write Lengths: {write_length}")
            print(f"  有效 token 数: {torch.sum(write_attention_mask).item()}")
            print(f"  Input IDs 形状: {write_input_ids.shape}")
            
            # 打印 Input IDs
            print(f"{GREEN}  Input IDs:{ENDC}")
            ids_list = write_input_ids.tolist()
            for i in range(0, len(ids_list), 20):
                chunk = ids_list[i:i+20]
                ids_str = " ".join(f"{id:6d}" for id in chunk)
                print(f"    {ids_str}")
            
            # 打印 Attention Mask
            print(f"{GREEN}  Attention Mask:{ENDC}")
            mask_list = write_attention_mask.tolist()
            for i in range(0, len(mask_list), 20):
                chunk = mask_list[i:i+20]
                mask_str = " ".join(f"{mask:6d}" for mask in chunk)
                print(f"    {mask_str}")
            
            # 打印解码文本
            valid_write_tokens = write_input_ids[write_attention_mask.bool()]
            decoded_write_text = tokenizer.decode(valid_write_tokens, skip_special_tokens=False)
            print(f"{GREEN}  解码文本:{ENDC}")
            for line in decoded_write_text.split('\n'):
                print(f"    {line}")
            
            if "labels" in batch["tokenized_write"]:
                labels = batch["tokenized_write"]["labels"][sample_idx]
                print(f"\n{YELLOW}[标签信息]{ENDC}")
                print(f"  Labels 形状: {labels.shape}")
                
                # 打印 Labels
                print(f"{GREEN}  Labels:{ENDC}")
                labels_list = labels.tolist()
                for i in range(0, len(labels_list), 20):
                    chunk = labels_list[i:i+20]
                    labels_str = " ".join(f"{label:6d}" for label in chunk)
                    print(f"    {labels_str}")
                
                # 创建标签的副本，将 IGNORE_IDX 替换为 padding token ID
                labels_for_decode = labels.clone()
                labels_for_decode[labels == IGNORE_IDX] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                
                # 使用 attention_mask 来正确解码标签
                valid_labels = labels_for_decode[write_attention_mask.bool()]
                
                if len(valid_labels) > 0:
                    print(f"  非忽略标签数量: {len(labels[labels != IGNORE_IDX])}")
                    decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=False)
                    print(f"{GREEN}  解码标签:{ENDC}")
                    for line in decoded_labels.split('\n'):
                        print(f"    {line}")
                    
                    # 同时也打印原始非忽略标签的解码结果，以便对比
                    non_ignored_labels = labels[labels != IGNORE_IDX]
                    if len(non_ignored_labels) > 0:
                        decoded_non_ignored = tokenizer.decode(non_ignored_labels, skip_special_tokens=False)
                        print(f"{GREEN}  仅非忽略标签解码:{ENDC}")
                        for line in decoded_non_ignored.split('\n'):
                            print(f"    {line}")
                else:
                    print(f"  非忽略标签数量: 0")
        
        print(f"\n{'='*80}\n")
    
    print("数据集样本调试信息打印完成")
    print("="*80 + "\n")
