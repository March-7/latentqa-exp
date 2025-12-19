CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc-per-node 1 -m lit.train \
    --target_model_name /data1/ckx/hf-checkpoints/meta-llama/Llama-3.1-8B-Instruct \
    --train_stimulus_completion data/train/stimulus_completion.json \
    --train_stimulus data/train/stimulus.json \
    --train_control data/train/control.json \
    --train_qa data/train/qa.json \
    --batch_size_training 1 \
    --gradient_accumulation_steps 16 \
    --use_swanlab