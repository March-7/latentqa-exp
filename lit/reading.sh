CUDA_VISIBLE_DEVICES=2 python3 -m lit.reading \
   --target_model_name /data1/ckx/hf-checkpoints/meta-llama/Llama-3.1-8B-Instruct \
   --decoder_model_name out/runs/000/checkpoints/epoch4-steps324200-2025-12-18_14-12-46 \
   --prompt "Answer the question based on the provided context."