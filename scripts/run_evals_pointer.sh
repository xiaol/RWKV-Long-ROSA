#!/bin/bash
# wait for the pointer_k4 run, then evaluate it on novels, code and the copy task
cd /root/x/RWKV-Long-ROSA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ps -p 2389914 >/dev/null; do sleep 30; done
python scripts/eval_adapter.py pointer_k4 --data tok --books 12 --max_t 32768 > results/eval_tok_pointer_k4.log 2>&1
python scripts/eval_adapter.py pointer_k4 --data tok_code --books 12 --max_t 32768 > results/eval_code_pointer_k4.log 2>&1
python scripts/copy_eval.py pointer_k4 > results/copy_pointer_k4.log 2>&1
