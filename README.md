
## Getting Started

### Installation

```bash
pip install -e . && pip install -r requirements/metrics.txt -r requirements/deepspeed.txt
```

### Finetuning

```bash
llamafactory-cli train examples/train_full/tennis_expert_qwen3vl_full_sft.yaml
```


### Evaluation

```bash
python inference.py \
  --model-path saves/qwen3-vl-8b/sft \
  --input-file data/tennis_data_test_stats_.json \
  --output-file path/to/save/inference_result.json \
  --window-size 5 \
  --video-fps 2.0 \
  --max-new-tokens 128 \

python llm_eval.py \
  --pred_path data/tennis_inference_result.json \
  --output_dir path/to/save \
  --output_json path/to/save/llm_eval_result.json \
  --model gemini-3-flash-preview \
  --api_key '' \
```

