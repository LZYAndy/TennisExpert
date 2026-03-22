# TennisExpert: Towards Expert-Level Analytical Sports Video Understanding

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](#)

This repository contains the official implementation of **Tennis Expert**, a comprehensive multimodal tennis understanding framework. It integrates a fine-grained video semantic parser with a memory-augmented model built on Qwen3-VL-8B to generate expert-level analytical commentary for professional tennis matches. 

This repository also provides instructions for accessing **Tennis VL**, the largest and most richly annotated tennis video benchmark to date, comprising over 200 professional matches (471.9 hours) and 40,000+ rally-level clips.

## 🌟 Key Features
* **Tennis VL Dataset:** A large-scale benchmark emphasizing expert analytical commentary, tactical reasoning, player decisions, and match momentum.
* **Video Semantic Parser:** Extracts structured key match elements including scores, shot sequences, ball bounces, and player locations in real time.
* **Long Short-Term Memory Mechanism:** Hierarchical memory modules that capture both immediate match momentum and long-term cumulative player statistics.

## 🛠️ Environment Setup

We recommend using a virtual environment (e.g., Conda) to manage dependencies.

```bash
# Clone the repository
git clone [https://github.com/yourusername/TennisExpert.git](https://github.com/yourusername/TennisExpert.git)
cd TennisExpert

# Create and activate a conda environment
conda create -n tennis_expert python=3.11 -y
conda activate tennis_expert

# Install the package and dependencies
pip install -r requirements.txt
```

## 📊 Dataset Preparation (Tennis VL)
The Tennis VL dataset contains JSON annotations and corresponding video clips. 


## 🚀 Getting Started
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

