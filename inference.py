import argparse
import json
import math
import os
import time
import uuid
from copy import deepcopy

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoProcessor

from llamafactory.model.custom.qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU inference with sliding-window memory for Qwen3-VL."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="saves/qwen3-vl-8b/sft",
        help="Path to the pretrained model."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/tennis_data_test_stats_.json",
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/tennis_inference_result.json",
        help="Path to the final merged output JSON file."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Sliding window size for historical turns."
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=2.0,
        help="Video sampling FPS. Keep this low to avoid OOM."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation for model loading."
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used to load the model."
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_str}")


def get_prefixes(input_file: str):
    """
    Determine memory/current prefixes based on the input filename.
    """
    if "window_1" not in input_file:
        mem_prefix = "\nVideo and metadata of several past rallies: "
        cur_prefix = "\nVideo and metadata of current rally: "
    else:
        mem_prefix = ""
        cur_prefix = ""
    return mem_prefix, cur_prefix


def run_inference_on_gpu(gpu_id, chunk_data, temp_output_file, config):
    """
    Run inference in a dedicated process on a single GPU.
    The sliding-window logic is executed within this process.
    """
    print(f"[GPU {gpu_id}] Initializing model on cuda:{gpu_id}...")

    torch_dtype = get_torch_dtype(config["torch_dtype"])

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_path"],
        torch_dtype=torch_dtype,
        device_map=f"cuda:{gpu_id}",
        attn_implementation=config["attn_implementation"]
    ).eval()

    processor = AutoProcessor.from_pretrained(config["model_path"])

    mem_prefix = config["mem_prefix"]
    cur_prefix = config["cur_prefix"]
    window_size = config["window_size"]
    video_fps = config["video_fps"]
    max_new_tokens = config["max_new_tokens"]

    with open(temp_output_file, "w", encoding="utf-8") as f_out:
        # Iterate over each match assigned to this GPU
        for item_idx, item in enumerate(tqdm(chunk_data, desc=f"GPU {gpu_id}", position=gpu_id)):
            conversations = item.get("conversations", [])
            if not conversations:
                continue

            video_path_list = item.get("videos", [])
            if isinstance(video_path_list, str):
                video_path_list = [video_path_list]
            video_counter = 0

            # Extract the system prompt
            if conversations[0]["from"] == "system":
                system_prompt = {
                    "role": "system",
                    "content": conversations[0]["value"]
                }
                start_idx = 1
            else:
                system_prompt = {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
                start_idx = 0

            # Reset the history buffer for each new match
            turn_history_buffer = []

            # Process each turn sequentially within the current match
            for i in range(start_idx, len(conversations)):
                turn = conversations[i]

                if turn["from"] != "human":
                    continue

                has_video = "<video>" in turn["value"]
                raw_human_text = turn["value"].replace("<video>", "").strip()

                current_video_path = None
                if has_video and video_counter < len(video_path_list):
                    current_video_path = video_path_list[video_counter]
                    video_counter += 1

                # =================================================
                # Build sliding-window history
                # =================================================
                valid_history_slice = (
                    turn_history_buffer[-window_size:] if window_size > 0 else []
                )
                messages = [system_prompt]

                for hist_idx, hist_turn in enumerate(valid_history_slice):
                    user_content = deepcopy(hist_turn["human_content"])

                    # Add MEM_PREFIX to the first text segment of the first history turn
                    if hist_idx == 0:
                        for part in user_content:
                            if part["type"] == "text":
                                part["text"] = mem_prefix + part["text"]
                                break

                    messages.append({"role": "user", "content": user_content})
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": hist_turn["gpt_text"]}]
                    })

                # =================================================
                # Build the current turn
                # =================================================
                current_user_content = []

                if current_video_path and os.path.exists(current_video_path):
                    current_user_content.append({
                        "type": "video",
                        "video": current_video_path,
                        "fps": video_fps
                    })

                current_user_content.append({
                    "type": "text",
                    "text": cur_prefix + raw_human_text
                })
                messages.append({"role": "user", "content": current_user_content})

                # =================================================
                # Generate prediction
                # =================================================
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                prediction_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                # =================================================
                # Extract ground truth and save result
                # =================================================
                if i + 1 < len(conversations):
                    from_gpt = conversations[i + 1]["value"]
                    if "\n\nMetadata: " in from_gpt:
                        commentary_gt, metadata_gt = from_gpt.split("\n\nMetadata: ", 1)
                    else:
                        commentary_gt, metadata_gt = from_gpt, ""
                else:
                    commentary_gt, metadata_gt = "", ""

                result = {
                    "video": current_video_path,
                    "prompt": raw_human_text,
                    "prediction": prediction_text,
                    "commentary_gt": commentary_gt,
                    "metadata_gt": metadata_gt
                }

                # Append immediately to the temporary JSONL file
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

                # =================================================
                # Update history memory with clean content (no prefix)
                # =================================================
                clean_human_content = []
                if current_video_path and os.path.exists(current_video_path):
                    clean_human_content.append({
                        "type": "video",
                        "video": current_video_path,
                        "fps": video_fps
                    })
                clean_human_content.append({
                    "type": "text",
                    "text": raw_human_text
                })

                turn_history_buffer.append({
                    "human_content": clean_human_content,
                    "gpt_text": prediction_text
                })


def main():
    args = parse_args()

    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found!")

    print(f"Detected {num_gpus} GPUs. Preparing data split...")

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Split data at the match/item level to ensure one match is not split across GPUs
    chunk_size = math.ceil(len(data) / num_gpus)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Create a unique run ID
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    print(f"Assigned unique run ID: {run_id}")

    mem_prefix, cur_prefix = get_prefixes(args.input_file)

    config = {
        "model_path": args.model_path,
        "window_size": args.window_size,
        "video_fps": args.video_fps,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "mem_prefix": mem_prefix,
        "cur_prefix": cur_prefix,
    }

    # Create a temporary JSONL file for each GPU
    temp_files = [f"temp_gpu_{i}_{run_id}.jsonl" for i in range(num_gpus)]
    processes = []

    # Launch multiprocessing workers
    for gpu_id in range(min(num_gpus, len(chunks))):
        p = mp.Process(
            target=run_inference_on_gpu,
            args=(gpu_id, chunks[gpu_id], temp_files[gpu_id], config)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll GPUs have finished. Merging results into a single JSON file...")

    # Merge temporary JSONL files into a standard JSON array
    all_results = []
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            with open(temp_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.strip():
                        all_results.append(json.loads(line.strip()))
            os.remove(temp_file)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(all_results, outfile, indent=2, ensure_ascii=False)

    print(f"Parallel inference with sliding window completed.")
    print(f"Standard JSON saved to: {args.output_file}")


if __name__ == "__main__":
    main()
