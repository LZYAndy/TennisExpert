import openai
import os
import argparse
import json
import ast
import time
from multiprocessing.pool import Pool
from tqdm import tqdm
import numpy as np
# --- GEMINI SDK IMPORTS ---
from google import genai
from google.genai import types

# Traditional Metrics Imports
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        Cider = None
        print("[Warning] 'pycocoevalcap' not installed. CIDEr metric will be skipped.")
        print("To install: pip install pycocoevalcap")
    
    # Ensure necessary NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')  
        nltk.download('punkt')      
        nltk.download('wordnet')
        nltk.download('omw-1.4')
except ImportError:
    print("[Warning] NLTK or rouge-score not installed. Traditional metrics will be skipped.")


# ==========================================
# PROMPT TEMPLATE
# ==========================================
SYSTEM_PROMPT = (
    "You are a senior Tennis Analyst and expert commentator evaluator. "
    "Your task is to evaluate a batch of **Generated Commentaries** against strict **Match Metadata** "
    "and **Reference Commentaries** (Ground Truth). "
)

USER_PROMPT_TEMPLATE = """
Evaluate the following batch of {num_samples} tennis commentaries.

### INPUT DATA (Chronological Order):
{batch_data}

### SCORING RUBRIC (0-20 points per category, Total 100):

1. **ACCURACY (0-20 pts):** Alignment with METADATA (players, shot types, score, court positions).
   - 20: Perfect factual match.
   - 0: Hallucinations (wrong player, wrong shot) or contradictions with Metadata.

2. **COHERENCE (0-20 pts):** Logical flow and pronoun usage.
   - 20: Natural narrative; events connect logically.
   - 0: Confusing structure; contradictions within the text.

3. **EXCITEMENT (0-20 pts):** Tone matches the event intensity.
   - 20: Highly engaging; emotive vocabulary fitting the moment.
   - 0: Robotic, flat, or mismatched tone (e.g., boring description of a winner).

4. **PROFESSIONALISM (0-20 pts):**  Domain terminology and depth of analysis.
   - 20: Insightful observation (e.g., noting "inside-out forehand" or "tactical adjustment").
   - 0: Superficial or generic description only.

5. **PACING (0-20 pts):** Length relative to event complexity.
   - 20: Concise for quick points; descriptive for long rallies.
   - 0: Severe mismatch (e.g., long paragraph for a simple double fault).

### OUTPUT INSTRUCTION:
Provide your evaluation **strictly** as a JSON object containing a "results" array.
Do NOT output any markdown or conversational text. 
Ensure EVERY sample provided in the input is evaluated and has a corresponding object in the array with its exact "id".

Format EXACTLY like this:
{{
    "results": [
        {{
            "id": "<sample_id>",
            "scores": {{
                "accuracy": <int>,
                "coherence": <int>,
                "excitement": <int>,
                "professionalism": <int>,
                "pacing": <int>
            }},
            "total_score": <int>
        }}
    ]
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Tennis Commentary Hybrid Evaluation")
    parser.add_argument("--pred_path", required=True, help="Path to your prediction JSON file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save individual results.")
    parser.add_argument("--output_json", required=True, help="Path to save the final merged JSON.")
    parser.add_argument("--eval_mode", default="all", choices=["all", "llm_only", "traditional_only"])
    
    # LLM Settings
    parser.add_argument("--api_key", default="", help="Gemini API key")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model to use")
    parser.add_argument("--num_tasks", default=4, type=int, help="Number of parallel processes.")
    parser.add_argument("--batch_size", default=10, type=int, help="Number of samples to evaluate per LLM request.") 
    return parser.parse_args()

# ==========================================
# TRADITIONAL METRICS FUNCTION
# ==========================================
def compute_traditional_metrics(data_samples):
    """
    Computes BLEU-4, METEOR, ROUGE-L, and CIDEr for the entire dataset.
    """
    print("\n[Traditional] Computing BLEU, METEOR, ROUGE...")
    
    bleu4_scores = []
    meteor_scores = []
    rouge_l_scores = []
    cider_gts = {}
    cider_res = {}
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction()

    for i, sample in tqdm(enumerate(data_samples), total=len(data_samples), desc="Calc Metrics"):

        pred = sample.get('prediction', "")
        ref = sample.get('commentary_gt', "")
        
        # 1. Pre-processing & Tokenization
        # lower case and tokenize
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        
        # 2. BLEU-4 (Sample-level)
        try:
            b4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        except:
            b4 = 0.0
        bleu4_scores.append(b4)

        # 3. METEOR (Sample-level)
        try:
            m_score = meteor_score([ref_tokens], pred_tokens)
        except Exception:
            m_score = 0.0
        meteor_scores.append(m_score)

        # 4. ROUGE-L (Sample-level)
        try:
            r_score = scorer.score(ref, pred)['rougeL'].fmeasure
        except:
            r_score = 0.0
        rouge_l_scores.append(r_score)

        # 5. Prepare data for CIDEr
        # CIDEr expects space-separated strings in specific dict format: {id: [text]}
        # We use the loop index 'i' as the unique ID
        cider_gts[str(i)] = [' '.join(ref_tokens)]
        cider_res[str(i)] = [' '.join(pred_tokens)]

    # === Compute Final Scores ===
    results = {
        "BLEU-4": np.mean(bleu4_scores),
        "METEOR": np.mean(meteor_scores),
        "ROUGE-L": np.mean(rouge_l_scores),
        "CIDEr": 0.0 # Default
    }
    
    # 6. Compute CIDEr (Corpus-level)
    if Cider is not None:
        try:
            print("Calculating CIDEr score (this computes TF-IDF over the corpus)...")
            cider_scorer = Cider()
            # compute_score returns (overall_score, list_of_scores_per_image)
            cider_score, _ = cider_scorer.compute_score(cider_gts, cider_res)
            results["CIDEr"] = cider_score
        except Exception as e:
            print(f"[Error] CIDEr calculation failed: {e}")
    
    return results

# ==========================================
# LLM EVALUATION FUNCTION (Worker - BATCHED)
# ==========================================
def evaluate_batch_llm(batch_samples, batch_index, output_dir, model_name, api_key):
    """Evaluates a chunk of samples in a single API call."""
    output_path = os.path.join(output_dir, f"batch_{batch_index}.json")
    
    if os.path.exists(output_path):
        return

    # 1. Format the batch into a single string
    batch_text = ""
    for sample in batch_samples:
        match_id = sample.get('match_id', 'unknown')
        turn_index = sample.get('turn_index', 'unknown')
        unique_key = f"match_{match_id}_turn_{turn_index}"
        
        batch_text += f"\n--- ID: {unique_key} ---\n"
        batch_text += f"1. METADATA: {sample.get('metadata_gt', '{}')}\n"
        batch_text += f"2. REFERENCE: \"{sample.get('commentary_gt', '')}\"\n"
        batch_text += f"3. PREDICTION: \"{sample.get('prediction', '')}\"\n"

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = USER_PROMPT_TEMPLATE.format(
            num_samples=len(batch_samples),
            batch_data=batch_text
        )
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json" 
            )
        )
        
        response_content = response.text
        if response_content.startswith("```"):
            response_content = response_content.strip("`").replace("json", "").replace("python", "").strip()
            
        eval_result = json.loads(response_content)
        
        # 2. Save the whole batch result
        full_record = {
            "batch_index": batch_index,
            "samples": batch_samples,
            "evaluation": eval_result # Expected format: {"results": [{...}, {...}]}
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_record, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Error processing batch {batch_index}: {e}")

# Helper to chunk data
def chunk_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def evaluate_batch_wrapper(args):
    return evaluate_batch_llm(*args)

# ==========================================
# MAIN
# ==========================================
def main():
    args = parse_args()

    with open(args.pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")

    final_report = {
        "traditional_metrics": {},
        "llm_metrics": {},
        "samples_evaluated": len(data)
    }

    # STEP 1: TRADITIONAL METRICS
    if args.eval_mode in ["all", "traditional_only"]:
        trad_scores = compute_traditional_metrics(data)
        final_report["traditional_metrics"] = trad_scores
        print("\n=== Traditional Metrics ===")
        for k, v in trad_scores.items():
            print(f"{k}: {v:.4f}")

    # STEP 2: LLM EVALUATION
    if args.eval_mode in ["all", "llm_only"]:
        if not args.api_key:
            print("\n[Error] API Key required. Skipping...")
        else:
            os.makedirs(args.output_dir, exist_ok=True)

            # Chunk data into batches
            batches = list(chunk_data(data, args.batch_size))
            print(f"\n[LLM] Evaluating {len(data)} samples in {len(batches)} batches (Size: {args.batch_size})...")
            
            # Prepare arguments for multiprocessing
            task_args = [(batch, i, args.output_dir, args.model, args.api_key) for i, batch in enumerate(batches)]

            with Pool(args.num_tasks) as pool:
                for _ in tqdm(
                    pool.imap_unordered(evaluate_batch_wrapper, task_args), 
                    total=len(batches), 
                    desc="LLM Eval Progress",
                    smoothing=0.1 
                ):
                    pass

            # Merge LLM Results
            print("Merging LLM results...")
            stats = {k: 0 for k in ["accuracy", "coherence", "excitement", "professionalism", "pacing", "total"]}
            count = 0
            
            for file_name in tqdm(os.listdir(args.output_dir)):
                if file_name.endswith(".json"):
                    with open(os.path.join(args.output_dir, file_name), "r", encoding="utf-8") as f:
                        record = json.load(f)
                        results_array = record.get("evaluation", {}).get("results", [])
                        
                        for eval_item in results_array:
                            scores = eval_item.get("scores", {})
                            total = eval_item.get("total_score", 0)
                            if scores:
                                count += 1
                                for k in stats.keys():
                                    if k == "total":
                                        stats[k] += total
                                    else:
                                        stats[k] += scores.get(k, 0)

            if count > 0:
                final_report["llm_metrics"] = {k: v / count for k, v in stats.items()}
                print("\n=== LLM Metrics ===")
                for k, v in final_report["llm_metrics"].items():
                    print(f"{k}: {v:.2f}")
                print(f"Successfully aggregated {count} individual samples.")
            else:
                print("No valid LLM results found.")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\nFinal report saved to {args.output_json}")

if __name__ == "__main__":
    main()
