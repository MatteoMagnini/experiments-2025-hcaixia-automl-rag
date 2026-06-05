#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def query_openrouter(
    api_key: str,
    model_name: str,
    sys_prompt: str,
    user_prompt: str,
    base_url: str = None,
    timeout: int = 30
):
    import urllib.request
    import urllib.error
    import json

    base_url = base_url or "https://openrouter.ai/api/v1"
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "experiments-agent/1.0"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "top_p": 0.95,
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            resp_body = response.read().decode("utf-8")
            try:
                return json.loads(resp_body), status_code
            except Exception:
                return {"error": "Invalid JSON response"}, status_code
    except urllib.error.HTTPError as e:
        status_code = e.code
        try:
            resp_body = e.read().decode("utf-8")
            return json.loads(resp_body), status_code
        except Exception:
            return {"error": e.reason}, status_code
    except Exception as e:
        return {"error": str(e)}, 500

def load_question_to_context_mapping(chunk_file_path: Path) -> dict[str, str]:
    """
    Builds a robust mapping from unique question strings to their corresponding context (Testo).
    Looks across all potential question columns (Domanda_1 to Domanda_5).
    """
    if not chunk_file_path.exists():
        print(f"Error: Chunk file not found at {chunk_file_path}")
        return {}
    
    print(f"Loading questions mapping from {chunk_file_path.name}...")
    try:
        # Load TSV
        df = pd.read_csv(chunk_file_path, sep="\t")
        mapping = {}
        
        # We search across all five question columns to build a complete mapping
        question_cols = ["Domanda_1", "Domanda_2", "Domanda_3", "Domanda_4", "Domanda_5"]
        valid_cols = [col for col in question_cols if col in df.columns]
        
        for _, row in df.iterrows():
            testo = row.get("Testo")
            if pd.isna(testo):
                continue
            testo_str = str(testo).strip()
            
            for col in valid_cols:
                q_val = row.get(col)
                if pd.notna(q_val):
                    q_str = str(q_val).strip()
                    if q_str:
                        mapping[q_str] = testo_str
                        
        print(f"Successfully loaded mapping with {len(mapping)} unique questions.")
        return mapping
    except Exception as e:
        print(f"Error reading question-to-context mapping: {e}")
        return {}


def coerce_content_to_string(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def pregenerate_gold_answers(
    csv_path: Path,
    mapping: dict[str, str],
    provider: str,
    api_key: str,
    model_name: str,
    base_url: str | None,
    batch_size: int,
    overwrite_existing: bool,
) -> None:
    """
    Reads a question CSV file, maps questions to their top chunk context,
    queries the selected LLM provider in parallel batches to generate gold standard answers,
    and updates the CSV file in-place (saving a backup first).
    """
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    print(f"\nProcessing {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    if "question" not in df.columns:
        print(f"Error: 'question' column not found in {csv_path.name}")
        return

    # Check if we already have the gold standard column
    if "gold_standard_answer" in df.columns and not overwrite_existing:
        # Filter to generate only missing ones
        missing_mask = df["gold_standard_answer"].isna() | (df["gold_standard_answer"].str.strip() == "")
        if not missing_mask.any():
            print(f"All questions in {csv_path.name} already have a gold standard answer. Use --overwrite to regenerate.")
            return
        print(f"Found {missing_mask.sum()} missing gold standard answers to generate out of {len(df)} total.")
        indices_to_generate = df[missing_mask].index.tolist()
    else:
        print(f"Generating gold standard answers for all {len(df)} questions in {csv_path.name}.")
        df["gold_standard_answer"] = ""
        indices_to_generate = list(range(len(df)))

    # Prepare prompts
    prompts_to_run = []
    metadata = [] # stores (index, question, context)
    
    for idx in indices_to_generate:
        question = str(df.loc[idx, "question"]).strip()
        context = mapping.get(question)
        if not context:
            print(f"Warning: No context found for question: '{question[:60]}...'")
            context = "Nessun contesto di riferimento trovato."
        
        # Medical expert prompt using ONLY the top chunk context
        prompt = (
            "Sei un assistente medico esperto. Rispondi alla domanda dell'utente utilizzando solo il testo di riferimento fornito (il gold standard).\n\n"
            f"Testo di riferimento (Gold Standard):\n{context}\n\n"
            f"Domanda:\n{question}\n\n"
            "Risposta:"
        )
        prompts_to_run.append(prompt)
        metadata.append((idx, question))

    if not prompts_to_run:
        print("No prompts to run.")
        return

    generated_answers = {}

    if provider == "openrouter":
        import time
        sys_prompt = "Sei un assistente medico esperto."
        
        def process_row(idx, prompt):
            attempt = 0
            max_retries = 3
            while attempt < max_retries:
                print(f"[Gold Standards Script] Row {idx} sending request to OpenRouter (attempt {attempt+1}/{max_retries})...", flush=True)
                resp_json, status_code = query_openrouter(
                    api_key, model_name, sys_prompt, prompt, base_url, timeout=30
                )
                if status_code == 200:
                    choices = resp_json.get("choices", [{}])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        print(f"[Gold Standards Script] Row {idx} successfully generated response.", flush=True)
                        return content.strip()
                else:
                    err_msg = resp_json.get("error", resp_json)
                    print(f"[Gold Standards Script] Row {idx} attempt {attempt+1} failed with status {status_code}: {err_msg}", flush=True)
                attempt += 1
                time.sleep(2 * attempt)
            print(f"[Gold Standards Script] Row {idx} failed completely after {max_retries} attempts.", flush=True)
            return "Non so."

        max_workers = int(os.environ.get("CONCURRENT_WORKERS", "1"))
        if max_workers <= 1:
            print(f"Generating gold answers with OpenRouter model ({model_name}) sequentially...", flush=True)
            for idx, p in tqdm(metadata, desc=f"Generating {csv_path.name}"):
                prompt_idx = indices_to_generate.index(idx)
                prompt = prompts_to_run[prompt_idx]
                generated_answers[idx] = process_row(idx, prompt)
        else:
            print(f"Generating gold answers with OpenRouter model ({model_name}) in parallel (max_workers={max_workers})...", flush=True)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_row, idx, prompts_to_run[i]): idx for i, (idx, _) in enumerate(metadata)}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {csv_path.name}"):
                    idx = futures[future]
                    try:
                        generated_answers[idx] = future.result()
                    except Exception:
                        generated_answers[idx] = "Non so."

    else:
        # Initialize the corresponding LLM Client
        from langchain_google_genai import ChatGoogleGenerativeAI
        try:
            print(f"Initializing Gemini model '{model_name}' via Google AI Studio...")
            lc_llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=0.0,
            )
        except Exception as e:
            print(f"Error initializing ChatGoogleGenerativeAI: {e}")
            sys.exit(1)

        print(f"Querying model ({model_name}) in batches of {batch_size}...")
        
        # Process in batches to display progress and handle errors gracefully
        for i in tqdm(range(0, len(prompts_to_run), batch_size), desc=f"Generating {csv_path.name}"):
            batch_prompts = prompts_to_run[i:i + batch_size]
            batch_meta = metadata[i:i + batch_size]
            
            try:
                # Langchain parallel batch invocation
                responses = lc_llm.batch(batch_prompts)
                for response, (idx, q) in zip(responses, batch_meta):
                    raw_content = response.content if hasattr(response, "content") else response
                    ans_text = coerce_content_to_string(raw_content)
                    generated_answers[idx] = ans_text.strip()
            except Exception as batch_err:
                print(f"\nError in parallel batch processing: {batch_err}. Falling back to sequential for this batch.")
                # Fallback to sequential execution for just this batch
                for prompt, (idx, q) in zip(batch_prompts, batch_meta):
                    try:
                        response = lc_llm.invoke(prompt)
                        raw_content = response.content if hasattr(response, "content") else response
                        ans_text = coerce_content_to_string(raw_content)
                        generated_answers[idx] = ans_text.strip()
                    except Exception as seq_err:
                        print(f"Error generating answer for question '{q[:40]}...': {seq_err}")
                        generated_answers[idx] = "Non so."

    # Assign generated answers back to the DataFrame
    for idx, ans in generated_answers.items():
        df.loc[idx, "gold_standard_answer"] = ans

    # Create backup before overwriting
    backup_path = csv_path.with_name(f"{csv_path.stem}_backup{csv_path.suffix}")
    print(f"Creating backup of original file at {backup_path.name}")
    shutil_df = pd.read_csv(csv_path)
    shutil_df.to_csv(backup_path, index=False)

    # Save updated file
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved updated data with 'gold_standard_answer' to {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Pregenerate gold standard answers using Gemini (Google AI Studio) or OpenRouter with top chunk context.")
    parser.add_argument("--provider", type=str, choices=["google", "openrouter"], default="google", help="LLM provider to use (default: google).")
    parser.add_argument("--model", type=str, default=None, help="Model name to use (default: gemini-1.5-flash for google, google/gemma-3-27b-it for openrouter).")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for concurrent API calls.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing gold_standard_answer values if present.")
    args = parser.parse_args()

    # Determine paths
    data_dir = Path(__file__).resolve().parent
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    chunk_tsv = data_dir / "chunk_questions_dataset.tsv"

    # Load provider and API keys
    provider = args.provider
    api_key = None
    model_name = args.model
    base_url = None

    if provider == "google":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not model_name:
            model_name = "gemini-3.5-flash"
        if not api_key:
            # Fallback check if user forgot to set google key but has openrouter
            if os.environ.get("OPENROUTER_API_KEY"):
                print("Warning: GEMINI_API_KEY/GOOGLE_API_KEY not found, but OPENROUTER_API_KEY is available. Switching provider to openrouter.")
                provider = "openrouter"
            else:
                print("Error: Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable was found.")
                print("Please set your Google AI Studio API key in your environment or a .env file.")
                sys.exit(1)

    # If provider is (or became) openrouter
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not model_name:
            model_name = "google/gemma-3-27b-it"
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            print("Error: OPENROUTER_API_KEY environment variable not found.")
            print("Please set your OpenRouter API key in your environment or a .env file.")
            sys.exit(1)

    print("=" * 60)
    print("Gold Standard Answer Generator")
    print(f"Provider: {provider.upper()}")
    print(f"Model: {model_name}")
    if base_url:
        print(f"Base URL: {base_url}")
    print("=" * 60)

    # Load mapping
    mapping = load_question_to_context_mapping(chunk_tsv)
    if not mapping:
        print("Error: Could not build question-to-context mapping.")
        sys.exit(1)

    # Process train and test
    pregenerate_gold_answers(
        csv_path=train_csv,
        mapping=mapping,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        batch_size=args.batch_size,
        overwrite_existing=args.overwrite,
    )

    pregenerate_gold_answers(
        csv_path=test_csv,
        mapping=mapping,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        batch_size=args.batch_size,
        overwrite_existing=args.overwrite,
    )

    print("\nGeneration task complete!")


if __name__ == "__main__":
    main()
