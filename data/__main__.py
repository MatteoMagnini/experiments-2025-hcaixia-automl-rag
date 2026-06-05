from random import seed
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import fire
from dotenv import load_dotenv
from tqdm import tqdm

from data import PATH as DATASET_PATH, CHUNKS_QUESTIONS_FILE, TEST_FILE_NAME, TRAINING_FILE_NAME
from documents import PATH as DOCUMENT_PATH, LOOKUP_FILE_NAME

# Load environment variables from .env file
load_dotenv()

CHUNK_FILE = DATASET_PATH / CHUNKS_QUESTIONS_FILE
TRAIN_FILE = DATASET_PATH / TRAINING_FILE_NAME
TEST_FILE = DATASET_PATH / TEST_FILE_NAME
LOOKUP_FILE = DOCUMENT_PATH / LOOKUP_FILE_NAME


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


def generate_gold_answers(
    df: pd.DataFrame,
    provider: str,
    model: str,
    api_key: str,
    base_url: str = None,
    lc_llm = None,
    batch_size: int = 15
) -> list[str]:
    prompts = []
    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        context = str(row["context"]).strip()
        
        # Medical expert prompt using the exact context from the dataset
        prompt = (
            "Sei un assistente medico esperto. Rispondi alla domanda dell'utente utilizzando solo il testo di riferimento fornito (il gold standard).\n\n"
            f"Testo di riferimento (Gold Standard):\n{context}\n\n"
            f"Domanda:\n{question}\n\n"
            "Risposta:"
        )
        prompts.append(prompt)
        
    answers = [None] * len(df)
    
    if provider == "openrouter":
        import time
        sys_prompt = "Sei un assistente medico esperto."
        
        def process_row(idx, prompt):
            attempt = 0
            max_retries = 3
            while attempt < max_retries:
                print(f"[Gold Answers] Row {idx} sending request to OpenRouter (attempt {attempt+1}/{max_retries})...", flush=True)
                resp_json, status_code = query_openrouter(
                    api_key, model, sys_prompt, prompt, base_url, timeout=30
                )
                if status_code == 200:
                    choices = resp_json.get("choices", [{}])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        print(f"[Gold Answers] Row {idx} successfully generated response.", flush=True)
                        return content.strip()
                else:
                    err_msg = resp_json.get("error", resp_json)
                    print(f"[Gold Answers] Row {idx} attempt {attempt+1} failed with status {status_code}: {err_msg}", flush=True)
                attempt += 1
                time.sleep(2 * attempt)
            print(f"[Gold Answers] Row {idx} failed completely after {max_retries} attempts.", flush=True)
            return "Non so."

        max_workers = int(os.environ.get("CONCURRENT_WORKERS", "1"))
        if max_workers <= 1:
            print("Generating gold answers with OpenRouter sequentially...", flush=True)
            for idx, p in enumerate(tqdm(prompts, desc="Gold Standard Answer Generation")):
                answers[idx] = process_row(idx, p)
        else:
            print(f"Generating gold answers with OpenRouter in parallel (max_workers={max_workers})...", flush=True)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_row, idx, p): idx for idx, p in enumerate(prompts)}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Gold Standard Answer Generation"):
                    idx = futures[future]
                    try:
                        answers[idx] = future.result()
                    except Exception:
                        answers[idx] = "Non so."
        return answers

    else:
        print(f"Generating gold answers in batches of {batch_size}...")
        for i in tqdm(range(0, len(prompts), batch_size), desc="Gold Standard Answer Generation"):
            batch_prompts = prompts[i:i+batch_size]
            try:
                # Langchain parallel batch invocation
                responses = lc_llm.batch(batch_prompts)
                for j, response in enumerate(responses):
                    raw_content = response.content if hasattr(response, "content") else response
                    ans_text = coerce_content_to_string(raw_content).strip()
                    answers[i + j] = ans_text
            except Exception as e:
                print(f"\nBatch generation failed: {e}. Falling back to sequential for this batch.")
                for j, prompt in enumerate(batch_prompts):
                    try:
                        response = lc_llm.invoke(prompt)
                        raw_content = response.content if hasattr(response, "content") else response
                        ans_text = coerce_content_to_string(raw_content).strip()
                        answers[i + j] = ans_text
                    except Exception as seq_e:
                        print(f"Error generating answer: {seq_e}")
                        answers[i + j] = "Non so."
        return answers


def main(provider: str = "google", model: str = None, batch_size: int = 15):
    """
    Performs train/test split proportional stratified sampling from chunk questions dataset,
    initializes the selected LLM provider, generates gold standard answers using the dataset context,
    and saves train.csv and test.csv with the 'gold_standard_answer' column.
    """
    # 1. Setup seed and load datasets
    seed(0)
    np.random.seed(0)
    
    print("Loading datasets...")
    chunk_df = pd.read_csv(CHUNK_FILE, sep="\t")
    lookup_df = pd.read_csv(LOOKUP_FILE)
    
    # 2. Extract columns and keep context ('Testo')
    df = chunk_df[["Domanda_1", "File", "Testo"]].copy()
    df.rename(columns={"Domanda_1": "question", "File": "file_name", "Testo": "context"}, inplace=True)
    df = df.merge(lookup_df, left_on="file_name", right_on="file_name")
    
    # 3. Stratified proportional sampling of exactly 100 elements for train set
    print("Creating stratified proportional train set (100 rows)...")
    train_df = df.groupby("file_name", group_keys=False).apply(
        lambda x: x.sample(n=max(1, round(len(x) * 100 / len(df))), random_state=0)
    )
    if len(train_df) > 100:
        train_df = train_df.sample(n=100, random_state=0)
    elif len(train_df) < 100:
        remaining = df.drop(train_df.index)
        extra = remaining.sample(n=100 - len(train_df), random_state=0)
        train_df = pd.concat([train_df, extra])

    # 4. Sample 200 elements for test set
    print("Creating test set (200 rows)...")
    test_df = df.drop(train_df.index).sample(n=200, random_state=0)

    # 5. Initialize the selected LLM Client
    lc_llm = None
    api_key = None
    base_url = None
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not model:
            model = "gemini-1.5-flash"
        if not api_key:
            print("Error: Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable was found.")
            print("Please set your Google AI Studio API key in your environment or a .env file.")
            sys.exit(1)
        print(f"Initializing Gemini model '{model}'...")
        lc_llm = ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=0.0)
    else:  # openrouter
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not model:
            model = "google/gemma-3-27b-it"
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            print("Error: OPENROUTER_API_KEY environment variable not found.")
            print("Please set your OpenRouter API key in your environment or a .env file.")
            sys.exit(1)
        print(f"Initializing OpenRouter model '{model}' (bypassing Langchain with direct HTTP)...")

    # 6. Generate gold standard answers GIVEN the context from the dataset
    print("\n--- Generating Gold Standard Answers for Train Set ---")
    train_df["gold_standard_answer"] = generate_gold_answers(
        train_df,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        lc_llm=lc_llm,
        batch_size=batch_size
    )
    
    print("\n--- Generating Gold Standard Answers for Test Set ---")
    test_df["gold_standard_answer"] = generate_gold_answers(
        test_df,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        lc_llm=lc_llm,
        batch_size=batch_size
    )

    # 7. Keep only expected columns and save files
    print("\nSaving final train and test CSVs...")
    train_to_save = train_df[["question", "id", "gold_standard_answer"]].copy()
    test_to_save = test_df[["question", "file_name", "id", "gold_standard_answer"]].copy()
    
    # Save files
    train_to_save.to_csv(TRAIN_FILE, index=False)
    test_to_save.to_csv(TEST_FILE, index=False)
    print("Train and test datasets successfully created, generated, and saved!")


if __name__ == "__main__":
    fire.Fire(main)