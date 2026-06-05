"""Answer generation via OpenRouter.

Given a batch of (question, retrieved-context) pairs, produce the system answers
that the generative metrics in `metrics.py` are then scored against. All network
access to the LLM lives here.
"""
import os


def query_openrouter(
    api_key: str,
    model_name: str,
    sys_prompt: str,
    user_prompt: str,
    base_url: str = None,
    timeout: int = None,
):
    if timeout is None:
        timeout = int(os.environ.get("OPENROUTER_TIMEOUT", "10"))
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


def generate_answers_for_eval(
    user_inputs: list[str],
    retrieved_contexts_list: list[list[str]],
    gen_model: str = None,
) -> list[str]:
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in environment.")
        return ["Non so." for _ in user_inputs]
    if not gen_model:
        gen_model = "google/gemma-3-27b-it"
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    from concurrent.futures import ThreadPoolExecutor
    import time

    sys_prompt = "Sei un assistente medico esperto."
    prompts = []
    for question, contexts in zip(user_inputs, retrieved_contexts_list):
        context_str = "\n\n".join(contexts)
        user_prompt = (
            "Sei un assistente medico esperto. Rispondi alla domanda dell'utente utilizzando solo il contesto fornito. "
            "Se il contesto non contiene le informazioni necessarie, rispondi 'Non so'.\n\n"
            f"Contesto:\n{context_str}\n\n"
            f"Domanda:\n{question}\n\n"
            "Risposta (compatta, al massimo 100 parole):"
        )
        prompts.append(user_prompt)

    answers = [None] * len(user_inputs)

    def process_row(idx, user_prompt):
        attempt = 0
        max_retries = 3
        while attempt < max_retries:
            print(f"[Eval Answers] Row {idx} sending request to OpenRouter (attempt {attempt+1}/{max_retries})...", flush=True)
            resp_json, status_code = query_openrouter(
                openrouter_api_key, gen_model, sys_prompt, user_prompt, base_url
            )
            if status_code == 200:
                choices = resp_json.get("choices", [{}])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    print(f"[Eval Answers] Row {idx} successfully generated response.", flush=True)
                    return content.strip()
            else:
                err_msg = resp_json.get("error", resp_json)
                print(f"[Eval Answers] Row {idx} attempt {attempt+1} failed with status {status_code}: {err_msg}", flush=True)
            attempt += 1
            time.sleep(2 * attempt)
        print(f"[Eval Answers] Row {idx} failed completely after {max_retries} attempts.", flush=True)
        return "Non so."

    max_workers = int(os.environ.get("CONCURRENT_WORKERS", "1"))
    if max_workers <= 1:
        print("Generating answers with OpenRouter sequentially..." + str(gen_model), flush=True)
        for idx, p in enumerate(prompts):
            answers[idx] = process_row(idx, p)
            print(f"[Eval Answers] Progress: {idx+1}/{len(prompts)} answers generated.", flush=True)
    else:
        print(f"Generating answers with OpenRouter in parallel (max_workers={max_workers})...", flush=True)
        from concurrent.futures import as_completed, TimeoutError as FuturesTimeoutError
        # Overall budget so a stuck worker can never hang the whole trial.
        per_request_budget = int(os.environ.get("OPENROUTER_TIMEOUT", "60")) * 6
        total_timeout = float(os.environ.get("EVAL_TOTAL_TIMEOUT", str(per_request_budget * len(prompts))))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, idx, p): idx for idx, p in enumerate(prompts)}
            completed = 0
            try:
                for future in as_completed(futures, timeout=total_timeout):
                    idx = futures[future]
                    try:
                        answers[idx] = future.result()
                    except Exception as e:
                        print(f"[Eval Answers] Exception during processing of row {idx}: {e}", flush=True)
                        answers[idx] = "Non so."
                    completed += 1
                    print(f"[Eval Answers] Progress: {completed}/{len(prompts)} answers generated.", flush=True)
            except FuturesTimeoutError:
                print(f"[Eval Answers] Overall timeout ({total_timeout}s) reached; cancelling pending requests.", flush=True)
                for future, idx in futures.items():
                    if not future.done():
                        future.cancel()
                        if answers[idx] is None:
                            answers[idx] = "Non so."
    # Fill any answer left unset (e.g. cancelled) with a safe default.
    answers = [a if a is not None else "Non so." for a in answers]
    # answer sample:
    print(f"Sample generated answer: {answers[0] if answers else 'No answers generated'}", flush=True)
    return answers
