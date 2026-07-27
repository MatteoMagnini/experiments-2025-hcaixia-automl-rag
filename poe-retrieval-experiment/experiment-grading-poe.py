"""
Grading runner (FILES PARALLEL + 2 MODELS PARALLEL)

✔ multiple files in parallel
✔ 2 models in parallel per file
✔ max 10 docs per file
✔ no DeepSeek
✔ retry logic
✔ safe doc cleaning
✔ checkpointing (dataset-aware, migrates old format)
✔ thread-safe done set
✔ deadlock-free checkpoint
"""

import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from PoE3.queryPoE import AskPoE
from grading_utils import docs_to_grading_input, merge_labels_into_docs


# ---------------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

TEST_MODE = args.test
TEST_N_QUERIES = 3

MAX_DOCS = 10
MAX_RETRIES = 3
MAX_PARALLEL_MODELS = 2
MAX_FILE_WORKERS = 3


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs"
RETRIEVAL_DIR = os.path.join(OUTPUT_DIR, "retrieval")

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "results_test.jsonl" if TEST_MODE else "results.jsonl"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "checkpoint_test.json" if TEST_MODE else "checkpoint.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]


# ---------------------------------------------------------------------------
# BOT CONFIGS (NO DEEPSEEK)
# ---------------------------------------------------------------------------

PREFIXES = ["3EAs"]
PERSONAS = ["User Design Persona"]

LLMS = [
    "llama-3.3-70b-instruct",
    "mistral-small-24B-instruct-2501",
]

BOTNAME_CONFIGS = [
    {
        "botname": f"{pfx}/{persona}-1.2-0.9-1-{llm}",
        "n_experts": pfx,
        "framework": persona,
        "llm": llm,
    }
    for pfx in PREFIXES
    for persona in PERSONAS
    for llm in LLMS
]


# ---------------------------------------------------------------------------
# LOCKS
# ---------------------------------------------------------------------------

_done_lock    = __import__("threading").Lock()
_results_lock = __import__("threading").Lock()


# ---------------------------------------------------------------------------
# CHECKPOINT (dataset-aware, migrates old format)
# ---------------------------------------------------------------------------

def load_checkpoint():
    done = set()

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            for entry in json.load(f):
                done.add(entry)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    full_id = f"{obj['dataset']}||{obj['file_name']}||{obj['botname']}"
                    done.add(full_id)
                except Exception:
                    continue

    print(f"Checkpoint loaded: {len(done)} completed tasks")
    return done


def save_checkpoint(done_snapshot):
    """Write checkpoint from a snapshot — always called outside _done_lock."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(done_snapshot), f)


def append_result(obj):
    with _results_lock:
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    print(f"[saved] {obj['task_id']}")


# ---------------------------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------------------------

def load_all_retrievals():
    records = []

    for fname in sorted(os.listdir(RETRIEVAL_DIR)):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(RETRIEVAL_DIR, fname)

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        entries = list(payload["results"].items())

        if TEST_MODE:
            entries = entries[:TEST_N_QUERIES]

        for file_name, content in entries:
            records.append({
                "file_name": file_name,
                "query":     content["query"],
                "docs":      content["docs"],
                "dataset":   fname.replace(".json", ""),
            })

    return records


# ---------------------------------------------------------------------------
# SAFE DOC CLEANING
# ---------------------------------------------------------------------------

def clean_docs(docs):
    cleaned = []
    for d in docs:
        text = d.get("text") or ""
        d["text"] = str(text)
        cleaned.append(d)
    return cleaned


# ---------------------------------------------------------------------------
# RETRY WRAPPER
# ---------------------------------------------------------------------------

def safe_ask_poe(botname, payload):
    for attempt in range(MAX_RETRIES):
        try:
            return AskPoE(
                botname=botname,
                query=json.dumps(payload),
                apikey=OPENROUTER_KEY,
            )
        except Exception as e:
            print(f"[retry {attempt}] {botname}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# SINGLE MODEL EXECUTION
# ---------------------------------------------------------------------------

def run_model(cfg, docs, grading_input, query, file_name, dataset, done):

    task_id = f"{dataset}||{file_name}||{cfg['botname']}"

    with _done_lock:
        if task_id in done:
            return 0

    try:
        poe_result = safe_ask_poe(cfg["botname"], grading_input)

        if poe_result is None:
            raise RuntimeError("AskPoE failed after retries")

        fdm = poe_result.get("final-decision-maker", {}) or {}
        raw = fdm.get("final-decision-maker-answer") or fdm.get("final_answer")

        if raw is None:
            print(f"    [debug] poe_result keys: {list(poe_result.keys())}")
            print(f"    [debug] fdm keys: {list(fdm.keys())}")
            print(f"    [debug] fdm content: {json.dumps(fdm, ensure_ascii=False)[:500]}")
            raise ValueError(f"Missing final_answer. fdm keys: {list(fdm.keys())}")

        parsed = json.loads(raw)
        labels = parsed.get("labels")

        if not isinstance(labels, list):
            raise ValueError(f"labels is not a list: {type(labels)}")

        graded_docs = merge_labels_into_docs(docs, labels)

        result = {
            "task_id":     task_id,
            "dataset":     dataset,
            "file_name":   file_name,
            "query":       query,
            "botname":     cfg["botname"],
            "graded_docs": graded_docs,
        }

        append_result(result)

        # Snapshot done under lock, then write checkpoint outside lock
        with _done_lock:
            done.add(task_id)
            snapshot = set(done)

        save_checkpoint(snapshot)

        return 1

    except Exception as e:
        print(f"✗ {cfg['botname']}: {e}")
        return 0


# ---------------------------------------------------------------------------
# GRADE ONE FILE
# ---------------------------------------------------------------------------

def grade_file(retrieval, done):

    file_name = retrieval["file_name"]
    query     = retrieval["query"]
    dataset   = retrieval["dataset"]
    docs      = retrieval["docs"]

    if not isinstance(docs, list):
        print(f"[BAD DOC FORMAT] {file_name}")
        return 0

    docs = clean_docs(docs[:MAX_DOCS])

    with _done_lock:
        pending_cfgs = [
            cfg for cfg in BOTNAME_CONFIGS
            if f"{dataset}||{file_name}||{cfg['botname']}" not in done
        ]

    if not pending_cfgs:
        print(f"[skip] {dataset}||{file_name}")
        return 0

    print(f"[start] {dataset}||{file_name} — {len(pending_cfgs)} models")

    grading_input = docs_to_grading_input(docs, query=query)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_MODELS) as executor:
        futures = [
            executor.submit(
                run_model, cfg, docs, grading_input, query, file_name, dataset, done
            )
            for cfg in pending_cfgs
        ]
        for f in as_completed(futures):
            completed += f.result()

    print(f"[done] {dataset}||{file_name} — {completed} new tasks")
    return completed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def grade_all():
    retrievals = load_all_retrievals()

    print("=" * 60)
    print("MODE: TEST" if TEST_MODE else "MODE: FULL")
    print("=" * 60)
    print(f"Retrieval records: {len(retrievals)}")
    print(f"Botname configs:   {len(BOTNAME_CONFIGS)}")
    print(f"Total tasks (max): {len(retrievals) * len(BOTNAME_CONFIGS)}")
    print(f"Results file:      {os.path.abspath(RESULTS_FILE)}")
    print(f"Checkpoint file:   {os.path.abspath(CHECKPOINT_FILE)}\n")

    done = load_checkpoint()

    total = 0

    with ThreadPoolExecutor(max_workers=MAX_FILE_WORKERS) as executor:
        futures = {
            executor.submit(grade_file, r, done): r
            for r in retrievals
        }
        for future in as_completed(futures):
            r = futures[future]
            try:
                total += future.result()
            except Exception as e:
                print(f"[FILE ERROR] {r['dataset']}||{r['file_name']}: {e}")

    print(f"\nDone. Total new tasks: {total}")
    print(f"Results → {os.path.abspath(RESULTS_FILE)}")


if __name__ == "__main__":
    grade_all()