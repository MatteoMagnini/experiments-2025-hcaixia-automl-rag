import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from Experiment import QueriesToExpertAgents, QueriesToMakeFinalDecision

import json

#from utilities import IsInTopic
from dotenv import load_dotenv

from os import getenv


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env") #solo per test


import time
from Experiment import QueriesToExpertAgents, QueriesToMakeFinalDecision

import json

#from utilities import IsInTopic
from dotenv import load_dotenv

from os import getenv
from pathlib import Path


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env") #solo per test

script_dir = Path(__file__).resolve().parent


def AskPoE(botname: str,
        query: str,
        apikey: str):
    """
    return a dict with experts and FDM answers, and a boolean
    """
    config_file = script_dir / botname / "args_dict.json"

    with open(config_file, 'r') as json_file:
        args_dict = json.load(json_file)
    #  ask to experts
    answers_experts = QueriesToExpertAgents(args_dict, botname, apikey, query)
    print("Experts answers collected going to FDM")
    #  Ask FDM
    final_answer = QueriesToMakeFinalDecision(args_dict, botname, apikey, answers_experts, query)

    answer = {"experts": answers_experts,
            "final-decision-maker": final_answer,
            "query": query,
            }

    return answer #IsInTopic(final_answer)




if __name__ == "__main__":
    s = time.time()
    args_dict = "chatFAQ_ss/args_dict.json"
    query = "cosa sono i DPI?"
    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment or .env file")

    print(f"Loaded API key: {api_key[:4]}...") 
    _, intopic = AskPoE(botname='chatFAQ_ss',
                                                query=query, apikey=api_key)
    print(f"query: {query} is in-topic? {intopic}")
    print(f"total time: {time.time() - s}")
