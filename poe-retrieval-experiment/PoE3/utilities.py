import json
import os


def is_baseline(args_dict: dict) -> bool:
    if 'baseline' in args_dict and args_dict['baseline']:
        return True
    return False


def is_with_personality(args_dict: dict):
    if 'no-personality' in args_dict and args_dict['no-personality']:
        return False
    if 'no-description' in args_dict and args_dict['no-description']:
        return False
    return True

def is_without_description(args_dict: dict):
    if args_dict['description_framework']=="no-description":
        return True
    return False

def is_openrouter(args_dict: dict):
    if "openrouterfile" in args_dict:
        return True
    return False

def check_json_integrity(config_file: str):
    with open(config_file, 'r') as json_file:
        args_dict = json.load(json_file)

    required_fields = {
        "name": str,
        "task": str,
        "context": str,
        "description_framework": str,
        "model": str,
        "output_dir": str,
        "input": str,
        "temperature": (float, int),
        "nucleus": (float, int),
        "alternatives": int,
        "resume": int,
        "cache_dir": str,
        "max_experts_number": int,
        "baseline": int,
        # "token": str
    }

    for field, expected_type in required_fields.items():
        if field not in args_dict:
            print(f"[ERROR] Missing field: {field}")
            return False

        value = args_dict[field]
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                print(f"[ERROR] Field '{field}' must be {expected_type}, got {type(value)}")
                return False
        else:
            if not isinstance(value, expected_type):
                print(f"[ERROR] Field '{field}' must be {expected_type}, got {type(value)}")
                return False

        if value in ("", None):
            print(f"[ERROR] Field '{field}' is empty or null")
            return False

    # Specific range checks
    if not (0 <= args_dict["temperature"] <= 2):
        print("[ERROR] 'temperature' must be between 0 and 2")
        return False
    if not (0 <= args_dict["nucleus"] <= 1):
        print("[ERROR] 'nucleus' must be between 0 and 1")
        return False

    # Path existence checks
    if not os.path.isfile(args_dict["input"]):
        print(f"[ERROR] Input file does not exist: {args_dict['input']}")
        return False
    if not os.path.isdir(args_dict["output_dir"]):
        print(f"[ERROR] Output directory does not exist: {args_dict['output_dir']}")
        return False

    if not "openrouterfile" in args_dict:
        if not os.path.isdir(args_dict["cache_dir"]):
            print(f"[ERROR] Cache directory does not exist: {args_dict['cache_dir']}")
            return False
    else:
        if not os.path.isdir(args_dict["openrouterfile"]):
            print(f"[ERROR] Cache directory does not exist: {args_dict['cache_dir']}")
            return False


    print("[INFO] Configuration passed all integrity checks.")
    return True



# print("get queries indexes must be revised!")
# def get_queries_last_index_for_baseline(queries_answers):
#     for n_query, query in enumerate(queries_answers):
#         if 'baseline-answer' not in queries_answers[query]:
#             return n_query+1
#     return 0

# def get_queries_last_index_for_experts(queries_answers):
#     #  get the query number of the last inserted item
#     for n_query, query in enumerate(queries_answers):
#         if 'expert-answer' not in queries_answers[query]:
#             return n_query+1
#     return 0
#
# def get_queries_last_index_for_an_expert_agent(queries_answers, agent_number):
#     #  get the query number of the last inserted item
#     for n_query, query in enumerate(queries_answers):
#         if 'expert-answers' not in query:
#             return n_query+1
#         for agent_answer in query['expert-answers']:
#             if agent_answer['expert-id']==agent_number:
#                 if agent_answer['expert-answer'] is None:
#                     return n_query+1
#             else:
#                 return 0
#     return 0

# def get_queries_last_index_for_final_decisor_maker(queries_answers):
#     n_query = 0
#     for n_query, query in enumerate(queries_answers):
#         if 'final-decisor-maker-answer' not in queries_answers[query]:
#             return n_query+1
#     return 0

def already_asked_to_query_baseline(queries_answers, n_query):
    try:
        if 'baseline-answer' in queries_answers[str(n_query)]:
            return True
        else:
            return False
    except KeyError:
        return False

def already_asked_to_query_final_decisor_maker(queries_answers, n_query):
    # n_query = 0
    # for n_query, query in enumerate(queries_answers):
    if 'final-decisor-maker-answer' not in queries_answers[n_query]:
        return False
    return True
    # return 0

def already_asked_to_query_expert_agents(queries_answers: dict,
                                         n_query: int,
                                         expert_agents):

    try:
        if 'experts-answers' not in queries_answers[str(n_query)]:
            return False
    except KeyError:
        return False

    #  if here, there are answers.
    #  check that all experts answered.
    experts_answers = queries_answers[str(n_query)]['experts-answers']
    if len(experts_answers) ==  len(expert_agents):
        return True
    else:
        return False



def already_asked_to_query_final_decision_maker(queries_answers, n_query):
    try:
        if "final-decision-maker-answer" not in queries_answers[str(n_query)]:
            return False
    except KeyError:
        return False
    return True