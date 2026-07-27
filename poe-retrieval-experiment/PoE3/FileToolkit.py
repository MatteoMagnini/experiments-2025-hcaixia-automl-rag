import codecs
import os
import random
import json
from pathlib import Path
from typing import List, Dict, Union

from prompts.final_decision_maker import FINAL_DECISOR_NO_DESCRIPTION_description_of_the_FDM
from utilities import is_openrouter, is_without_description


# from PoE.utilities import  get_queries_last_index_for_experts, \
#     get_queries_last_index_for_final_decisor_maker, get_queries_last_index_for_baseline

script_dir = Path(__file__).parent

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Read a JSONL (JSON Lines) file and return a list of dictionaries.

    Args:
        file_path: Path to the .jsonl file

    Returns:
        List of parsed JSON objects (dict)
    """
    file_path = Path(file_path)
    data = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {e}"
                )

    return data

def check_output_dir(args_dict):
    """
        Check the ouput directory exist, otherwise create it
    """
    os.makedirs(args_dict['output_dir'], exist_ok=True)


def check_input_file(args_dict):
    """
        Check input file exist, otherwise throw a FileNotFoundError
    """
    if not os.path.isfile(args_dict['input']):
        raise FileNotFoundError(f"{args_dict['input']} is not a file")


def get_outputdir(args_dict):
    output_dir = list(Path(args_dict['output_dir']).parts)

    if 'no-personality' in args_dict and args_dict['no-personality']:
        args_dict['description_framework'] = 'no-personality'
    if 'no-description' in args_dict and args_dict['no-description']:
        args_dict['description_framework'] = 'no-description'

    experiment_name = "-".join(
        [str(args_dict[x]).strip() for x in ['description_framework', 'temperature', 'nucleus', 'alternatives']])
    experiment_name += f"-{args_dict['model'].split('/')[-1]}"

    output_dir.append(experiment_name)

    outputdir = Path().joinpath(*output_dir).absolute().__str__()

    return outputdir


def update_args_dict(args_dict: dict):
    """
        Add filenames to args_dict
    """

    args_dict['output_dir'] = get_outputdir(args_dict)

    print(f"output_dir: {args_dict['output_dir']}")

    #  psychologist-creator-filename
    args_dict['psychologist-filename'] = Path().joinpath(args_dict['output_dir'],
                                                         "psychologist.json").absolute().__str__()
    #  project-manager-creator-filename
    args_dict['project-manager-filename'] = Path().joinpath(args_dict['output_dir'],
                                                            "project-manager.json").absolute().__str__()

    #  experts-list-filename
    args_dict['experts-list-filename'] = Path().joinpath(args_dict['output_dir'],
                                                         "experts-list.json").absolute().__str__()

    #  experts-list-filename
    args_dict['experts-filename'] = Path().joinpath(args_dict['output_dir'],
                                                    "experts.json").absolute().__str__()

    #  final-decisor-filename
    args_dict['final-decision-maker-filename'] = Path().joinpath(args_dict['output_dir'],
                                                                 "final_decisor.json").absolute().__str__()

    #  update queries_answers_filename
    args_dict['queries_answers_filename'] = Path().joinpath(args_dict['output_dir'],
                                                            "queries_answers.json").absolute().__str__()

    # print(f"args_dict: {args_dict}")


def save_args_dict(args_dict):
    """
        Save the args_dict into a JSON

        Exclude the model and the tokenizer
    """
    #  filter out model and tokenizer from args_dict
    if not is_openrouter(args_dict):
        args_dict = {k: v for k, v in args_dict.items() if k not in ['model', 'tokenizer']}

    #  save args_dict
    with open(f"{args_dict['output_dir']}/args_dict.json", 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"args_dict saved to {args_dict['output_dir']}/args_dict.json")


def read_args_dict(args_dict_filename):
    """
        load args_dict from disk
    """
    with open(args_dict_filename, 'r') as f:
        args_dict = json.load(f)

    return args_dict


def LoadPsychologist(args_dict):
    """
        Load psychologist from disk
    """
    with open(args_dict['psychologist-filename'], 'r') as f:
        psychologist = json.load(f)

    return {k: v for k, v in psychologist.items() if k in ['name', 'description']}


def SavePsychologist(args_dict, psychologist):
    with codecs.open(args_dict['psychologist-filename'], 'w', 'utf-8') as json_file:
        json.dump(psychologist, json_file, indent=4)


def LoadProjectManager(args_dict):
    """
        Load Project Manager from disk
    """
    with open(args_dict['project-manager-filename'], 'r') as f:
        projectmanager = json.load(f)

    return {k: v for k, v in projectmanager.items() if k in ['name', 'description']}


def SaveProjectManager(args_dict, project_manager):
    with codecs.open(args_dict['project-manager-filename'], 'w', 'utf-8') as json_file:
        json.dump(project_manager, json_file, indent=4)


def check_psychologist_exist(args_dict):
    """
        check if the experts list exist
    """

    if 'psychologist-filename' in args_dict:
        if os.path.exists(args_dict['psychologist-filename']):
            return True
        else:
            return False


def check_project_manager_exist(args_dict):
    """
        check if the experts list exist
    """

    if 'project-manager-filename' in args_dict:
        if os.path.exists(args_dict['project-manager-filename']):
            return True
        else:
            return False


def check_experts_list_exist(args_dict):
    """
        check if the experts list exist
    """

    if 'experts-list-filename' in args_dict:
        if os.path.exists(args_dict['experts-list-filename']):
            return True
        else:
            return False


def check_final_decisor_exist(args_dict):
    """
        check if the final decisor exist
    """

    if 'final-decision-maker-filename' in args_dict:
        if os.path.exists(args_dict['final-decision-maker-filename']):
            return True
        else:
            return False


def LoadExpertsFieldList(args_dict, botname):
    experts_path = script_dir / botname /"experts-list.json"
    with codecs.open(experts_path, 'r', 'utf-8') as json_file:
        fields_list = json.load(json_file)
    return fields_list['list']


def SaveExpertsFieldList(args_dict, experts_field):
    with codecs.open(args_dict['experts-list-filename'], 'w', 'utf-8') as json_file:
        json.dump(experts_field, json_file, indent=4)


def SaveExperts(args_dict, experts):
    #  save the data generated up to here
    with codecs.open(args_dict['experts-filename'], 'w', 'utf-8') as json_file:
        json.dump(experts, json_file, indent=4)


def check_experts_exist(args_dict):
    return os.path.exists(args_dict['experts-filename'])


def LoadExperts(args_dict, botname):
    """
        Load Project Manager from disk
    """

    experts_path = script_dir / botname / args_dict['experts-filename']
    """
        Load Project Manager from disk
    """
    with open(experts_path, 'r') as f:
        experts = json.load(f)

    return experts


def LoadSingleExpertAgent(args_dict, agent_id):
    """
        Load Project Manager from disk
    """
    with open(args_dict['experts-filename'], 'r') as f:
        experts = json.load(f)
    return experts[agent_id]


def check_final_decision_maker_exist(args_dict):
    return os.path.exists(args_dict['final-decision-maker-filename'])


def SaveFinalDecisionMaker(args_dict, final_decision_maker):
    with codecs.open(args_dict['final-decision-maker-filename'], 'w', 'utf-8') as json_file:
        json.dump(final_decision_maker, json_file, indent=4)


def LoadFinalDecisionMaker(args_dict, botname):
    final_dm_path = script_dir / botname / args_dict['final-decision-maker-filename']
    """
        Load Project Manager from disk
    """

    if is_without_description(args_dict):
        experts_fields = LoadExpertsFieldList(args_dict, botname=botname)
        description = FINAL_DECISOR_NO_DESCRIPTION_description_of_the_FDM.format(experts_fields=experts_fields,
                                                                                 task=args_dict['task'],
                                                                                 context=args_dict['context'])

        finaldecisionmaker = {"name": "Final Decision Maker", "description": description}
    else:
        with open(final_dm_path, 'r') as f:
            finaldecisionmaker = json.load(f)
    return {k: v for k, v in finaldecisionmaker.items() if k in ['name', 'description']}


def SaveQueriesAnswers(args_dict, queries_answers):
    #  save after each query.
    with codecs.open(args_dict['queries_answers_filename'], 'w', 'utf-8') as json_file:
        json.dump(queries_answers, json_file, indent=4)


def randomize_labels(actions):
    # actions = ["Moral action", "Immoral action"]
    random.shuffle(actions)
    return f"[A]: {actions[0]}\t [B]: {actions[1]}"

def get_queries(args_dict):
    queries = list()

    # if Path(args_dict['input']).parts[-1].endswith('.json'):
    #     #  if json
    #     queries_json = read_json(Path(args_dict['input']))
    #     for query in queries_json:
    #         if args_dict["dataset"] in ["empatheticDialogues_classification",
    #                                     "empatheticDialogues_generation"]:
    #             query_str = f"\n\n\n This is the story:\n'''\n{query["situation"]}\n'''\n{query["speaker"]}"
    #             queries.append(query_str)
    #
    #         elif args_dict["dataset"] in ["hitom", "opentom", "tomato"]:
    #             query_str = f"\n\n\n This is the story:\n'''\n{query["story"]}\n'''\n{query["question"]}\n{query["choices"]}\n"
    #             queries.append(query_str)
    #
    #         elif args_dict["dataset"] == "empatheticDialogues_generation":
    #             query_str = f"\n\n\n This is the story:\n'''\n{query["story"]}\n'''\n{query["question"]}\n{query["choices"]}\n"
    #             queries.append(query_str)
    #
    #         else:
    #             raise NotImplementedError("not implemented dataset")
    # elif Path(args_dict['input']).parts[-1].endswith('.jsonl'):
    #     queries_json = read_jsonl(Path(args_dict['input']))
    #     for query in queries_json:
    #         if args_dict["dataset"] in ["moral_story_choose_moral_action",
    #                                     ]:
    #
    #             moral_action = query["moral_action"]
    #             immoral_action = query["immoral_action"]
    #             labels = randomize_labels([moral_action, immoral_action])
    #
    #             query_str = f"This is the situation:\n'''{query["situation"]}. {query["intention"]}. {query["norm"]}.'''\n{labels}"
    #
    #             # query_str = f"\n\n\n This is the story:\n'''\n{query["situation"]}\n'''\n{query["speaker"]}"
    #             queries.append(query_str)
    #
    #
    # else:
        #  if txt
    with codecs.open(args_dict['input'], 'r', 'utf-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            queries.append(line)
    return queries


# def get_queries_answers_for_baselines(args_dict):
#     """
#
#     :param args_dict:
#     :return: queries_answers, queries_last_index
#     """
#     #  try to open queries answers file to resume, if exist
#     queries_last_index = 0
#     try:
#         with open(args_dict['queries_answers_filename'], 'r') as json_file:
#             queries_answers = json.load(json_file)  # Load the JSON file into a Python dictionary
#         if args_dict['resume']:
#             queries_last_index = get_queries_last_index_for_baseline(queries_answers)
#
#     except FileNotFoundError:
#         queries_answers = dict()
#
#     return queries_answers, queries_last_index

# def get_queries_answers_for_single_expert_agent(args_dict):
#     """
#
#     :param args_dict:
#     :return: queries_answers, queries_last_index
#     """
#     raise NotImplementedError("fix this function to consider experts")
#     #  try to open queries answers file to resume, if exist
#     queries_last_index = 0
#     try:
#         with open(args_dict['queries_answers_filename'], 'r') as json_file:
#             queries_answers = json.load(json_file)  # Load the JSON file into a Python dictionary
#         if args_dict['resume']:
#             queries_last_index = get_queries_last_index_for_an_expert_agent(queries_answers)
#
#     except FileNotFoundError:
#         queries_answers = dict()
#
#     return queries_answers, queries_last_index


def get_queries_answers(args_dict):
    if 'resume' in args_dict and args_dict['resume']:
        try:
            with open(args_dict['queries_answers_filename'], 'r') as json_file:
                return json.load(json_file)  # Load the JSON file into a Python dictionary
        except FileNotFoundError:
            return dict()
    #  if not resume
    return dict()

# def get_queries_answers_for_final_decisor_maker(args_dict):
#     """
#
#     :param args_dict:
#     :return: queries_answers, queries_last_index
#     """
#     raise NotImplementedError("fix this function to consider final_decisor_maker")
#     #  try to open queries answers file to resume, if exist
#     queries_last_index = 0
#     try:
#         with open(args_dict['queries_answers_filename'], 'r') as json_file:
#             queries_answers = json.load(json_file)  # Load the JSON file into a Python dictionary
#         if args_dict['resume']:
#             queries_last_index = get_queries_last_index_for_final_decisor_maker(queries_answers)
#
#     except FileNotFoundError:
#         queries_answers = dict()
#
#     return queries_answers, queries_last_index
