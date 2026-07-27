import json

from prompts.tools_instructions import CLEAN_LIST_SYSTEM, CLEAN_LIST_USER, EXTRACT_BASE_USER, \
    EXTRACT_NAME_SYSTEM, EXTRACT_DESCRIPTION_SYSTEM, EXTRACT_JUSTIFICATION_SYSTEM, \
    EXTRACT_GRADE_SYSTEM, EXTRACT_FINAL_ANSWER_SYSTEM, \
    EXTRACT_CONFIDENCE_SCORE_SYSTEM, EXTRACT_REASONING_STEPS_SYSTEM, EXTRACT_CONCLUSION_SYSTEM

from prompts.chat_template import chat_template
import requests
from time import time
# import numpy as np
from nltk import sent_tokenize
from pathlib import Path

import os


from utilities import is_openrouter
#  set seed

seed_value = 23

PATTERN_TO_REMOVE = "<|start_header_id|>assistant<|end_header_id|>"

script_dir = Path(__file__).parent #per path relativi


##################################
#  initialize the model
##################################

##################################
def SendToLLM(args_dict,
              messages, model=None,
              tokenizer=None,
              device=None,
              temperature=1.2,
              nucleus=0.0,
              alternatives=2,
              max_tokens=1024):
    # if is_openrouter(args_dict):
    return SendToLLMOpenRouter(messages=messages,
                                   args_dict=args_dict,
                                   temperature=temperature,
                                   nucleus=nucleus,
                                   max_tokens=max_tokens
                                   )

def SendToLLMOpenRouter(messages: list,
                        args_dict: dict, 
                        apikey:str,
                        botname : str
                        # temperature=1.2,
                        # nucleus=0.0,
                        # max_tokens=150
                        ) -> dict:
    """

    :param messages:  the list of messages
    :param args_dict:  the config_file
    :param temperature:  the temperature
    :param max_tokens: the max number of tokens generated

    :return: response_message, messages, generation_probability.item(), generation_time

    """

    #  load openrouter config file
    config_path = script_dir / botname / args_dict['openrouterfile']
    with open(config_path, 'r') as f:
        openrouter_config = json.load(f)

    #  read apikey - classic
    # apikey = open(openrouter_config["headers"]["Authorization"], 'r').read().strip()
    #  read apikey for chatFAQ app integration
    

    openrouter_config["headers"]["Authorization"] = f"Bearer {apikey}"
    openrouter_config["dumps"]["messages"] = messages

    start_time = time()
    response = requests.post(
        url=openrouter_config["url"],
        headers=openrouter_config["headers"],
        data=json.dumps(openrouter_config["dumps"])
    )
    generation_time = time() - start_time

    text_response_json = json.loads(response.text)
    try:
        response_message = text_response_json['choices'][0]['message']['content']
    except KeyError: #KeyError: 'choices'
        print(f"Error in response: {text_response_json}")
        raise KeyError("The response does not contain 'choices' or 'message' keys.")


    outdict = {
        "response_message": response_message,

        "generation_probability": 'unknown',
        "generation_time": generation_time,
        "messages": messages,

        "generated_sequences": [response_message],
        "probabilities": []
    }
    return outdict


def update_messages(messages: list,
                    role: str,
                    query: str,
                    ):
    messages.append({"role": role,
                     "content": query})
    return messages


def clean_list(args_dict,
               list_string: str,
               model=None,
               tokenizer=None,
               device=None,
               temperature=1.2,
               nucleus=0.0,
               alternatives=2,
               max_tokens=4096) -> str:
    """

    :param list_string: (str) the list to clean
    :param model:  (str)the Large language model
    :param temperature: (float) the temperature
    :param max_tokens: (int) the max number of tokens generated
    :return: (str)the text generated

    """

    messages = list()
    messages.append({"role": "system",
                     "content": CLEAN_LIST_SYSTEM})
    messages.append({"role": "user",
                     "content": CLEAN_LIST_USER.format(text=list_string)})
    outdict = SendToLLM(
        args_dict=args_dict,
        messages=messages,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        alternatives=alternatives,
                        max_tokens=max_tokens)
    return outdict["response_message"]
    # return response_message


def extract_list_items(list_string: str) -> list:
    return [item.strip() for item in list_string.split('\n')]


def extract_base(args_dict: dict, base_prompt: str,
                 string: str,
                 model=None,
                 tokenizer=None,
                 device=None,
                 temperature=1.2,
                 nucleus=0.0,
                 max_tokens=4096,
                 ) -> str:
    """

        base_prompt is the prompt to use to extract the data from

                    Base extract method

    """

    messages = list()
    messages.append({"role": "system",
                     "content": base_prompt})
    messages.append({"role": "user",
                     "content": EXTRACT_BASE_USER.format(text=string)})
    outdict = SendToLLM(
        args_dict=args_dict,
        messages=messages,
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=temperature,
        nucleus=nucleus,
        max_tokens=max_tokens)
    return outdict["response_message"]


def extract_name(args_dict: dict,
                 list_string: str,
                 model=None,
                 tokenizer=None,
                 device=None,
                 temperature=1.2,
                 nucleus=0.0,
                 max_tokens=64) -> str:
    return extract_base(args_dict=args_dict,
                        base_prompt=EXTRACT_NAME_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_description(args_dict: dict, list_string: str,
                        model=None,
                        tokenizer=None,
                        device=None,
                        temperature=1.2,
                        nucleus=0.0,
                        max_tokens=4096) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_DESCRIPTION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_grade(args_dict: dict, list_string: str,
                  model=None,
                  tokenizer=None,
                  device=None,
                  temperature=1.2,
                  nucleus=0.0,
                  max_tokens=32) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_GRADE_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_justification(args_dict: dict, list_string: str,
                          model=None,
                          tokenizer=None,
                          device=None,
                          temperature=1.2,
                          nucleus=0.0,
                          max_tokens=248) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_JUSTIFICATION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_final_answer(args_dict: dict, list_string: str,
                         model=None,
                         tokenizer=None,
                         device=None,
                         temperature=1.2,
                         nucleus=0.0,
                         max_tokens=1024) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_FINAL_ANSWER_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_confidence_score(args_dict: dict, list_string: str,
                             model=None,
                             tokenizer=None,
                             device=None,
                             temperature=1.2,
                             nucleus=0.0,
                             max_tokens=16) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_CONFIDENCE_SCORE_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_reasoning_steps(args_dict: dict, list_string: str,
                            model=None,
                            tokenizer=None,
                            device=None,
                            temperature=1.2,
                            nucleus=0.0,
                            max_tokens=2048) -> list:
    outdict = extract_base(args_dict=args_dict, base_prompt=EXTRACT_REASONING_STEPS_SYSTEM,
                           string=list_string,
                           model=model,
                           tokenizer=tokenizer,
                           device=device,
                           temperature=temperature,
                           nucleus=nucleus,
                           max_tokens=max_tokens,
                           )
    return sent_tokenize(outdict, language='english')
    # return extract_list_items(response_message)


def extract_conclusion(args_dict: dict, list_string: str,
                       model=None,
                       tokenizer=None,
                       device=None,
                       temperature=1.2,
                       nucleus=0.0,
                       max_tokens=1024) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_CONCLUSION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def create_experts_answers_string(query_answers: list, experts: list) -> str:
    experts_answers_string = '[\n'
    for ans, expert in zip(query_answers, experts):
        expert_string = "{\n"
        try:
            expert_string += f"\"expert-name\": \"{expert['name']}\",\n"
        except KeyError:
            pass
        expert_string += f"\"expert-field\": \"{expert['field']}\",\n"
        expert_string += f"\"answer\": \"{ans['final_answer']}\",\n"
        expert_string += f"\"grade\": \"{ans['grade']}\",\n"
        expert_string += f"\"confidence-score\": \"{ans['confidence_score']}\",\n"
        expert_string += f"\"justification\": \"{ans['justification']}\",\n"
        expert_string += f"\"reasoning-steps\": \"{ans['reasoning_steps']}\",\n"
        expert_string += f"\"conclusion\": \"{ans['conclusion']}\"\n"
        expert_string += '}\n'

        experts_answers_string += expert_string
    # print(f"{experts_answers_string=}")

    experts_answers_string += '\n]'

    return experts_answers_string
