import json
import re

from tqdm import tqdm

from FileToolkit import SaveQueriesAnswers, LoadExperts, LoadFinalDecisionMaker, get_queries, \
    LoadSingleExpertAgent, LoadExpertsFieldList, get_queries_answers
from ModelRequests import SendToLLM, create_experts_answers_string, extract_grade, extract_confidence_score, \
    extract_reasoning_steps, extract_conclusion, extract_justification, extract_final_answer, SendToLLMOpenRouter
from prompts.experts import ASK_TO_EXPERT_USER, ASK_TO_EXPERT_SYSTEM, ASK_TO_EXPERT_NO_DESCRIPTION_SYSTEM 
from prompts.final_decision_maker import ASK_FINAL_ANSWER_USER, ASK_FINAL_ANSWER_SYSTEM
from utilities import already_asked_to_query_expert_agents, already_asked_to_query_final_decision_maker


import re
from json_repair import repair_json


from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel

class ExpertOutput(BaseModel):
    reasoning_steps: str
    confidence_score: float
    grade: int
    justification: str
    conclusion: str
    final_answer: str


def extract_json_braces(text: str) -> str:
    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove unclosed <think> block
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

    # Extract content from markdown code fences if present
    fence_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)```",
        cleaned
    )

    cleaned = (
        fence_match.group(1).strip()
        if fence_match
        else cleaned
    )

    # Extract JSON object
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    extracted = match.group(0) if match else cleaned.strip()

    # Repair malformed JSON
    return repair_json(extracted)



def run_single_expert(
    expert_ID,
    expert,
    args_dict,
    apikey,
    botname,
    query,
    max_retries=2
):
    """
    Single expert worker executed in parallel.
    """

    # -------------------------
    # BUILD PROMPT
    # -------------------------
    if args_dict['description_framework'] == "no-description":

        system_msg = ASK_TO_EXPERT_NO_DESCRIPTION_SYSTEM.format(
            field=expert['field'],
            task=args_dict['task'],
            context=args_dict['context']
        )

    else:

        system_msg = ASK_TO_EXPERT_SYSTEM.format(
            expert_description=expert['description'],
            task=args_dict['task'],
            context=args_dict['context']
        )

    base_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": ASK_TO_EXPERT_USER.format(query=query)}
    ]

    messages = base_messages.copy()

    last_raw = None

    # -------------------------
    # RETRY LOOP
    # -------------------------
    for attempt in range(max_retries + 1):

        try:

            outdict = SendToLLMOpenRouter(
                messages=messages,
                args_dict=args_dict,
                apikey=apikey,
                botname=botname
            )

            raw_answer = outdict['response_message']
            last_raw = raw_answer

            # -------------------------
            # PARSE JSON
            # -------------------------
            tmp_answer = extract_json_braces(raw_answer)
            tmp_dict = json.loads(tmp_answer)

            # -------------------------
            # PYDANTIC VALIDATION
            # -------------------------
            validated = ExpertOutput.model_validate(tmp_dict)

            tmp_dict = validated.model_dump()

            # -------------------------
            # SUCCESS
            # -------------------------
            outdict['expert-id'] = expert_ID
            outdict['final_answer'] = tmp_dict['final_answer']
            outdict['grade'] = tmp_dict['grade']
            outdict['confidence_score'] = tmp_dict['confidence_score']
            outdict['reasoning_steps'] = tmp_dict['reasoning_steps']
            outdict['justification'] = tmp_dict['justification']
            outdict['conclusion'] = tmp_dict['conclusion']

            return outdict

        except Exception as e:

            print(
                f"Validation failed "
                f"(expert {expert_ID}, attempt {attempt}): {e}"
            )

            # -------------------------
            # MAX RETRIES REACHED
            # -------------------------
            if attempt == max_retries:

                return {
                    "expert-id": expert_ID,
                    "error": str(e),
                    "raw_answer": last_raw
                }

            # -------------------------
            # RETRY WITH FEEDBACK
            # -------------------------
            messages = base_messages + [{
                "role": "user",
                "content": f"""
Your previous output was INVALID.

Validation error:
{str(e)}

You MUST fix your response.

STRICT RULES:
- Output ONLY valid JSON
- ALL fields are mandatory
- No markdown
- No explanations
- No extra keys
- No missing keys

Required schema:

{{
  "reasoning_steps": "string",
  "confidence_score": float,
  "grade": integer,
  "justification": "string",
  "conclusion": "string",
  "final_answer": "string"
}}

Previous invalid output:

{raw_answer}
"""
            }]

    # no valid responses
    return {
        "expert-id": expert_ID,
        "error": "Unknown failure",
        "raw_answer": last_raw
    }


def AskToExperts(
    args_dict: dict,
    apikey,
    botname: str,
    experts,
    query=None,
    max_retries=2,
    max_workers=8
):
    """
    Parallel expert execution with:
    - retries
    - validation
    - pydantic schema enforcement
    """

    answers = []


    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                run_single_expert,
                expert_ID,
                expert,
                args_dict,
                apikey,
                botname,
                query,
                max_retries
            )
            for expert_ID, expert in enumerate(experts)
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Experts processed",
            ascii=True
        ):

            result = future.result()
            answers.append(result)

    # keep original ordering
    answers.sort(key=lambda x: x['expert-id'])

    return answers




def QueriesToExpertAgents(args_dict, botname, apikey, query):
    #  Load experts

    if args_dict['description_framework']=='no-description':
        # load expertize fields
        expertizes = LoadExpertsFieldList(args_dict, botname)
        # rename the dict fields from 'list' to 'field'
        experts = list()
        for field in expertizes:
            experts.append({"field": field})
    else:
        #  load expert personalities
        experts = LoadExperts(args_dict, botname)

    #  Load queries and answers
    return AskToExperts(args_dict=args_dict,apikey=apikey,
                                     experts=experts, botname=botname,
                                     query=query)





def QueriesToMakeFinalDecision(args_dict, botname, apikey, answers_experts, query):
    #  load agents
    if args_dict['description_framework']=='no-description':
        # load expertize fields
        expertizes = LoadExpertsFieldList(args_dict, botname)
        # rename the dict fields from 'list' to 'field'
        experts = list()
        for field in expertizes:
            experts.append({"field": field})
    else:
        #  load expert personalities
        experts = LoadExperts(args_dict, botname)
    
    final_decision_maker = LoadFinalDecisionMaker(args_dict, botname)

    outdict_fdm = AskToFinalDecisionMaker(
        args_dict,apikey,
        botname=botname,
        final_decision_maker=final_decision_maker,
        experts=experts,
        query=query,
        query_answers=answers_experts)
    #  update results in the dictionary

    # SaveResults(args_dict, outdict_fdm)

    return outdict_fdm

def AskToFinalDecisionMaker(args_dict: dict,
                            apikey,
                            botname,
                            final_decision_maker,
                            experts,
                            query="",
                            query_answers=list()):
    experts_answers = create_experts_answers_string(query_answers, experts)
    messages = [{"role": "system",
                 "content": ASK_FINAL_ANSWER_SYSTEM.format(
                     description=final_decision_maker['description'],
                     task=args_dict['task'],
                     context=args_dict['context'],
                 )
                 }, {"role": "user", "content": ASK_FINAL_ANSWER_USER.format(query=query,
                                                                             experts_answers=experts_answers,
                                                                             )
                     }]
    outdict = SendToLLMOpenRouter(args_dict=args_dict,apikey=apikey, botname=botname,
        messages=messages,
                        # model=args_dict['model'],
                        # tokenizer=args_dict['tokenizer'],
                        # device=args_dict['device'],
                        # temperature=args_dict['temperature'],
                        # nucleus=args_dict['nucleus'],
                        # max_tokens=1024
                                  )
    raw_answer = outdict['response_message']
    outdict_fdm = {f"final-decision-maker-{k}": v for k, v in outdict.items()}

    try:
        tmp_answer_final = extract_json_braces(raw_answer)
        tmp_dict = json.loads(tmp_answer_final)
        reasoning_steps = tmp_dict['reasoning_steps']
        conclusion = tmp_dict['conclusion']
        final_answer = tmp_dict['final_answer']

    except:
        #  using LLM strategy

        reasoning_steps = extract_reasoning_steps(raw_answer,
                                                  model=args_dict['model'],
                                                  tokenizer=args_dict['tokenizer'],
                                                  device=args_dict['device'],
                                                  temperature=0,
                                                  nucleus=0,
                                                  max_tokens=512,
                                                  )

        conclusion = extract_conclusion(raw_answer,
                                        model=args_dict['model'],
                                        tokenizer=args_dict['tokenizer'],
                                        device=args_dict['device'],
                                        temperature=0,
                                        nucleus=0,
                                        max_tokens=256,
                                        )
        final_answer = extract_final_answer(raw_answer,
                                            model=args_dict['model'],
                                            tokenizer=args_dict['tokenizer'],
                                            device=args_dict['device'],
                                            temperature=0,
                                            nucleus=0,
                                            max_tokens=256,
                                            )

    outdict_fdm['final-decision-maker-answer'] = final_answer
    outdict_fdm['final-decision-maker-reasoning-steps'] = reasoning_steps
    outdict_fdm['final-decision-maker-conclusion'] = conclusion
    return outdict_fdm
