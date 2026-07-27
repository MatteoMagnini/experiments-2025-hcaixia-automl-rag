from PoE3.PoE.CreateExperts import get_experts_fields
from PoE3.PoE.ModelRequests import extract_description, extract_name, SendToLLM
from PoE3.PoE.FileToolkit import SaveFinalDecisionMaker
from PoE3.PoE.prompts.final_decision_maker import CREATION_FINAL_DECISOR_USER, CREATION_FINAL_DECISOR_SYSTEM
import json


def CreateFinalDecisionMaker(args_dict,
                             experts
                             ):
    experts_fields_list = get_experts_fields(experts)
    messages = list()
    messages.append({"role": "system",
                     "content": CREATION_FINAL_DECISOR_SYSTEM.format(
                         psychologist_description=args_dict['psychologist']['description'],
                         description_framework=args_dict['description_framework'],
                         experts_fields=experts_fields_list,
                         task=args_dict['task'],
                         context=args_dict['context'])
                     })
    messages.append({"role": "user",
                     "content": CREATION_FINAL_DECISOR_USER.format(
                         description_framework=args_dict['description_framework'],
                         # experts_fields=experts_fields_list,
                         task=args_dict['task'],
                         context=args_dict['context'])})

    outdict = SendToLLM(args_dict=args_dict,
                        messages=messages,
                        model=args_dict['model'],
                        tokenizer=args_dict['tokenizer'],
                        device=args_dict['device'],
                        temperature=args_dict['temperature'],
                        nucleus=args_dict['nucleus'],
                        max_tokens=1024)
    answer = outdict["response_message"]
    #  if the data is parsable as JSON use it, otherwise, use the extracted data
    try:
        tmp_answer = answer.replace("```json\n", "").replace("```", "")
        tmp_final_decisor_dict = json.loads(tmp_answer)
        name = tmp_final_decisor_dict['name']
        description = tmp_final_decisor_dict['description']

    except:
        name = extract_name(args_dict=args_dict,
                            list_string=answer,
                            model=args_dict['model'],
                            tokenizer=args_dict['tokenizer'],
                            device=args_dict['device'],
                            temperature=0,
                            nucleus=0,
                            max_tokens=12)
        description = extract_description(args_dict=args_dict,
                            list_string=answer,
                                          model=args_dict['model'],
                                          tokenizer=args_dict['tokenizer'],
                                          device=args_dict['device'],
                                          temperature=0,
                                          nucleus=0,
                                          max_tokens=1024)
    print(f"Final Decision Maker: {name}\nDescription: {description}")

    outdict["name"] = name
    outdict["description"] = description
    return outdict
