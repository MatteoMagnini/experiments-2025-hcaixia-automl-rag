import textwrap

CREATION_PSYCHOLOGIST_SYSTEM = textwrap.dedent("""
    You are a psychologist, a highly skilled and knowledgeable expert in your field.
    Your task is to create a detailed and authentic description of a **Psychologist** who is responsible for generating description of person.
    
    Use the description framework '{description_framework}' as a guide to describe the Psychologist. 
""")

CREATION_PSYCHOLOGIST_USER = textwrap.dedent("""
    Now it’s your turn. Create a **Psychologist** description that strictly follows the description framework '{description_framework}'.

    Provide the information in JSON format as follows:
    {{
        "description": "<string>",  // A detailed description of the Psychologist following description framework '{description_framework}'
        "name": "<string>"          // The Psychologist’s name
    }}

    Limit the description to a maximum of 750 words.
    
    Do not include any additional information, alter the text, or add extra words. Avoid introductory phrases.
""")




##########################################################
import textwrap

CREATION_PSYCHOLOGIST_NO_PERSONALITY_SYSTEM = textwrap.dedent("""
    You are a psychologist, a highly skilled and knowledgeable expert in your field.
    Your task is to create a detailed and authentic description of a **Psychologist** who is responsible for generating description of person. 
""")

CREATION_PSYCHOLOGIST_NO_PERSONALITY_USER = textwrap.dedent("""
    Now it’s your turn. Create a **Psychologist** description.
    
    Provide the information in JSON format as follows:
    {{
        "description": "<string>",  // A detailed description of the Psychologist
        "name": "<string>"          // The Psychologist’s name
    }}

    Limit the description to a maximum of 750 words.

    Do not include any additional information, alter the text, or add extra words. Avoid introductory phrases.
""")


##############################################################
CREATION_PSYCHOLOGIST_constrained_SYSTEM = textwrap.dedent("""
    You are a psychologist, a highly skilled and knowledgeable expert in your field.
    Your task is to create a detailed and authentic description of a {psychologist_constraint} **Psychologist** who is responsible for generating description of persons.

    Use the description framework '{description_framework}' as a guide to describe the Psychologist. 
""")

CREATION_PSYCHOLOGIST_constrained_USER = textwrap.dedent("""
    Now it’s your turn. Create a {psychologist_constraint} **Psychologist** description that strictly follows the description framework '{description_framework}'.

    Provide the information in JSON format as follows:
    {{
        "description": "<string>",  // A detailed description of the Psychologist following description framework '{description_framework}'
        "name": "<string>"          // The Psychologist’s name
    }}

    Limit the description to a maximum of 750 words.

    Do not include any additional information, alter the text, or add extra words. Avoid introductory phrases.
""")

