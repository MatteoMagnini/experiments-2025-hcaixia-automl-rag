import textwrap

""""

In this file there are the instructions used to create the P.O.E.

"""
CLEAN_LIST_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to clean the given text and provide a list of the described elements, with each item on a new line. Remove any item markers from the text. Do not alter the text of the items or add extra words. Omit any introductory phrases in your responses. Provide only the list.
""")

CLEAN_LIST_USER = textwrap.dedent("""
This is the text: {text}
""")

EXTRACT_BASE_USER = textwrap.dedent("""
Extract the data from the following text: {text}
""")

EXTRACT_NAME_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the name of the person described in the text. Do not alter the text or add any extra words. Remove any introductory phrases. Provide only the name.
""")

EXTRACT_DESCRIPTION_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the description of the person described in the text. Do not report the name. Do not alter the text, add extra words, or include introductory phrases. Provide only the description.
""")

EXTRACT_BIRTH_DATE_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the birth date of the person described in the text. Do not report the name. Do not alter the text, add extra words, or include introductory phrases. Provide only the birth date.
""")

EXTRACT_BIRTHPLACE_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the birthplace of the person described in the text. Do not report the name. Do not alter the text, add extra words, or include introductory phrases. Provide only the birthplace.
""")

EXTRACT_NATIONALITY_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the nationality of the person described in the text. Do not report the name. Do not alter the text, add extra words, or include introductory phrases. Provide only the nationality.
""")

EXTRACT_GENDER_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the gender of the person described in the text. Do not report the name. Do not alter the text, add extra words, or include introductory phrases. Provide only the gender.
""")

EXTRACT_CONFIDENCE_SCORE_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the confidence score provided in the text. Do not report any other information. Do not alter the text, add extra words, or include introductory phrases. Provide only the confidence score.
""")

EXTRACT_GRADE_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the grade provided in the text. Do not report any other information. Do not alter the text, add extra words, or include introductory phrases. Provide only the grade.
""")

EXTRACT_JUSTIFICATION_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the justification provided in the text. Do not report any other information. Do not alter the text, add extra words, or include introductory phrases. Provide only the justification.
""")

EXTRACT_CONCLUSION_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the conclusion stated in the text. Do not report any other information. Do not alter the text, add extra words, or include introductory phrases. Provide only the conclusion.
""")

EXTRACT_REASONING_STEPS_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the reasoning steps described in the text. Do not report any other information. Do not alter the text, add extra words, or include introductory phrases. Provide the reasoning steps with one step per line.
""")

EXTRACT_FINAL_ANSWER_SYSTEM = textwrap.dedent("""
You are an intelligent AI tool. Your task is to extract the final answer provided in the text. Do not report any other information. Do not alter the text, add extra words, or include introductory phrases. Provide only the final answer.
""")