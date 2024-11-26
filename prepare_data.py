"""
Prepare data from CSV database to LLM prompts
"""

import pandas as pd
import simplejson as json
import random

json_template = {
    "messages": [
        {
            "role": "system",
            "content": "You are an expert in behavioural science research that investigates how behavioural interventions can affect the choices people make. You specialize in providing concise, technical predictions for the impact of nudges on sustainable food choices, focusing on numerical values for the r-coefficient and Cohen's d.In every response, regardless of the level of available information, you will predict these values. In cases where the available data is insufficient or confidence in the prediction is low, you will explicitly state your level of confidence along with the numerical estimates. Your communication style is brief and highly technical."
        },
        {
            "role": "user",
            "content": "The research project ran an experiment in {} which investigates how {} affect {}."
        },
        {
            "role": "assistant",
            "content": "I understood the research background. Please tell me more about the experiment setup."
        },
        {
            "role": "user",
            "content": "The experiment is set up in {} where data is collected {} over a period of {} days. By {}, the experiment aims to observe {} as a result."
        },
        {
            "role": "assistant",
            "content": "I understood the experiment setup. Please tell more about the targeted audience."
        },
        {
            "role": "user",
            "content": "The targeted audience are {} who are in the {} category. There are a total of {} sampled audience.  Based on the information provided, please ensure you predict the effect direction (positive or negative), the r-coefficient value and the Cohens’ d value. Don’t be afraid to give your predicted value."
        },
        {
            "role": "assistant",
            "content": "The effect will be {}. Cohen's d: {}. r-coefficient: {}."
        }
    ]
}

df = pd.read_csv(file_path)
df['totalparticipants'] = df['sampleobscontrol'] + df['sampleobstreat']
df['totalparticipants'] = pd.to_numeric(df['totalparticipants'], errors='coerce')
df['sampleobscontrol'] = pd.to_numeric(df['sampleobscontrol'], errors='coerce')
df['sampleobstreat'] = pd.to_numeric(df['sampleobstreat'], errors='coerce')
df['r_coefficient'] = df['r_coefficient'].apply(lambda x: abs(round(x, 2)))
df['cohens_d'] = df['cohens_d'].apply(lambda x: abs(round(x, 2)))
df['true_direction'] = df.apply(lambda row: 'positive' if row['effectdirection'] == row['desired_direction'] else 'negative', axis=1)
df['totalparticipants'] = df['totalparticipants'].apply(lambda x: round(x, 0))

prompt1 = """The research project ran an experiment in {} which investigates how {} affect {}."""
def fill_prompt1(row, prompt1):
    values = [row[col] for col in ['studydatayear','interventiondescription','outcomedescription']]
    filled = prompt1.format(*values)
    return filled
prompt2 = """The experiment is set up in {} where data is collected {} over a period of {} days. By {}, the experiment aims to observe {} as a result."""
def fill_prompt2(row, prompt2):
    values = [row[col] for col in ['locationcountry','studyfrequency',
                                   'studydurationdays','interventiondescription',
                                   'outcomemeasuredescription']]
    filled = prompt2.format(*values)
    return filled
prompt3 = """The targeted audience are {} who are in the {} category. There are a total of {} sampled audience.  Based on the information provided, please ensure you predict the effect direction (positive or negative), the r-coefficient value and the Cohens’ d value. Don’t be afraid to give your predicted value."""
def fill_prompt3(row, prompt3):
    values = [row[col] for col in ['populationdescription', 'populationcategory', 'sampleobs']]
    filled = prompt3.format(*values)
    return filled
prompt4 = """The effect will be {}. Cohen's d: {}. r-coefficient: {}."""
def fill_prompt4(row, prompt4):
    values = [row[col] for col in ['true_direction', 'cohens_d', 'r_coefficient']]
    filled = prompt4.format(*values)
    return filled

df['prompt_research'] = df.apply(fill_prompt1, axis=1)
df['prompt_experiment'] = df.apply(fill_prompt2, axis=1)
df['prompt_audience'] = df.apply(fill_prompt3, axis=1)
df['prompt_result'] = df.apply(fill_prompt4, axis=1)
df.to_csv('dataset_with_prompts_filled.csv', index=False)

random.seed(1220)
for _, row in df.iterrows():
      json_output = json_template.copy()
      json_output["messages"][1]['content'] = row['prompt_research']
      json_output['messages'][3]['content'] = row['prompt_experiment']
      json_output['messages'][5]['content'] = row['prompt_audience']
      json_output['messages'][6]['content'] = row['prompt_result']

      res = random.choices(['T', 'V', 'S'], [0.7,0.1,0.2])[0]

      if res == 'T':
        with open('prompts_train.json', 'a') as file:
            json.dump(json_output, file, ensure_ascii=False, encoding="utf-8")
            file.write('\n')
      elif res == 'S':
        with open('prompts_test_with_answers.json', 'a') as file:
            json.dump(json_output, file, ensure_ascii=False, encoding="utf-8")
            file.write('\n')
      else:
        with open('prompts_validate.json', 'a') as file:
            json.dump(json_output, file, ensure_ascii=False, encoding="utf-8")
            file.write('\n')