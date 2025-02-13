import openai
import os
import credentials
os.environ["OPENAI_API_KEY"] = credentials.api_key
openai.api_key = os.getenv("OPENAI_API_KEY")
from evaluate_gpt4 import eval
from generate import generate_one_step
import pandas as pd
from tqdm import tqdm
import numpy as np



# load the dataset

cables=pd.read_csv("../amazon-dataset/cables.csv")
amps=pd.read_csv("../amazon-dataset/home_audio.csv")
teles=pd.read_csv("../amazon-dataset/televisions.csv")
cables = cables.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
amps = amps.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
teles = teles.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
prod_list=[prod for prod in amps["title"]]+[prod for prod in teles["title"]]+[prod for prod in cables["title"]]
spec_list=[feat for feat in amps["feature"]]+[feat for feat in teles["feature"]]+[feat for feat in cables["feature"]]
input=list()
assert len(prod_list)==len(spec_list)
for i in tqdm(range(20)):
    input.append("<product> "+prod_list[i]+"\n"+"<features> "+spec_list[i]+"\n\n"+"<product> "+prod_list[20+i]+"\n"+"<features> "+spec_list[20+i]+"\n\n"+"<product> "+prod_list[40+i]+"\n"+"<functionalities> "+spec_list[40+i]+"\n\n")



# start with hand-crafted instruction
first_instruction="""
I'll give you as "input" a sequence of products with their functionalities. Each product is in the form:

<product> {product name}
<features> {list of features of the product}

You have to write a description of a luxury ship room containing these products. Do not copy the features, try to focus on the user experience instead of the technical details.
Write it with an engaging tone for the ship website.
"""
best_instruction=first_instruction
prompt_df=pd.DataFrame(columns=["prompt", "results", "mean_value"])
best_generations_df=pd.read_csv("./initial_instruction_evaluations.csv")
max_value_list=list(best_generations_df["eval"])
max_mean_value=np.mean(max_value_list)
print("Instruction:\t"+first_instruction+"\n\nMean value evaluation:\t"+str(max_mean_value))
prompt_df.loc[len(prompt_df)]=[first_instruction, max_value_list, max_mean_value]


# start the GPT-4 loop to generate new instructions and evaluate generations made by Alpaca
for i in range(20):
    # GPT-4 generate new instruction
    instruction_generator_prompt=f"""
    This is the actual intruction of my instruction-following model. 

    '''{best_instruction}'''

    Generate a better instruction knowing that my model tends not to consider all products so the intruction have to ensure that the description will integrate all of them.
    Don't forget to describe the structure of the input.
    Output only the new generated intruction.
    """
    new_instruction = openai.ChatCompletion.create(
            model="gpt-4",
            messages= [ {"role": "system","content": "You are a helpful assistant."},
                        {"role": "user","content": instruction_generator_prompt}])
    new_instruction=new_instruction["choices"][0]["message"]["content"]
    # Alpaca model makes generation using the new instruction
    actual_generations_df=generate_one_step(new_instruction)
    # GPT-4 evaluates generations
    actual_eval_list=eval(actual_generations_df["response"])
    # if mean value is better than the previous one, new_instruction become best_instruction
    actual_meam_value=np.mean(actual_eval_list)
    prompt_df.loc[len(prompt_df)]=[new_instruction, actual_eval_list, actual_meam_value]
    print("Instruction:\t"+new_instruction+"\n\nMean value evaluation:\t"+str(actual_meam_value)+"\n\n")
    if actual_meam_value>max_mean_value:
        actual_generations_df["eval"]=actual_eval_list
        actual_generations_df.to_csv("GPT-4_best_instruction_evaluations.csv",index=False)
        max_mean_value=actual_meam_value
        max_value_list=actual_eval_list
        best_instruction=new_instruction
    prompt_df.to_csv("GPT-4_loop_results.csv",index=False)
    