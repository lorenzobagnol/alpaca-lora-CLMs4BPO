from typing import Literal
from tqdm import tqdm
import openai
import os
import credentials
import pandas as pd
import numpy as np

from evaluate_with_gpt4 import eval
from generate import generate_on_dataset_one_step
from utils.dataset_utils import load_product_dataset, generate_input, single_to_multiple_prod_list


os.environ["OPENAI_API_KEY"] = credentials.api_key
openai.api_key = os.getenv("OPENAI_API_KEY")

def prompt_generator_loop(iterations:int):
    """
    Generate instructions for the Alpaca model to generate descriptions of luxury ship rooms containing products from the Amazon dataset.
    """

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
    for i in range(iterations):
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
        actual_generations_df=generate_on_dataset_one_step(new_instruction)
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
        




if __name__ == "__main__":
    
    dataset_path="./dataset/electronic-products.csv"
    data = load_product_dataset(dataset_path)
    input_list = generate_input(data, return_type="once")