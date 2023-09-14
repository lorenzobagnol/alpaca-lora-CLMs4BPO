import openai
import os
import credentials
os.environ["OPENAI_API_KEY"] = credentials.api_key
openai.api_key = os.getenv("OPENAI_API_KEY")
from utils.prompter import Prompter
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

cables=pd.read_csv("../amazon-dataset/cables.csv")
amps=pd.read_csv("../amazon-dataset/home_audio.csv")
teles=pd.read_csv("../amazon-dataset/televisions.csv")
cables = cables.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
amps = amps.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
teles = teles.sample(frac=1, random_state=10).reset_index(drop=True)[:20]

eval_prompt="""
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

[Question]
I'll give you as "input" a sequence of products with their functionalities. Each product is in the form:

<product> {product name}
<features> {list of features of the product}

You have to write a description of a luxury ship room containing these products. Do not copy the features, try to focus on the user experience instead of the technical details.
Write it with an engaging tone for the ship website.
"""

def eval(response):
    prod_list=[prod for prod in amps["title"]]+[prod for prod in teles["title"]]+[prod for prod in cables["title"]]
    spec_list=[feat for feat in amps["feature"]]+[feat for feat in teles["feature"]]+[feat for feat in cables["feature"]]
    input=list()
    assert len(prod_list)==len(spec_list)
    for i in tqdm(range(20),desc="Evaluating"):
        input.append("<product> "+prod_list[i]+"\n"+"<features> "+spec_list[i]+"\n\n"+"<product> "+prod_list[20+i]+"\n"+"<features> "+spec_list[20+i]+"\n\n"+"<product> "+prod_list[40+i]+"\n"+"<functionalities> "+spec_list[40+i]+"\n\n")

    val_list=list()
    for i in tqdm(range(20)):
        prompt=eval_prompt+"Input:\n"+input[i]+"[The Start of Assistant's Answer]\n"+response[i]+"\n[The End of Assistant's Answer]"
        val = openai.ChatCompletion.create(
            model="gpt-4",
            messages= [ {"role": "system","content": "You are a helpful assistant."},
                        {"role": "user","content": prompt}],
            temperature=0)
        val_list.append(float(val["choices"][0]["message"]["content"].split("[[")[1].split("]]")[0]))
    return val_list

