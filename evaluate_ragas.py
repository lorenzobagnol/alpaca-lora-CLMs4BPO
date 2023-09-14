import os
os.environ["OPENAI_API_KEY"] = "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"
from ragas import evaluate
import datasets
from ragas.metrics import answer_relevancy
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from utils.prompter import Prompter
import torch
from peft import PeftModel
import torch
import pandas as pd
from tqdm import tqdm

cables=pd.read_csv("../amazon-dataset/cables.csv")
amps=pd.read_csv("../amazon-dataset/home_audio.csv")
teles=pd.read_csv("../amazon-dataset/televisions.csv")
cables = cables.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
amps = amps.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
teles = teles.sample(frac=1, random_state=10).reset_index(drop=True)[:20]

instruction= """
I'll give you as "input" a sequence of products with their features. Each product is in the form:\n\
\n\
<product> {product name}\n\
<features> {list of features of the product}\n\
\n\
You have to write a description of a luxury ship room containing these products. Write a text emphasizing the functionalities related to each product providing with the main things a user can do with them.\
Write it with an engaging tone for the ship website.\n\
"""

prod_list=[prod for prod in amps["title"]]+[prod for prod in teles["title"]]+[prod for prod in cables["title"]]
spec_list=[feat for feat in amps["feature"]]+[feat for feat in teles["feature"]]+[feat for feat in cables["feature"]]
input=list()
assert len(prod_list)==len(spec_list)
for i in tqdm(range(20)):
    input.append("<product> "+prod_list[i]+"\n"+"<functionalities> "+spec_list[i]+"\n\n"+"<product> "+prod_list[20+i]+"\n"+"<functionalities> "+spec_list[20+i]+"\n\n"+"<product> "+prod_list[40+i]+"\n"+"<functionalities> "+spec_list[40+i]+"\n\n")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


response=pd.read_csv("./two_step.csv")["response"].to_list()
df=pd.DataFrame(columns=["question","answer"])
prompter = Prompter()
for i in range(len(input)):
    prompt = prompter.generate_prompt(instruction, input[i])
    df.loc[len(df)]=[prompt,response[i]]
data=datasets.Dataset.from_pandas(df,preserve_index=False)
results = evaluate(data, metrics=[answer_relevancy])
print("answer relevancy: "+str(results["answer_relevancy"]))