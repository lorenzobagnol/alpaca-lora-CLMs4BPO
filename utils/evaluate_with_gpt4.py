import openai
import os

import pandas as pd
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

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

def eval_with_openai(response):

	df = pd.read_csv("./dataset/electronic-products.csv", index_col=False)

	val_list=list()
	for i in tqdm(range(20),desc="Evaluating"):

		input = ""
		for index, row in df.loc[df["room"]==i]:
			input = input + "<product> "+row["title"]+"\n<features> "+str(row["feature"])+"\n\n"

		prompt=eval_prompt+"Input:\n"+input[i]+"[The Start of Assistant's Answer]\n"+response[i]+"\n[The End of Assistant's Answer]"
		val = openai.ChatCompletion.create(
			model="gpt-4o-mini",
			messages= [ {"role": "system","content": "You are a helpful assistant."},
						{"role": "user","content": prompt}],
			temperature=0)
		val_list.append(float(val["choices"][0]["message"]["content"].split("[[")[1].split("]]")[0]))
	return val_list

