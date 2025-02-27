from transformers import GenerationConfig
import torch
import torch
import pandas as pd
from tqdm import tqdm

from model.alpaca_lora import AlpacaLoraModel


instruction_cot_1="""
I'll give you as "input" a product with his features in the form:

<product> {product name}
<features> {list of features of the product} 

Provide me with the main things a user can do with this product. Do not copy the features, try to focus on the user experience instead of the technical details.
"""

instruction_cot_2 = """
I'll give you as "input" a sequence of products with their functionalities. Each product is in the form:

<product> {product name}
<functionalities> {functionalities of the product}

You have to write a description of a luxury ship room containing these products. Ensure that you include each of these products in your description. Write a text emphasizing the functionalities related to each product.
Write it with an engaging tone for the ship website.


"""

basic_instruction = """
	I'll give you as "input" a sequence of products with their functionalities. Each product is in the form:

	<product> {product name}
	<features> {list of features of the product}

	You have to write a description of a luxury room containing these products. Do not copy the features of each product, instead try to focus on the user experience related to each product without listing technical details.
	Write it in an appealing tone for the website that will advertise the room.
	"""

# def generate_two_step(step_one_instruction:str, step_two_instruction:str, input_product_list:list):

#     response_one=list()
#     generation_config = GenerationConfig(
#         num_beams=4,
#         max_new_tokens=400,
#     )
#     for i in tqdm(range(len(input_product_list))):
#         prompt = prompter.generate_prompt(step_one_instruction, input_product_list[i])
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#         input_ids=input_ids.to(device)
#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#             )
#         response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
#         response= prompter.get_response(response)
#         response_one.append(response)

#     generation_config = GenerationConfig(
#         temperature=0.2,
#         top_p=0.75,
#         top_k=40,
#         num_beams=1,
#         max_new_tokens=800,
#         do_sample=True,
#         repetition_penalty=1.16
#     )

#     input_2=list()
#     response_2=list()

#     for i in tqdm(range(20)):
#         input="<product> "+prod_list[i]+"\n"+"<functionalities> "+response_one[i]+"\n\n"+"<product> "+prod_list[20+i]+"\n"+"<functionalities> "+response_one[20+i]+"\n\n"+"<product> "+prod_list[40+i]+"\n"+"<functionalities> "+response_one[40+i]+"\n\n"
#         input_2.append(input)
#         prompt = prompter.generate_prompt(step_two_instruction, input)
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#         input_ids = input_ids.to(device)
#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#             )
#         response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
#         response= prompter.get_response(response)
#         response_2.append(response)


#     df=pd.DataFrame(columns=["input_1","response"])
#     for i in range(len(response_2)):
#         df.loc[len(df)]=[input_1[i],response_2[i]]
#     return df

def generate_on_dataset(alpaca_model: AlpacaLoraModel, instruction:str):

	df = pd.read_csv("./dataset/electronic-products.csv", index_col=False)

	output_df = pd.DataFrame(columns=["room", "response"])

	for i in tqdm(range(20),desc="Generating"):
		input = ""
		for index, row in df.loc[df["room"]==i]:
			input = input + "<product> "+row["title"]+"\n<features> "+str(row["feature"])+"\n\n"

		response = alpaca_model.generate(instruction, input)

		temp_df = pd.DataFrame([[i, response]], columns=["room", "response"])
		output_df = pd.concat([output_df, temp_df], ignore_index=True)

	return df


if __name__=="__main__":

	alpaca_model = AlpacaLoraModel()
	df = generate_on_dataset(alpaca_model, basic_instruction)