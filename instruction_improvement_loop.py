import openai
import os
import pandas as pd
import numpy as np

from utils.evaluate_with_gpt4 import eval_with_openai
from generate import generate_on_dataset
from model.alpaca_lora import AlpacaLoraModel


def instruction_improvement_loop(iterations:int):
	"""
	Generate instructions for the Alpaca model to generate descriptions of luxury ship rooms containing products from the Amazon dataset.
	"""

	alpaca_model = AlpacaLoraModel()

	# start with hand-crafted instruction
	first_instruction = """
	I'll give you as "input" a sequence of products with their functionalities. Each product is in the form:

	<product> {product name}
	<features> {list of features of the product}

	You have to write a description of a luxury room containing these products. Do not copy the features of each product, instead try to focus on the user experience related to each product without listing technical details.
	Write it in an appealing tone for the website that will advertise the room.
	"""
	best_instruction = first_instruction
	best_mean_value = 0

	results_df = pd.DataFrame(columns=["instruction", "GPT-4_eval", "GPT-4_eval_mean_value"])

	# start the GPT-4 loop to generate new instructions and evaluate generations made by Alpaca
	for i in range(iterations):

		# GPT-4 generate new instruction
		instruction_generator_prompt=f"""
		This is the intruction I gave to an instruction-following model. 

		'''{best_instruction}'''

		Generate a better instruction for this model knowing that it tends not to consider all products so the intruction have to ensure that the description will integrate all of them.
		Don't forget to describe the structure of the input.
		Output only the new generated intruction.
		"""

		new_instruction = openai.ChatCompletion.create(
				model = "gpt-4o-mini",
				messages = [ {"role": "system","content": "You are a helpful assistant."},
							{"role": "user","content": instruction_generator_prompt}]
				)
		new_instruction = new_instruction["choices"][0]["message"]["content"]

		# Alpaca model makes generation using the new instruction
		actual_generations_df = generate_on_dataset(alpaca_model, new_instruction)

		# GPT-4 evaluates generations
		actual_eval_list = eval_with_openai(actual_generations_df["response"])
		# if mean value is better than the previous one, new_instruction become best_instruction
		actual_meam_value = np.mean(actual_eval_list)
		print("Instruction:\t"+new_instruction+"\n\nMean value evaluation:\t"+str(actual_meam_value)+"\n\n")
		if actual_meam_value>best_mean_value:
			actual_generations_df["eval"] = actual_eval_list
			actual_generations_df.to_csv("GPT-4_best_instruction_evaluations.csv",index = False)
			best_mean_value = actual_meam_value
			best_instruction = new_instruction

		# save results to csv	
		results_df.loc[len(results_df)] = [new_instruction, actual_eval_list, actual_meam_value]
		results_df.to_csv("GPT-4_loop_results.csv",index = False)
		


if __name__ == "__main__":

	openai.api_key = os.getenv("OPENAI_API_KEY")
	instruction_improvement_loop(10)


