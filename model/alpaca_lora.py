from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import torch
from peft import PeftModel
import torch

from utils.prompter import Prompter


class AlpacaLoraModel():

	def __init__(self):

		if torch.cuda.is_available():
			device = "cuda"
		else:
			device = "cpu"
		try:
			if torch.backends.mps.is_available():
				device = "mps"
		except:  
			pass
		self.device = device
		self.prompter = Prompter()
		self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
		model = LlamaForCausalLM.from_pretrained(
			"huggyllama/llama-7b",
			load_in_8bit=False,
			torch_dtype=torch.float16,
			device_map="auto",
		)
		model = PeftModel.from_pretrained(
					model,
					"tloen/alpaca-lora-7b",
					torch_dtype=torch.float16,
				).to(self.device)

		# unwind broken decapoda-research config
		model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
		model.config.bos_token_id = 1
		model.config.eos_token_id = 2

		model.eval()
		model = torch.compile(model) # necessary from torch 2.x
		
		self.model = model

		self.gen_conf = GenerationConfig(
							temperature=0.2,
							top_p=0.75,
							top_k=40,
							num_beams=1,
							max_new_tokens=800,
							do_sample=True,
							repetition_penalty=1.16
							)

	def generate(self, instruction: str, input: str) -> str :
		
		prompt = self.prompter.generate_prompt(instruction, input)
		input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
		input_ids = input_ids.to(self.device)
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				generation_config=self.gen_conf,
				return_dict_in_generate=True,
				output_scores=True,
			)
		response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
		response = self.prompter.get_response(response)
		return response
	
	def set_generation_conf(self, value: GenerationConfig):
		self.gen_conf = value