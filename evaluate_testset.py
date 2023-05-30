import os
import sys
import json
from tqdm import tqdm
import fire
import gradio as gr
import torch
import random
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from utils.preprocess_webnlg import preprocess_webnlg_val
from utils.prompter import Prompter
from nltk.translate.bleu_score import corpus_bleu


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def main(
    load_8bit: bool = False,
    base_model: str = "huggyllama/llama-7b",
    lora_weights: str = "./lora-fixed-prompts",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        #model=PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    else: 
        print("no GPU found")
        exit

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)
        

    test=load_dataset("json", data_files="/home/bagnol/progetti/NLG/alpaca-lora/models_and_results/chatGPT-prompts/dataset/test.json")["train"]
    shuffled_test=test.shuffle(10)
    references=list()
    pred=list()
    output=list()
    with open("/home/bagnol/progetti/NLG/alpaca-lora/models_and_results/huggyllama-llama-7b/basemodel_prediction.json", "w") as file:
        for i in tqdm(range(100), "Making predictions on test set..."):
            pred.append(next(evaluate(shuffled_test[i]["instruction"],shuffled_test[i]["input"])).split("\n\n###")[0])
            references.append(shuffled_test[i]["output"])
            output.append(shuffled_test[i])
            output[i]["prediction"]=pred[i]
        json.dump(output, file, indent=4)
    print("Calculating BLEU score...")
    print("BLUE score is:")    
    print(corpus_bleu(references,pred))
        
if __name__ == "__main__":
    fire.Fire(main)