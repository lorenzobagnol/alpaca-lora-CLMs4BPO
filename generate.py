from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from utils.prompter import Prompter
import torch
from peft import PeftModel
import torch
import pandas as pd
from tqdm import tqdm

two_steps=False

cables=pd.read_csv("../amazon-dataset/cables.csv")
amps=pd.read_csv("../amazon-dataset/home_audio.csv")
teles=pd.read_csv("../amazon-dataset/televisions.csv")
cables = cables.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
amps = amps.sample(frac=1, random_state=10).reset_index(drop=True)[:20]
teles = teles.sample(frac=1, random_state=10).reset_index(drop=True)[:20]

single_instruction="""
I'll give you as "input" a sequence of products with their functionalities. Each product is in the form:

<product> {product name}
<features> {list of features of the product}

You have to write a description of a luxury ship room containing these products. Do not copy the features, try to focus on the user experience instead of the technical details.
Write it with an engaging tone for the ship website.
"""

instruction_1="""
I'll give you as "input" a product with his features in the form:

<product> {product name}
<features> {list of features of the product} 

Provide me with the main things a user can do with this product. Do not copy the features, try to focus on the user experience instead of the technical details.
"""

instruction_2 = """
I'll give you as "input" a sequence of products with their functionalities. Each product is in the form:

<product> {product name}
<functionalities> {functionalities of the product}

You have to write a description of a luxury ship room containing these products. Write a text emphasizing the functionalities related to each product.
Write it with an engaging tone for the ship website.


"""

prod_list=[prod for prod in amps["title"]]+[prod for prod in teles["title"]]+[prod for prod in cables["title"]]
spec_list=[feat for feat in amps["feature"]]+[feat for feat in teles["feature"]]+[feat for feat in cables["feature"]]
input_1=list()
assert len(prod_list)==len(spec_list)
for i in range(len(prod_list)):
    input_1.append("<product> "+prod_list[i]+"\n<features> "+str(spec_list[i])+"\n\n")


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

prompter = Prompter()
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
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
        ).to(device)

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)


def generate_two_step():
    response_1=list()
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=400,
    )
    for i in tqdm(range(len(input_1))):
        prompt = prompter.generate_prompt(instruction_1, input_1[i])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids=input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response= prompter.get_response(response)
        response_1.append(response)

    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=800,
        do_sample=True,
        repetition_penalty=1.16
    )

    input_2=list()
    response_2=list()
    for i in tqdm(range(20)):
        input="<product> "+prod_list[i]+"\n"+"<functionalities> "+response_1[i]+"\n\n"+"<product> "+prod_list[20+i]+"\n"+"<functionalities> "+response_1[20+i]+"\n\n"+"<product> "+prod_list[40+i]+"\n"+"<functionalities> "+response_1[40+i]+"\n\n"
        input_2.append(input)
        prompt = prompter.generate_prompt(instruction_2, input)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response= prompter.get_response(response)
        response_2.append(response)


    df=pd.DataFrame(columns=["input_2","response"])
    for i in range(len(response_2)):
        df.loc[len(df)]=[input_2[i],response_2[i]]
    return df

def generate_one_step(instruction=single_instruction):
    generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=800,
    do_sample=True,
    repetition_penalty=1.16
    )

    one_step_input=list()
    one_step_response=list()
    for i in tqdm(range(20),desc="Generating"):
        input="<product> "+prod_list[i]+"\n"+"<features> "+spec_list[i]+"\n\n"+"<product> "+prod_list[20+i]+"\n"+"<features> "+spec_list[20+i]+"\n\n"+"<product> "+prod_list[40+i]+"\n"+"<features> "+spec_list[40+i]+"\n\n"
        one_step_input.append(input)
        prompt = prompter.generate_prompt(instruction, input)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response= prompter.get_response(response)
        one_step_response.append(response)


    df=pd.DataFrame(columns=["input","response"])
    for i in range(len(one_step_response)):
        df.loc[len(df)]=[one_step_input[i],one_step_response[i]]
    return df
