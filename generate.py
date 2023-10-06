from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from utils.prompter import Prompter
import torch
from peft import PeftModel
import torch
import pandas as pd
from tqdm import tqdm



single_instruction="""
The input for this task will consist of a series of products, each accompanied by its unique features. The structure of each product input is as follows:

<product> {product name}
<features> {list of features exclusive to the product}

Your mission is to expertly craft a captivating narrative of a high-end ship room, housing these highlighted products. It is essential to incorporate every single product mentioned in the input into your description. Even though it isn't compulsory to name all of the specific features, they should serve as guidelines for creating your narrative of how each product boosts the deluxe experience and convenience offered by the room.

Strive to produce an imagery-rich illustration of the guest experience that will pique the interest of prospective visitors browsing the ship's website. Give equal emphasis to all products and present them in a manner where their presence amplifies the enchantment of the room without creating any disruption. 

Remember that your role is not just to list products, it's about intricately integrating those products into the story of the room, making the reader comprehend why each product plays a key role in enhancing the room's charm.

Ensure that all products are equally considered and incorporated into the narrative seamlessly. The narrative should aim to evoke the desires of the readers, making the room’s allure practically irresistible.
"""


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


def generate_one_step(input):
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=800,
        do_sample=True,
        repetition_penalty=1.16
    )
    prompt = prompter.generate_prompt(input)
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
    return response
