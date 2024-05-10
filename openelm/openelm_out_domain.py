#!/home/bizon/anaconda3/envs/coptic/bin/python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import torch
from dotenv import load_dotenv
import datasets
import pandas as pd
import datasets
import time



load_dotenv(override=True)

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

model = AutoModelForCausalLM.from_pretrained(
    "apple/OpenELM-3B", trust_remote_code=True
).to("cuda")


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", trust_remote_code=True
)

tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def gen_response(prompts):
    tokenized_prompt = tokenizer(
        prompts,
        padding=True,
    )
    tokenized_prompt = torch.tensor(tokenized_prompt["input_ids"], device=0)
    tokenized_prompt = tokenized_prompt.to("cuda")
    output_ids = model.generate(
        tokenized_prompt,
        max_length=1024,
        pad_token_id=0,
    )
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    output_text = [
        output[len(prompts[i]):].split("\n")[0] for i, output in enumerate(output_text)
    ]
    return output_text


BATCH_SIZE = 32

df = pd.read_csv("out_of_domain_names_with_prompts.csv")
df = df[:1000]
with open("openelm_3B_out_of_domain_names.txt", "w+", buffering=1) as f:
    for i in range(0, len(df), BATCH_SIZE):
        print("Running batch", i)
        current_time = time.time()
        batch = df.iloc[i : i + BATCH_SIZE]
        no_cot_batch = batch["no_cot"].tolist()
        cot_batch = batch["cot"].tolist()
        no_cot_answers = gen_response(no_cot_batch)
        cot_answers = gen_response(cot_batch)
        for j, (no_cot_answer, cot_answer) in enumerate(
            zip(no_cot_answers, cot_answers)
        ):
            f.write(f"{i+j}: {no_cot_answer}\n")
            f.write(f"{i+j}: {cot_answer}\n")
            f.write("\n\n")
        print("Finished batch with time: ", i, time.time() - current_time)
