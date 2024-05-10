#!/home/bizon/anaconda3/envs/coptic/bin/python
import os

os.environ["TRANSFORMERS_CACHE"] = (
    "/mnt/storage/maxenis/yoruba_english/chain-of-thought-demo/transformers_cache"
)

from transformers import pipeline, BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset

import requests
import torch

import datasets
import pandas as pd
import time
import datasets
from dotenv import load_dotenv

load_dotenv(override=True)

BATCH_SIZE = 8

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HUGGING_FACE_AUTH_HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

model_id = "meta-llama/Meta-Llama-3-70B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "use_auth_token": HUGGING_FACE_API_KEY,
    },
    device_map="balanced_low_0",
    batch_size=BATCH_SIZE,
)

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

base_no_cot = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: The answer is 8.
"""

base_cot = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
"""


def gen_response(pipe, prompt):
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    outputs = [
        output[0]["generated_text"][len(prompt[i]) :].split("\n")[0]
        for i, output in enumerate(outputs)
    ]
    return outputs


pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"

df = pd.read_csv("data_with_prompts.csv")
dataset = datasets.Dataset.from_pandas(df)
with open("llama_70B_outputs_cot.txt", "w+", buffering=1) as f:
    i = 0
    start = time.time()
    for output in pipe(
        KeyDataset(dataset, "cot"),
        batch_size=BATCH_SIZE,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        return_full_text=False,
    ):
        if i > BATCH_SIZE:
            break
        f.write(f"{time.time() - start}\n")
        f.write(f"{output}\n")
        i += 1

    # for i in range(0, len(df), BATCH_SIZE):
    #     if i > 1:
    #         break
    #     batch = df.iloc[i:i+BATCH_SIZE]
    #     no_cot_batch = batch["no_cot"].tolist()
    #     cot_batch = batch["cot"].tolist()
    #     current_time = time.time()
    #     no_cot_answers = gen_response(pipe, no_cot_batch)
    #     print(time.time() - current_time)
    #     current_time = time.time()
    #     cot_answers = gen_response(pipe, cot_batch)
    #     print(time.time() - current_time)
    #     for j, (no_cot_answer, cot_answer) in enumerate(zip(no_cot_answers, cot_answers)):
    #         print(i)
    #         f.write(f"{i+j}: {no_cot_answer}\n")
    #         f.write(f"{i+j}: {cot_answer}\n")
    #         f.write("\n\n")
