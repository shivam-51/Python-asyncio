from common import call_openai_api, call_openai_api_async, plot_bar_chart, BLOCK_SIZE
import time
import asyncio
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()  # This line brings all environment variables from .env into os.environ


async def call_openai_async(user_prompt, idx):
    start_time = time.time()
    response = ""
    try:
        response = await call_openai_api_async(user_prompt, idx)
    except Exception as e:
        print(e)
    end_time = time.time()
    return response, start_time, end_time

async def process_block(df_block, start, block_size):
    responses = ""
    end = min(start+block_size, len(df_block))
    try:
        tasks = []
        for idx in range(start, end):
            user_prompt = df_block.loc[idx, "COMMENT"]
            # task = asyncio.create_task(call_openai(user_prompt))
            tasks.append(call_openai_async(user_prompt, idx))

        responses = await asyncio.gather(*tasks)
    except Exception as e:
        print(e)

    return responses

async def parallel_computations(df: pd.DataFrame):
    start = time.time()

    for i in range(0, len(df), BLOCK_SIZE):
        responses = await process_block(df, i, BLOCK_SIZE)
        for idx in range(i, min(i+BLOCK_SIZE, len(df))):
            df.loc[idx, "Response"] = responses[idx-i][0]

    end = time.time()
    df.to_csv("feedback_response_concurrent.csv", index=False)
    return end - start

df = pd.read_csv("feedback.csv")
asyncio.run(parallel_computations(df))
