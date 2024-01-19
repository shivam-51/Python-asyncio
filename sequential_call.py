from common import call_openai_api_async, call_openai_api
import time
# import asyncio
import pandas as pd

def sequential_computations(df: pd.DataFrame):
    start = time.time()
    for i in range(0, len(df)):
        try:
            df.loc[i, "Response"] = call_openai_api(df.loc[i, "COMMENT"], i)
        except Exception as e:
            print(e)

    end = time.time()
    df.to_csv("feedback_response_sequential.csv", index=False)
    print(f"Time taken: {end - start}")


df = pd.read_csv("feedback.csv")
# loop = asyncio.get_event_loop()

# asyncio.run()
sequential_computations(df)
