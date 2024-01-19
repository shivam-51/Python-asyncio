from openai import AsyncAzureOpenAI, AzureOpenAI
import time
import asyncio
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os

load_dotenv()  # This line brings all environment variables from .env into os.environ

asyncClient = AsyncAzureOpenAI()
client = AzureOpenAI()

SYSTEM_PROMPT="""Identify themes and sentiment from review"""

# BLOCK_SIZE denotes the number of rows to be processed concurrently.
BLOCK_SIZE = 10

async def call_openai_api_async(user_prompt, idx):
    print("Starting ", idx)
    response = await asyncClient.chat.completions.create(
        model="gpt-35", # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT },
            {"role": "user", "content": user_prompt }
        ]
    )
    print("Ending ", idx)
    return response.choices[0].message.content


def call_openai_api(user_prompt, idx):
    print("Starting ", idx)
    response = client.chat.completions.create(
        model="gpt-35", # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT },
            {"role": "user", "content": user_prompt }
        ]
    )
    print("Ending ", idx)
    return response.choices[0].message.content

def plot_bar_chart():
    dict = {10: 23.58059287071228, 20: 6.149170875549316, 30: 5.537899017333984, 40: 13.432697057723999, 50: 6.4363768100738525, 60: 2.8452088832855225, 70: 4.165992021560669, 80: 10.412950038909912, 90: 7.203731060028076, 100: 6.975428819656372, 110: 7.5954270362854, 120: 12.27223801612854, 130: 12.983277082443237, 140: 9.043008804321289, 150: 6.535268068313599, 160: 10.68006682395935, 170: 9.475073099136353, 180: 9.157772064208984, 190: 10.040534973144531, 200: 10.699626922607422, 210: 10.08786916732788, 220: 11.060771942138672, 230: 11.225385904312134, 240: 12.682886838912964, 250: 12.655791997909546, 260: 12.691540956497192, 270: 13.029635190963745, 280: 12.3940110206604, 290: 10.272783041000366, 300: 10.345827102661133, 310: 13.338971853256226, 320: 14.507338047027588, 330: 13.823668956756592, 340: 16.462684869766235, 350: 15.685950994491577, 360: 11.542593002319336, 370: 17.991539001464844, 380: 14.815891981124878, 390: 13.455880880355835, 400: 15.1792311668396, 410: 11.09329891204834, 420: 32.02232813835144, 430: 20.999429941177368, 440: 18.793107986450195, 450: 14.154467105865479, 460: 11.83663272857666, 470: 13.771175146102905, 480: 12.57044005393982, 490: 10.368336915969849}
    dict[1] = 322
    plt.xlabel("Batch Size")
    plt.ylabel("Time Taken (s)")
    plt.bar(dict.keys(), dict.values())
    plt.show()

def plot_time_chart():
    list = [{'start_time': 1705219070.593516, 'end_time': 1705219071.971154}, {'start_time': 1705219070.63379, 'end_time': 1705219071.973603}, {'start_time': 1705219070.635401, 'end_time': 1705219072.046541}, {'start_time': 1705219070.63663, 'end_time': 1705219072.390647}, {'start_time': 1705219070.638122, 'end_time': 1705219072.977778}, {'start_time': 1705219070.639322, 'end_time': 1705219071.9410748}, {'start_time': 1705219070.640554, 'end_time': 1705219071.985274}, {'start_time': 1705219070.641685, 'end_time': 1705219072.726229}, {'start_time': 1705219070.64288, 'end_time': 1705219072.180782}, {'start_time': 1705219070.644038, 'end_time': 1705219072.060489}, {'start_time': 1705219072.981099, 'end_time': 1705219073.4316978}, {'start_time': 1705219072.988919, 'end_time': 1705219074.0714948}, {'start_time': 1705219072.993692, 'end_time': 1705219073.419997}, {'start_time': 1705219072.996257, 'end_time': 1705219073.635256}, {'start_time': 1705219072.998333, 'end_time': 1705219073.560251}, {'start_time': 1705219073.000355, 'end_time': 1705219073.429588}, {'start_time': 1705219073.002064, 'end_time': 1705219073.621378}, {'start_time': 1705219073.0045922, 'end_time': 1705219073.787584}, {'start_time': 1705219073.006243, 'end_time': 1705219073.5255702}, {'start_time': 1705219073.0076978, 'end_time': 1705219074.069603}, {'start_time': 1705219074.071899, 'end_time': 1705219074.5335522}, {'start_time': 1705219074.073768, 'end_time': 1705219074.537293}, {'start_time': 1705219074.075179, 'end_time': 1705219074.729894}, {'start_time': 1705219074.0767941, 'end_time': 1705219074.5594418}, {'start_time': 1705219074.078203, 'end_time': 1705219074.510176}, {'start_time': 1705219074.0799181, 'end_time': 1705219074.543445}, {'start_time': 1705219074.081335, 'end_time': 1705219075.4108522}, {'start_time': 1705219074.083034, 'end_time': 1705219074.65802}, {'start_time': 1705219074.084394, 'end_time': 1705219074.552632}, {'start_time': 1705219074.0857222, 'end_time': 1705219074.508327}]

    start_times = [entry["start_time"] for entry in list]
    end_times = [entry["end_time"] for entry in list]

    fig, ax = plt.subplots()
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        ax.plot([start, end], [i, i], marker='o', label=f'Task {i + 1}')

    ax.set_yticks(range(len(list)))
    ax.set_yticklabels([f'Task {i + 1}' for i in range(len(list))])
    ax.set_xlabel('Time')
    ax.set_title('Start and End Times of Tasks')
    ax.legend()
    plt.show()


# plot_bar_chart()
# plot_time_chart()