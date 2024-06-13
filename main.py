import discord
from inference import Inference
import time
from collections import deque
import asyncio
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=str, help="model weight")
args = parser.parse_args()

with open("./setting.json") as f:
    setting = json.load(f)
TOKEN = setting["discord"]["TOKEN"]
CHANNEL_ID = setting["discord"]["CHANNELID"]

inference = Inference()
if not args.weight:
    inference.model_load("./model_weight/personal_finetuning.pt")
else: 
    inference.model_load(f"./model_weight/{args.weight}")

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = deque()
        self.message_check_intervbal = 10
        self.response_delay = 5
        self.last_message_time = None

    async def on_ready(self):
        await self.change_presence(status=discord.Status.online, activity=discord.Game("대기중"))
        self.loop.create_task(self.check_messages())

    async def check_messages(self):
        while True:
            await asyncio.sleep(self.message_check_intervbal)
            current_time = time.time()
            if self.last_message_time and (current_time - self.last_message_time) >= self.response_delay:
                if self.messages:
                    combined_message = "<ENTER>".join(self.messages)
                    answer = inference.inference(combined_message)
                    for splited_ans in answer.split("<ENTER>"):
                        if not splited_ans: continue 
                        await self.last_chanel.send(splited_ans)
                    self.messages.clear()
     
    async def on_message(self, message):

        if message.author == self.user:
            return
        
        if message.content == "ping":
            await message.channel.send(f"pong {message.author.mention}")
        
        else:
            self.messages.append(message.content)
            self.last_message_time = time.time()
            self.last_chanel = message.channel

intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(TOKEN)