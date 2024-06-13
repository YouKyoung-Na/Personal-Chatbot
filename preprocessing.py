# 전처리 방법론 코드
# 코드 조각들만 일단 모아둠

import json
import pandas as pd
import numpy as np
import re
import seaborn as sns
import tqdm


# 정규표현식
def cleansing2(data):
    data = re.sub(r'ㅋ+', '', data)      # "ㅋ"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'[ㅜㅠ]+', '', data)  # "ㅜ" 또는 "ㅠ"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'ㅎ+', '', data)     # "ㅎ"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'ㄱㄱ+', '', data)          # "ㄱㄱ"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'ㄲㄲ+', '', data)          # "ㄲㄲ"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'ㅅㅅ+', '', data)          # "ㅅㅅ"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'ㅁㅁ+', '', data)          # "ㅁㅁ"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'@@+', '', data)          # "@@"로 시작하는 모든 연속된 문자 시퀀스 제거
    data = re.sub(r'이모티콘', '', data)    #  "#@이모티콘#" 제거
    data = re.sub(r'사진', '', data)     # "#@시스템#" 제거
    data = re.sub(r'^(<ENTER>)', '', data)
    data = re.sub(r'(<ENTER>)+', '<ENTER>', data)
    data = re.sub(r'#@기타#', '', data)       # "#@기타#" 제거
    data = re.sub(r'#@URL#', '', data)       # "#@URL#" 제거
    data = re.sub(r'#검색#', '', data)       # "#검색#" 제거
    return data

path = "./dataset/한국어SNS/Train/개인및관계.json"
data = pd.read_json(path)

# 대화 내용만 추출하기
chat_logs = []

for idx in range(len(data)):
    chat = [(conversation["participantID"], conversation["utterance"]) for conversation in data.iloc[idx].data["body"]]
    chat_logs.append(chat)

prev_presenter = None

# 대화 합치기
merged_conversations = []
for chats in tqdm.tqdm(chat_logs):
    merged_conversation = []
    merged_chat = ""
    
    for chat in chats:    
        p, conversation = chat

        if prev_presenter == p:
            merged_chat += conversation + "<ENTER>"

        else:
            merged_conversation.append(merged_chat)
            merged_chat = conversation + "<ENTER>"
            prev_presenter = p

    merged_conversations.append(merged_conversation[1:])

# 대화 sliding window
QA_set = []

for merged_conversation in tqdm.tqdm(merged_conversations):
    for Q, A in zip(merged_conversation, merged_conversation[1:]):
        QA_set.append([Q, A])

print(" === === === END === === ===")
df = pd.DataFrame(QA_set, columns=["Q", "A"])

# df의 마지막 <ENTER>은 제거
adv_QA_set = []

for i in tqdm.tqdm(range(len(df))):
    question = df.iloc[i].Q[:-7]
    answer = df.iloc[i].A[:-7]

    adv_QA_set.append([question, answer])

print("END")

adv_df = pd.DataFrame(adv_QA_set, columns=["Q", "A"])
adv_df.to_csv("./adv_conversation.csv")