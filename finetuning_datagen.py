import numpy as np
import pandas as pd
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, help="data data")
args = parser.parse_args()

person1 = None
person2 = None

with open(args.datapath, "r", encoding="UTF8") as f:
    while True:
        line = f.readline()
        
        if line[0] == "[":
            if not person1:
                person1 = line.split()[0]
            
            if person1 and not person2:
                if person1 != line.split()[0]:
                    person2 = line.split()[0]

        if person1 and person2: break
    f.close()

print(f"[+] person 1 is {person1}")
print(f"[+] person 2 is {person2}")

conversation = []

prev_presenter = None

with open(args.datapath, "r", encoding="UTF8") as f:
    for line in f.readlines():
        try:
            if line[0] == "[":
                if line.split()[0] == person1:
                    if prev_presenter == person1:
                        conversation[-1] += ("<ENTER>" + " ".join(line.split()[3:]))
                    else:
                        conversation.append(" ".join(line.split()[3:]))

                    prev_presenter = person1

                elif line.split()[0] == person2:
                    if prev_presenter == person2:
                        conversation[-1] += ("<ENTER>" + " ".join(line.split()[3:]))
                    else:
                        conversation.append(" ".join(line.split()[3:]))

                    prev_presenter = person2
        except:
            pass
    f.close()

QA_set = []

for Q, A in zip(conversation, conversation[1:]):
    QA_set.append([Q, A])

set1 = QA_set[0::2]
set2 = QA_set[1::2]

set1_df = pd.DataFrame(set1, columns=["Q", "A"])
set2_df = pd.DataFrame(set2, columns=["Q", "A"])

set1_df.to_csv(f"./dataset/finetuning_{person1}.csv")
set2_df.to_csv(f"./dataset/finetuning_{person2}.csv")