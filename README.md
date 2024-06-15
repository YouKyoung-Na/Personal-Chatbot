# Personal-Chatbot
**2024년도 1학기 자연어처리**    
해당 레포지토리는 KoGPT2 모델을 이용하여 개인의 스타일을 반영한 개인화 챗봇입니다.

## Contributor
문성수(dalcw@jnu.ac.kr), 나유경(me6zero@jnu.ac.kr)

## Installation
먼저 본인 컴퓨터 버전에 맞는 [Pytorch](https://pytorch.org/)를 설치하고, 아래의 명령어를 실행합니다.
```
pip install -r requirements.txt
```

## Directory Structure
```bash
├── dataset/
│   └── README.md
├── examples/
│   ├── KoGPTbased_chatbot.ipynb
│   └── evaluation.ipynb
├── model_weight/
│   ├── train/
│   ├── README.md
│   ├── backbone.pt
│   └── personal_finetuning.pt
├── README.md
├── finetuning_datagen.py
├── inference.py
├── main.py
├── preprocessing.py
├── requirements.txt
├── setting.json
└── training.py
``` 

## Download Weight File
아래의 링크 바로가기를 누르면, 모델의 가중치를 다운받을 수 있습니다.   
위의 디렉토리 구조처럼 추가하면 됩니다.   
[링크 바로가기](https://drive.google.com/drive/folders/1d_IKEIy4X45HSN7i3n_z3l9LjMisAht7)

## Discord Server
디스코드 서버를 실행하는 명령어는 다음과 같습니다.   
모델의 가중치는 위의 디렉토리 구조처럼 model_weight에 저장되어 있어야 합니다.
```
python3 main.py --weight [model weight]
```

## Model Training
새로운 모델을 학습시키기 위한 코드는 다음과 같습니다.   
모델의 학습 가중치는 model weight/train에 저장됩니다.  
```
python3 training.py --traindata [data path] --pretrain [pretrain model] --epoch [epoch]
```

*preprocessor.py은 리팩토링 작업 없이 code-dummi 형태로 임시 푸시함*
