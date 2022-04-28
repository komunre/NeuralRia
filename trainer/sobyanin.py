from itertools import count
from time import sleep
import wordgen2
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from mytokenizer import WordsDataset
import mytokenizer
import requests


model = wordgen2.NeuralNewsGen(wordgen2.train_dataloader).to(device=wordgen2.device)
model.load_state_dict(torch.load("model.pth"))

train_dataloader = wordgen2.train_dataloader

print("Vocab size: " + str(len(train_dataloader.dataset)))

model.eval()

while True:
    announce = ' '.join(wordgen2.predict(wordgen2.train_dataloader, model, "Собянин заявил об огромном", 3))
    print(announce)
    token = open("secrets/token.txt").read()
    r = requests.post('http://informasoft.ru/NeuralNews/Create', data={'token': token, 'news': announce.encode(encoding='UTF-8',errors='strict')})
    if r.status_code != 220:
        print("Something is wrong..." + str(r.status_code))
    sleep(60)