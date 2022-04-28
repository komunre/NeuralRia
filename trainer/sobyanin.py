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
from os.path import exists

dataloader = DataLoader(WordsDataset(wordgen2.data, 2), batch_size=wordgen2.batch_size)

print("Vocab size: " + str(len(dataloader.dataset)))

model = wordgen2.NeuralNewsGen(dataloader).to(device=wordgen2.device)
if exists("model_twolen.pth"):
    model.load_state_dict(torch.load("model_twolen.pth"))
else:
    wordgen2.train(dataloader, model, wordgen2.loss_fn, wordgen2.optimizer, 12, 2)
    torch.save(model.state_dict(), "model_twolen.pth")

model.eval()

while True:
    announce = ' '.join(wordgen2.predict(wordgen2.train_dataloader, model, "Сергей Собянин", 7, 2))
    print(announce)
    token = open("secrets/token.txt").read()
    r = requests.post('http://informasoft.ru/NeuralNews/Create', data={'token': token, 'news': announce.encode(encoding='UTF-8',errors='strict')})
    if r.status_code != 220:
        print("Something is wrong..." + str(r.status_code))
    sleep(60)