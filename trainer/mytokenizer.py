import codecs
import os
from random import randint
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def indexToFloat(index, max):
    if (index == 0): return 0
    return (float)(index / max)

def floatToIndex(float, max):
    index = (int)(float*max)
    return index

class WordsDataset(Dataset):
    def __init__(self, texts, seqlen):
        self.text_arr = texts

        self.sequence_length = seqlen
        self.create_per_word()
    
    def create_per_word(self):
        self.words = ['']
        self.indices_str = [0]
        for text in self.text_arr:
            split = text.split(" ")

            for word in split:
                if word in self.words: 
                    self.indices_str.append(self.words.index(word))

                word = self.cleanup_word(word)

                self.words.append(word)
                self.indices_str.append(len(self.words) - 1)
        
        self.dataset = self.words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        value = torch.tensor(self.indices_str[index:index+self.sequence_length], device="cuda")
        
        train = torch.tensor(self.indices_str[index+1:index+self.sequence_length+1], device="cuda")

        return value, train

    def cleanup_word(self, word):
        word = word.replace(",", "")
        word = word.replace("!", "")
        word = word.replace(".", "")
        word = word.replace(";", "")
        word = word.replace("?", "")
        word = word.replace("\n", "")
        return word

    def stringToIndices(self, str):
        words = []
        split = str.split(" ")

        for word in split:
            word = self.cleanup_word(word)
            words.append(word)

        if (len(words) < 5):
            words.append('')

        fivewords = words[len(words) - 5::]
        
        for word in fivewords:
            if not word in self.words:
                exit("Ouch. No such word as " + word)
        
        value = np.array([ 
            [indexToFloat(self.words.index(fivewords[0]), len(self.words))],
            [indexToFloat(self.words.index(fivewords[1]), len(self.words))],
            [indexToFloat(self.words.index(fivewords[2]), len(self.words))],
            [indexToFloat(self.words.index(fivewords[3]), len(self.words))],
            [indexToFloat(self.words.index(fivewords[4]), len(self.words))],
        ], dtype=np.float32)

        return (torch.from_numpy(value), torch.from_numpy(value))

    def getByIndex(self, index):
        str = ""
        for i in index[0]:
            str += self.words[floatToIndex(i, len(self.words))] + " "
        return str

    def get_word_by_index(self, index):
        return self.words[index]

    def get_word_index(self, word):
        return self.indices_str[self.words.index(word)]

def tokenize(paths, separator):
    result = []
    for filename in paths:
        f = codecs.open(filename, "r", "utf-8")
        text = f.read()
        arr = text.split(separator)
        for e in arr:
            result.append(e)

    return result