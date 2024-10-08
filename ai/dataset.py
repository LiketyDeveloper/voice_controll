import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ai.nltk_utils import tokenize, stem, bag_of_words

from ai import DEVICE
from config import DATASET_FILE_PATH

class CommandDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()

        with open(DATASET_FILE_PATH, "r", encoding="utf-8") as file:
            self.commands_data = json.load(file)

        self.commands = []
        self.vocabulary = set()
        self.data = []

        for command_data in self.commands_data:
            command = command_data["command"]
            self.commands.append(command)

            for pattern in command_data["patterns"]:
                tokens = tokenize(pattern)
                self.vocabulary.update(tokens)
                self.data.append((command, tokens))

        ignore_words = [".", ","]
        self.vocabulary = [stem(word) for word in self.vocabulary if word not in ignore_words]
        self.vocabulary = sorted(self.vocabulary)

        self.x = []
        self.y = []

        for command, query in self.data:
            self.x.append(bag_of_words(query, self.vocabulary))
            self.y.append(self.commands.index(command))

        self.x = torch.tensor(self.x).to(DEVICE)
        self.y = torch.tensor(self.y).to(DEVICE)

        self.n_samples = len(self.x)
                
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples