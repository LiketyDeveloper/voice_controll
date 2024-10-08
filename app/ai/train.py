import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ai.nltk_utils import tokenize, stem, bag_of_words


DATASET_PATH = "dataset.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CommandIdentifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.activation(out)
        out = self.l2(out)
        out = self.activation(out)
        out = self.l3(out)
        
        return out


class CommandDataset(Dataset):
    
    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        with open(dataset_path, "r", encoding="utf-8") as file:
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
    

def train():
    dataset = CommandDataset(dataset_path=DATASET_PATH)
    
    for record in dataset:
        print(record, "\n\n-----\n\n")