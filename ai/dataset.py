import json
from loguru import logger

import torch
from torch.utils.data import Dataset

from ai.nltk_utils import tokenize, stem, bag_of_words, get_stopwords, filter_numbers
from ai import DEVICE

from config import DATASET_FILE_PATH

class CommandDataset(Dataset):
    
    __instance = None
    __init_called = False
    
    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super(CommandDataset, cls).__new__(cls)
        return cls.__instance
    
    def __init__(self) -> None:
        if not CommandDataset.__init_called:
            super().__init__()

            with open(DATASET_FILE_PATH, "r", encoding="utf-8") as file:
                self.commands_data = json.load(file)

            self.commands = []
            self.vocabulary = []
            self.data = []

            for command_data in self.commands_data:
                command = command_data["command"]
                self.commands.append(command)

                for pattern in command_data["patterns"]:
                    tokens = tokenize(pattern)
                    self.vocabulary.extend(tokens)
                    self.data.append((command, tokens))

            stop_words = get_stopwords()
            stop_words.extend([".", ","])
            self.vocabulary = [stem(word) for word in self.vocabulary if word not in stop_words]
            self.vocabulary = filter_numbers(set(self.vocabulary))
            self.vocabulary = sorted(self.vocabulary)
                    
            self.x = []
            self.y = []
            
            for command, query in self.data:
                self.x.append(bag_of_words(query, self.vocabulary))
                self.y.append(self.commands.index(command))
                
            self.x = torch.tensor(self.x).to(DEVICE)
            self.y = torch.tensor(self.y).to(DEVICE)
            
            self.n_samples = len(self.data)
            
            logger.debug("Dataset initialized")
            CommandDataset.__init_called = True
        else:
            self = CommandDataset.__instance

    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    
    def __len__(self):
        return self.n_samples