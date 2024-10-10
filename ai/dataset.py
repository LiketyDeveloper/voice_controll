import json
from loguru import logger

import torch
from torch.utils.data import Dataset
import pymorphy3


from ai.nltk_utils import tokenize, stem, bag_of_words, get_stopwords, is_number
from ai import DEVICE

from config.nn import DATASET_FILE_PATH, VOCABULARY_FILE_PATH


def preprocess(command, pattern):
    tokens = tokenize(pattern)
    morph = pymorphy3.MorphAnalyzer()
    stop_words = get_stopwords()
    stop_words.extend([".", ","])
    
    tokens = [word for word in tokens 
              if not is_number(word) 
              and morph.parse(word)[0].score > 0.4
              and word not in stop_words
    ]
    
    for_vocab = [stem(token) for token in tokens]
    for_train_data = [(command, [stem(token) for token in tokens])]
    
    word_forms = ('gent', 'datv', 'accs', 'ablt', 'loct')
    for form in word_forms:
        
        form_tokens = []
        for word in tokens:
            new_word = morph.parse(word)[0].inflect({form})
            if not new_word:
                break
            form_tokens.append(new_word.word)
            for_vocab.append(stem(new_word.word))
        
        if form_tokens:
            for_train_data.append((command, [stem(token) for token in form_tokens]))
            

    return for_vocab, for_train_data



class CommandDataset(Dataset):
    
    __instance = None
    __init_called = False
    
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(CommandDataset, cls).__new__(cls)
        return cls.__instance
    
    
    def __init__(self, for_training=False) -> None:
        if not CommandDataset.__init_called:
            super().__init__()
            
            if for_training:

                with open(DATASET_FILE_PATH, "r", encoding="utf-8") as file:
                    self.commands_data = json.load(file)
                    

                self.commands = []
                self.vocabulary = []
                self.data = []

                for command_data in self.commands_data:
                    command = command_data["command"]
                    self.commands.append(command)
                    
                    for pattern in command_data["patterns"]:
                        for_vocab, for_train_data = preprocess(command, pattern)
                        
                        self.vocabulary.extend(for_vocab)
                        self.data.extend(for_train_data)
                        
                self.vocabulary = sorted(set(self.vocabulary))

                new_data = []
                for record in self.data:
                    if record not in new_data:
                        new_data.append(record)
                        
                self.data = new_data
                            
                with open(VOCABULARY_FILE_PATH, "w", encoding="utf8") as file:
                    data = {
                        "data_lenght": len(self.data), 
                        "commands": self.commands,
                        "vocabulary": self.vocabulary
                    }
                    json.dump(data, file, indent=4, ensure_ascii=False)
                
                self.x = []
                self.y = []
                
                for command, query in self.data:
                    self.x.append(bag_of_words(query, self.vocabulary))
                    self.y.append(self.commands.index(command))
                    
                self.x = torch.tensor(self.x).to(DEVICE)
                self.y = torch.tensor(self.y).to(DEVICE)
                print(self.x.shape)
                print(self.y[0])
                
                self.n_samples = len(self.data)
            else:
                with open(VOCABULARY_FILE_PATH, "r", encoding="utf8") as file:
                    data = json.load(file)
                    
                self.n_samples = data["data_lenght"]
                self.commands = data["commands"]
                self.vocabulary = data["vocabulary"]
            
            logger.debug("Dataset initialized")
            CommandDataset.__init_called = True
        else:
            self = CommandDataset.__instance

    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    
    def __len__(self):
        return self.n_samples