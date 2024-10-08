import json

import torch 
import torch.nn as nn

from ai.nltk_utils import tokenize, stem, bag_of_words
from ai import DEVICE
from ai.dataset import CommandDataset

from config import MODEL_FILE_PATH

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
    
    def invoke(self, query):
        dataset = CommandDataset()
        
        sentence = tokenize(query)
        sentence = [stem(word) for word in sentence]
        
        print(sentence)
        sentence = bag_of_words(sentence, dataset.vocabulary)     
        
        sentence = torch.Tensor(sentence).to(DEVICE)  
        res = self(sentence)
        if all(res < 0):
            return "Непонятная команда"
        return dataset.commands[self(sentence).argmax()]
    

def load_model():
    data = torch.load(MODEL_FILE_PATH)
    
    input_size = data["input_size"]
    hidden_dize = data["hidden_size"]
    output_size = data["output_size"]
    
    model_state = data["model_state"]
    
    model = CommandIdentifier(input_size, hidden_dize, output_size)
    model.load_state_dict(model_state)
    model.eval()
    model.to(DEVICE)
    
    return model
    
