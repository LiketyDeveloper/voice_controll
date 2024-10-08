from loguru import logger

import torch 
import torch.nn as nn

from ai.nltk_utils import tokenize, stem, bag_of_words
from ai import DEVICE
from ai import CommandDataset

from config import MODEL_FILE_PATH

class CommandIdentifier(nn.Module):
    """Neural Network class to identify commands from text input"""

    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()
        self._dataset = CommandDataset()
        
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
        
        sentence = bag_of_words(sentence, dataset.vocabulary)     
        
        sentence = torch.Tensor(sentence).to(DEVICE)  
        res = self(sentence)
        
        logger.debug("\n", "\n".join([
            f"{command}{"\t" if len(command) > 4 else "\t\t"}>>\t{probability.item()}" for command, probability in zip(dataset.commands, res)]))

        # if all(res < 0):
        #     return "Непонятная команда"
        
        return dataset.commands[self(sentence).argmax()]
    

def load_model() -> CommandIdentifier:
    """Load command identifier model from file"""

    data = torch.load(MODEL_FILE_PATH, weights_only=True)

    model = CommandIdentifier(
        input_size=data["input_size"],
        hidden_size=data["hidden_size"],
        num_classes=data["output_size"]
    )

    model.load_state_dict(data["model_state"])
    model.eval()
    model.to(DEVICE)

    return model

    
