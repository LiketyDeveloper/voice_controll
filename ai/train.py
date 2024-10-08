from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai.model import CommandIdentifier
from ai.dataset import CommandDataset
from ai import DEVICE

from config import  MODEL_FILE_PATH

    

def train():
    dataset = CommandDataset()
    
    model = CommandIdentifier(input_size=len(dataset.vocabulary), hidden_size=8, num_classes=len(dataset.commands))
    model = model.to(DEVICE)
    
    BATCH_SIZE = 8
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1000):
        for index, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            output = model(x)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} loss: {loss.item():.4f}")
            
    data = {
        "model_state": model.state_dict(),
        "input_size": len(dataset.vocabulary),
        "hidden_size": 8,
        "output_size": len(dataset.commands),
    }
    torch.save(data, MODEL_FILE_PATH)
    logger.success("NN successfully trained")
    
    