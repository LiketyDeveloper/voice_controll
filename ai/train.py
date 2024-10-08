from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai.model import CommandIdentifier
from ai import CommandDataset
from ai import DEVICE

from config import  MODEL_FILE_PATH
    

def train_model() -> None:
    """
    Train the neural network model for command identification.

    This function will create a model, create a DataLoader from the dataset, train the model using the DataLoader, and then save the model to a file.
    """
    dataset = CommandDataset()
    model = CommandIdentifier(input_size=len(dataset.vocabulary), hidden_size=8, num_classes=len(dataset.commands))
    model.to(DEVICE)

    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    data = {
        "model_state": model.state_dict(),
        "input_size": len(dataset.vocabulary),
        "hidden_size": 8,
        "output_size": len(dataset.commands),
    }
    torch.save(data, MODEL_FILE_PATH)
    logger.success("NN successfully trained")
    
    