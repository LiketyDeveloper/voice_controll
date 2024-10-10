from loguru import logger
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai.model import CommandIdentifier
from ai.dataset import CommandDataset
from ai import DEVICE
from tqdm import tqdm

from util import get_path

from config.nn import MODEL_FILE_PATH, NN_TRAIN_EPOCHS


def accuracy(preds, labels):
    correct = 0
    for pred, label in zip(preds, labels):
        if pred.argmax() == label:
            correct += 1
    
    return correct / len(preds)


def train_model() -> None:
    """
    Train the neural network model for command identification.

    This function will create a model, create a DataLoader from the dataset, train the model using the DataLoader, and then save the model to a file.
    """
    dataset = CommandDataset()
    model = CommandIdentifier(input_size=len(dataset.vocabulary), hidden_size=8, num_classes=len(dataset.commands))
    model.to(DEVICE)

    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    accuracy_values = []
    loss_values = []
    for epoch in range(NN_TRAIN_EPOCHS):
        loss_val = 0
        accuracy_val = 0
        
        for inputs, labels in data_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item
            
            optimizer.step()
            
            acc_current = accuracy(outputs, labels)
            accuracy_val += acc_current
            
        accuracy_values.append(accuracy_val/len(data_loader))
        loss_values.append(loss_val/len(data_loader))
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Average loss: {loss_val/len(data_loader):.4f}")
            print(f"Accuracy: {accuracy_val/len(data_loader):.4f}")

    
    data = {
        "model_state": model.state_dict(),
        "input_size": len(dataset.vocabulary),
        "hidden_size": 8,
        "output_size": len(dataset.commands),
    }
    torch.save(data, MODEL_FILE_PATH)
    logger.success("NN successfully trained")
    
    plt.plot([i for i in range(NN_TRAIN_EPOCHS)], accuracy_values)
    
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title('Точность классификатора команд')
    
    plt.savefig(get_path("metrics", "accuracy.png"))
    plt.close()
    
    logger.success("Accuracy image saved")
    
    plt.plot([i for i in range(NN_TRAIN_EPOCHS)], loss_values)
    
    plt.xlabel('Эпоха')
    plt.ylabel('Процент потерь')
    plt.title('Потери классификатора команд')
    
    plt.savefig(get_path("metrics", "loss.png"))
    plt.close()
    logger.success("Loss image saved")
    