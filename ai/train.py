from loguru import logger
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======
from tqdm import tqdm
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai.model import CommandIdentifier
from ai.dataset import CommandDataset
from ai import DEVICE

from util import get_path, get_labels, id2label

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
    hidden_size = 16
    
<<<<<<< HEAD
    dataset = CommandDataset(for_training=True)
    model = CommandIdentifier(input_size=len(dataset.vocabulary), hidden_size=hidden_size, num_classes=len(dataset.commands))
=======
    dataset = CommandDataset()

    model = CommandIdentifier(input_size=len(dataset.vocabulary), hidden_size=hidden_size, num_classes=len(get_labels()))
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
    model.to(DEVICE)

    train_dataloader = DataLoader(dataset["train"], batch_size=8, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    accuracy_values = []
    loss_values = []
<<<<<<< HEAD
    for epoch in range(NN_TRAIN_EPOCHS):
        loss_val = 0
        accuracy_val = 0
        
        for inputs, labels in data_loader:
=======
    for epoch in (pbar := tqdm(range(NN_TRAIN_EPOCHS))):
        loss_val = 0
        accuracy_val = 0
        
        for inputs, labels in train_dataloader:
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item
            
            optimizer.step()
            
            acc_current = accuracy(outputs, labels)
            accuracy_val += acc_current
            
<<<<<<< HEAD
        accuracy_values.append(accuracy_val/len(data_loader))
        loss_values.append(loss_val/len(data_loader))
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1} >> Average loss: {loss_val/len(data_loader):.4f}, Average accuracy: {accuracy_val/len(data_loader):.4f}")
            print(f"")

=======
        accuracy_values.append(accuracy_val/len(train_dataloader))
        loss_values.append(loss_val/len(train_dataloader))
            
        pbar.set_description(f"Epoch {epoch+1} >> Average loss: {loss_val/len(train_dataloader):.4f}, Average accuracy: {accuracy_val/len(train_dataloader):.4f}")
        
        
    test_dataloader = DataLoader(dataset["test"], batch_size=8, shuffle=True)
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    
    accuracy_model = total_correct / len(dataset["train"])
    logger.success(f"Final model accuracy on test dataset: {accuracy_model:.4f}")
    
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
    
    data = {
        "model_state": model.state_dict(),
        "input_size": len(dataset.vocabulary),
        "hidden_size": hidden_size,
<<<<<<< HEAD
        "output_size": len(dataset.commands),
=======
        "output_size": len(get_labels())
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
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
    