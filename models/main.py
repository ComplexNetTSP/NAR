def train(model, train_dataset, validation_dataset=None, optimizer=None, epochs=10, batch_size=5):
    x_loss_weight = 0.5
    h_loss_weight = 1 - x_loss_weight
    # for plotting save the losses
    train_losses = []
    val_losses = []
    y_losses = []
    y_val_losses = []
    for epoch in range(epochs):
        batch_count = len(train_dataset) // batch_size
        for i in range(batch_count):
            model.train()
            cumulated_loss = 0
            cumulated_y_loss = 0
            for j in range(i*batch_size, (i+1)*batch_size):
                graph = train_dataset[j] 
                y, loss, y_loss = model(graph)
                loss_x, loss_hints = loss[0], loss[1] # loss for the output
                
                y_loss_x, y_loss_hints = y_loss[0], y_loss[1] # loss for the hints

                cumulated_loss += x_loss_weight * loss_x + h_loss_weight * loss_hints
                cumulated_y_loss += x_loss_weight * y_loss_x + h_loss_weight * y_loss_hints
            
            cumulated_loss /= batch_size
            cumulated_y_loss /= batch_size

            optimizer.zero_grad()
            cumulated_loss.backward()
            optimizer.step()

        train_losses.append(cumulated_loss.item()) 
        y_losses.append(cumulated_y_loss.item())

        if validation_dataset:
            model.eval()
            cumulated_loss_val = 0
            cumulated_y_loss_val = 0
            with torch.no_grad():
                for k in range(len(validation_dataset)):
                    graph = validation_dataset[k]
                    y, loss, y_loss = model(graph)

                    loss_x, loss_hints = loss[0], loss[1]
                    y_loss_x, y_loss_hints = y_loss[0], y_loss[1]

                    cumulated_loss_val += x_loss_weight * loss_x + h_loss_weight * loss_hints
                    cumulated_y_loss_val += x_loss_weight * y_loss_x + h_loss_weight * y_loss_hints

            cumulated_loss_val /= len(validation_dataset)
            cumulated_y_loss_val /= len(validation_dataset)

            print(f'Epoch {epoch}, loss {cumulated_loss:.4f}, validation loss {cumulated_loss_val:.4f} || '
                  f'y_loss {cumulated_y_loss:.4f}, validation y_loss {cumulated_y_loss_val:.4f}')
            val_losses.append(cumulated_loss_val.item())
            y_val_losses.append(cumulated_y_loss_val.item())

        else:
            print(f'Epoch {epoch}, loss {cumulated_loss.item()}')
        
        # EARLY STOPPING
        if len(val_losses) > 5:
                    if val_losses[-1] > val_losses[-2] > val_losses[-3] > val_losses[-4] > val_losses[-5]:
                        print('Early stopping')
                        break
    return train_losses, val_losses, y_losses, y_val_losses




# MAIN ----------------------------------------------------------------------------------------------------
import sys

import torch
from torch.utils.data import random_split

from generator import *
from network import Network

import json

def main():
    # Load parameters from config file
    with open('config.json') as f:
        config = json.load(f)

    n = config.get('n', [20, 50])
    p = config.get('p', 0.3)
    size = config.get('size', 100)
    train_set_size = config.get('train_set_size', 0.8)
    hidden_dim = config.get('hidden_dim', 128)
    lr = config.get('lr', 0.001)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    
    print(f"Loaded parameters from config file: n: {n} p: {p} size: {size} train_set_size: {train_set_size} hidden_dim: {hidden_dim} lr: {lr} epochs: {epochs} batch_size: {batch_size}")

    root = './data'
    path = f'{root}/n={n}_p={p}'
    
    # Create a dataset of random graphs
    dataset = RandomGraphDataset(root=path, n=n, p=p, gen_num_graph=size, n=n, p=p)
    train_set_size_ = int(train_set_size * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_set_size_, len(dataset) - train_set_size_])
    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    # Create a model
    model = Network(latent_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Train the model
    train_loss, val_losses, y_losses, y_val_losses = train(model=model, train_dataset=train_dataset, validation_dataset=test_dataset, optimizer=optimizer(model.parameters(), lr=lr), epochs=epochs, batch_size=batch_size)
