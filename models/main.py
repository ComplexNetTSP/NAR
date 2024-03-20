import torch

def train(model, train_dataset, validation_dataset=None, optimizer=None, epochs=10, batch_size=5):
    x_loss_weight = 0.5
    h_loss_weight = 1 - x_loss_weight

    loss_edges_train, loss_reach_train, loss_parents_train = [], [], []
    loss_edges_val, loss_reach_val, loss_parents_val = [], [], []

    for epoch in range(epochs):
        batch_count = len(train_dataset) // batch_size
        
        cumulated_loss_edges_epoch, cumulated_loss_reach_epoch, cumulated_loss_parents_epoch = 0, 0, 0

        for i in range(batch_count):
            model.train()
            cumulated_loss_edges, cumulated_loss_reach, cumulated_loss_parents = 0, 0, 0
            for j in range(i*batch_size, (i+1)*batch_size):
                graph = train_dataset[j] 
                loss_edges, loss_reach, loss_parents = model(graph)
                loss_edges_output, loss_edges_hints = loss_edges[0], loss_edges[1] # loss for the edges

                cumulated_loss_edges += x_loss_weight * loss_edges_output + h_loss_weight * loss_edges_hints
                cumulated_loss_reach += loss_reach
                cumulated_loss_parents += loss_parents

            cumulated_loss_edges /= batch_size
            cumulated_loss_reach /= batch_size
            cumulated_loss_parents /= batch_size
            
            optimizer.zero_grad()
            cumulated_loss_edges.backward()
            optimizer.step()

            cumulated_loss_edges_epoch += cumulated_loss_edges
            cumulated_loss_reach_epoch += cumulated_loss_reach
            cumulated_loss_parents_epoch += cumulated_loss_parents

        # Convert tensors to lists and append to the respective lists
        loss_edges_train.append(cumulated_loss_edges_epoch.item() / batch_count)
        loss_reach_train.append(cumulated_loss_reach_epoch.item() / batch_count)
        loss_parents_train.append(cumulated_loss_parents_epoch.item() / batch_count)

        if validation_dataset:
            model.eval()
            with torch.no_grad():
                cumulated_loss_edges_val, cumulated_loss_reach_val, cumulated_loss_parents_val = 0, 0, 0
                for k in range(len(validation_dataset)):
                    graph = validation_dataset[k]
                    loss_edges, loss_reach, loss_parents = model(graph)
                    loss_edges_output, loss_edges_hints = loss_edges[0], loss_edges[1] # loss for the edges

                    cumulated_loss_edges_val += x_loss_weight * loss_edges_output + h_loss_weight * loss_edges_hints
                    cumulated_loss_reach_val += loss_reach
                    cumulated_loss_parents_val += loss_parents 

                cumulated_loss_edges_val /= len(validation_dataset)
                cumulated_loss_reach_val /= len(validation_dataset)
                cumulated_loss_parents_val /= len(validation_dataset)

                loss_edges_val.append(cumulated_loss_edges_val.item())
                loss_reach_val.append(cumulated_loss_reach_val.item())
                loss_parents_val.append(cumulated_loss_parents_val.item())

                print(f'Epoch {epoch}, loss_edges {cumulated_loss_edges_epoch.item() / batch_count}, loss_reach {cumulated_loss_reach_epoch.item() / batch_count}, loss_parents {cumulated_loss_parents_epoch.item() / batch_count}, loss_edges_val {cumulated_loss_edges_val.item()}, loss_reach_val {cumulated_loss_reach_val.item()}, loss_parents_val {cumulated_loss_parents_val.item()}')
        
        else:
            print(f'Epoch {epoch}, loss_edges {cumulated_loss_edges_epoch.item() / batch_count}, loss_reach {cumulated_loss_reach_epoch.item() / batch_count}, loss_parents {cumulated_loss_parents_epoch.item() / batch_count}')

    if validation_dataset:
        return loss_edges_train, loss_reach_train, loss_parents_train, loss_edges_val, loss_reach_val, loss_parents_val
    return loss_edges_train, loss_reach_train, loss_parents_train

# MAIN ----------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import random_split

from generator import RandomGraphDataset
from network import Network

import json
import os

if __name__ == '__main__':
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

    root = os.path.join(os.getcwd(), 'data')
    path = os.path.join(root, f'n={n[0]}-{n[1]}_p={p}_size={size}')
    
    # Create a dataset of random graphs
    dataset = RandomGraphDataset(root=path, gen_num_graph=size, n=n, p=p)
    print(dataset.len())
    train_set_size_ = int(train_set_size * dataset.len())
    train_dataset, test_dataset = random_split(dataset, [train_set_size_, len(dataset) - train_set_size_])
    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    # Create a model
    model = Network(latent_dim=hidden_dim)
    optimizer = torch.optim.Adam
    # use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print('Using GPU')
    else:
        print('Using CPU')

    # Train the model
    train_loss, val_losses, y_losses, y_val_losses = train(model=model, train_dataset=train_dataset, validation_dataset=test_dataset, optimizer=optimizer(model.parameters(), lr=lr), epochs=epochs, batch_size=batch_size)