import torch

def train(model, train_dataset, validation_dataset=None, optimizer=None, epochs=10, batch_size=32):
    x_loss_weight = 0.5
    h_loss_weight = 1 - x_loss_weight
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    y_losses = []
    y_val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to train mode
        epoch_loss = 0
        epoch_y_loss = 0
        
        # Batch-wise training
        for i in range(0, len(train_dataset), batch_size):
            batch_graphs = train_dataset[i:i+batch_size]
            
            optimizer.zero_grad()  # Zero gradients
            
            batch_loss = 0
            batch_y_loss = 0
            
            for graph in batch_graphs:
                y, loss, y_loss = model(graph)
                loss_x, loss_hints = loss[0], loss[1]
                y_loss_x, y_loss_hints = y_loss[0], y_loss[1]
                
                batch_loss += x_loss_weight * loss_x + h_loss_weight * loss_hints
                batch_y_loss += x_loss_weight * y_loss_x + h_loss_weight * y_loss_hints
            
            batch_loss /= len(batch_graphs)  # Average loss over batch
            batch_y_loss /= len(batch_graphs)  # Average y loss over batch
            
            batch_loss.backward()  # Backpropagation
            optimizer.step()  # Optimizer step
            
            epoch_loss += batch_loss.item()
            epoch_y_loss += batch_y_loss.item()

        # Average losses over all batches
        epoch_loss /= (len(train_dataset) // batch_size)
        epoch_y_loss /= (len(train_dataset) // batch_size)
                
        # Append losses to respective lists
        train_losses.append(epoch_loss)
        y_losses.append(epoch_y_loss)
        
        # Validation
        if validation_dataset:
            model.eval()  # Set the model to evaluation mode
            cumulated_loss_val = 0
            cumulated_y_loss_val = 0
            
            with torch.no_grad():
                for graph in validation_dataset:
                    y, loss, y_loss = model(graph)
                    loss_x, loss_hints = loss[0], loss[1]
                    y_loss_x, y_loss_hints = y_loss[0], y_loss[1]
                    
                    cumulated_loss_val += x_loss_weight * loss_x + h_loss_weight * loss_hints
                    cumulated_y_loss_val += x_loss_weight * y_loss_x + h_loss_weight * y_loss_hints
            
            # Average losses over validation dataset
            cumulated_loss_val /= len(validation_dataset)
            cumulated_y_loss_val /= len(validation_dataset)
            
            print(f'Epoch {epoch+1}, loss {epoch_loss:.4f}, validation loss {cumulated_loss_val:.4f} || '
                  f'y_loss {epoch_y_loss:.4f}, validation y_loss {cumulated_y_loss_val:.4f}')
            
            val_losses.append(cumulated_loss_val.item())
            y_val_losses.append(cumulated_y_loss_val.item())

        else:
            print(f'Epoch {epoch+1}, loss {epoch_loss:.4f} || y_loss {epoch_y_loss:.4f}')  
        
        # Early stopping
        if len(val_losses) > 5:
            if all(val_losses[-1] > val_losses[i] for i in range(-2, -6, -1)):
                print('Early stopping')
                break
    
    return train_losses, val_losses, y_losses, y_val_losses

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