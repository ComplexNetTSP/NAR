import torch
from torch.utils.data import random_split
from generator.generator import RandomGraphDataset
from model.network import Network
import matplotlib.pyplot as plt
import os

def train(model, train_dataset, validation_dataset=None, optimizer=None, epochs=10, batch_size=5, patience=5, model_save_path = None):
    x_loss_weight = 0.5
    h_loss_weight = 1 - x_loss_weight

    
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    loss_edges_train, reach_accuracy_train, parents_accuracy_train, edge_accuracy_train = [], [], [], []
    loss_edges_val, reach_accuracy_val, parents_accuracy_val, edge_accuracy_val = [], [], [], []

    for epoch in range(epochs):
        batch_count = len(train_dataset) // batch_size
        
        cumulated_loss_edges_epoch, cumulated_reach_accuracy_epoch, cumulated_parents_accuracy_epoch = 0, 0, 0
        cumulated_edge_accuracy_epoch = 0

        for i in range(batch_count):
            model.train()
            cumulated_loss_edges, cumulated_reach_accuracy, cumulated_parents_accuracy = 0, 0, 0
            cumulated_edge_accuracy = 0
            for j in range(i*batch_size, (i+1)*batch_size):
                graph = train_dataset[j] 
                loss_edges, edge_accuracy, reach_accuracy, parents_accuracy = model(graph)
                loss_edges_output, loss_edges_hints = loss_edges[0], loss_edges[1] # loss for the edges

                cumulated_loss_edges += x_loss_weight * loss_edges_output + h_loss_weight * loss_edges_hints
                cumulated_reach_accuracy += reach_accuracy
                cumulated_parents_accuracy += parents_accuracy
                cumulated_edge_accuracy += edge_accuracy

            cumulated_loss_edges /= batch_size
            cumulated_reach_accuracy /= batch_size
            cumulated_parents_accuracy /= batch_size
            cumulated_edge_accuracy /= batch_size
            
            optimizer.zero_grad()
            cumulated_loss_edges.backward()
            optimizer.step()

            cumulated_loss_edges_epoch += cumulated_loss_edges
            cumulated_reach_accuracy_epoch += cumulated_reach_accuracy
            cumulated_parents_accuracy_epoch += cumulated_parents_accuracy
            cumulated_edge_accuracy_epoch += cumulated_edge_accuracy

        # Convert tensors to lists and append to the respective lists
        loss_edges_train.append(cumulated_loss_edges_epoch.item() / batch_count)
        reach_accuracy_train.append(cumulated_reach_accuracy_epoch / batch_count)
        parents_accuracy_train.append(cumulated_parents_accuracy_epoch.item() / batch_count)
        edge_accuracy_train.append(cumulated_edge_accuracy_epoch / batch_count)

        if validation_dataset:
            model.eval()
            with torch.no_grad():
                cumulated_loss_edges_val, cumulated_reach_accuracy_val, cumulated_parents_accuracy_val = 0, 0, 0
                cumulated_edge_accuracy_val = 0
                for k in range(len(validation_dataset)):
                    graph = validation_dataset[k]
                    loss_edges, edge_accuracy, reach_accuracy, parents_accuracy = model(graph)
                    loss_edges_output, loss_edges_hints = loss_edges[0], loss_edges[1] # loss for the edges

                    cumulated_loss_edges_val += x_loss_weight * loss_edges_output + h_loss_weight * loss_edges_hints
                    cumulated_reach_accuracy_val += reach_accuracy
                    cumulated_parents_accuracy_val += parents_accuracy
                    cumulated_edge_accuracy_val += edge_accuracy

                cumulated_loss_edges_val /= len(validation_dataset)
                cumulated_reach_accuracy_val /= len(validation_dataset)
                cumulated_parents_accuracy_val /= len(validation_dataset)
                cumulated_edge_accuracy_val /= len(validation_dataset)

                loss_edges_val.append(cumulated_loss_edges_val.item())
                reach_accuracy_val.append(cumulated_reach_accuracy_val)
                parents_accuracy_val.append(cumulated_parents_accuracy_val.item())
                edge_accuracy_val.append(cumulated_edge_accuracy_val)

                print(f'Epoch {epoch}, loss_edges {cumulated_loss_edges_epoch.item() / batch_count:.5f}, edges_acccuracy_train {edge_accuracy_train[-1]:.5f}%, reach_accuracy {cumulated_reach_accuracy_epoch / batch_count:.5f}, parents_accuracy {cumulated_parents_accuracy_epoch.item() / batch_count:.5f}, loss_edges_val {cumulated_loss_edges_val.item():.5f}, edge_accuracy_val {cumulated_edge_accuracy_val:.5f}%, reach_accuracy_val {cumulated_reach_accuracy_val:.5f}, parents_accuracy_val {cumulated_parents_accuracy_val.item():.5f}')

                # Check for early stopping
                if cumulated_loss_edges_val < best_val_loss:
                    best_val_loss = cumulated_loss_edges_val
                    no_improvement_count = 0
                    # Save the best model state
                    if model_save_path:
                        torch.save(model.state_dict(), model_save_path)
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                    break
        
        else:
            print(f'Epoch {epoch}, loss_edges {cumulated_loss_edges_epoch.item() / batch_count}, reach_accuracy {cumulated_reach_accuracy_epoch.item() / batch_count}, parents_accuracy {cumulated_parents_accuracy_epoch.item() / batch_count}')

        # Early stopping condition
        if no_improvement_count >= patience:
            break

    if validation_dataset:
        return loss_edges_train, reach_accuracy_train, parents_accuracy_train, edge_accuracy_train, loss_edges_val, reach_accuracy_val, parents_accuracy_val, edge_accuracy_val
    return loss_edges_train, reach_accuracy_train, parents_accuracy_train, edge_accuracy_train

# MAIN ----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Parameters
   
    ## Graph generation
    test_size = 350
    validation_size = 50
    n = [20, 100]
    p = 0.3
    model_save_path = os.getcwd()+"/best_model.pth"
    
    ## Training
    batch_size = 32
    lr = 0.0002
    epochs = 10

    # Dataset
    dataset = RandomGraphDataset(root='./data/train_validation_set', gen_num_graph=test_size+validation_size, n=n, p=p, type='erdos_renyi')
    train_dataset, test_dataset = random_split(dataset, [test_size, validation_size])

    # Model and Optimizer
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    loss_edges_train, reach_accuracy_train, parents_accuracy_train, edge_accuracy_train, loss_edges_val, reach_accuracy_val, parents_accuracy_val, edge_accuracy_val = train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=test_dataset,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=model_save_path
    )