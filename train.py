import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(
    net, 
    trainloader, 
    validationloader, 
    input_shape, 
    class_weights, 
    num_stations, 
    num_epochs, 
    learning_rate, 
    smallest_event_level, 
    verbose=True
):
    # Initialize optimizer, loss function, and learning rate scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Initialize device and weight tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    w = torch.ones(num_stations).to(device)
    
    # Initialize storage for losses and scores
    train_running_loss = []
    validation_running_loss = []
    train_f1_scores = []
    valid_f1_scores = []
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []

        # Training loop
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            labels = labels.type(torch.LongTensor).to(device)
            loss = ((criterion(outputs, labels) * (labels > 0) * w).mean(0)).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        # Calculate training metrics
        train_loss = running_loss / len(trainloader)
        train_running_loss.append(train_loss)
        
        predictions_train = np.concatenate(y_pred_train).reshape(-1, num_stations).flatten()
        ground_truth_train = np.concatenate(y_true_train).reshape(-1, num_stations).flatten()
        acc, prc, rcl, prcrcl_ratio, prcrcl_avg, csi = get_metrics_binar(
            ground_truth_train, predictions_train, labels=[1, 2], smallest_event_level=smallest_event_level
        )
        train_f1_scores.append(csi)

        # Validation loop
        net.eval()
        running_loss_val = 0.0
        y_true_valid = []
        y_pred_valid = []

        with torch.no_grad():
            for i, data in enumerate(validationloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                labels = labels.type(torch.LongTensor).to(device)
                
                loss_val = ((criterion(outputs, labels) * (labels > 0) * w).mean(0)).mean()
                running_loss_val += loss_val.item()
                
                predicted = torch.argmax(outputs, dim=1)
                y_true_valid.extend(labels.cpu().numpy())
                y_pred_valid.extend(predicted.cpu().numpy())

        # Calculate validation metrics
        valid_loss = running_loss_val / len(validationloader)
        validation_running_loss.append(valid_loss)
        
        predictions_valid = np.concatenate(y_pred_valid).reshape(-1, num_stations).flatten()
        ground_truth_valid = np.concatenate(y_true_valid).reshape(-1, num_stations).flatten()
        acc_test, prc_test, rcl_test, prcrcl_ratio_test, prcrcl_avg_test, csi_test = get_metrics_binar(
            ground_truth_valid, predictions_valid, labels=[1, 2], smallest_event_level=smallest_event_level
        )
        valid_f1_scores.append(csi_test)
        
        # Step the scheduler
        scheduler.step(valid_loss)
        
        # Print epoch results if verbose
        if verbose:
            print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            print(f'Train acc {acc:.3f}, precision {prc:.3f}, recall {rcl:.3f}, CSI {csi:.3f}, avg {prcrcl_avg:.3f}')
            print(f'Validation acc {acc_test:.3f}, precision {prc_test:.3f}, recall {rcl_test:.3f}, CSI {csi_test:.3f}, avg {prcrcl_avg_test:.3f}')

    return net, train_running_loss, validation_running_loss
