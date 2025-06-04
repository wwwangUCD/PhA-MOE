import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mult=[2, 4, 2, 1], activation='relu'):
        super(MLP, self).__init__()

        # Dynamically create layers based on the mult list
        layers = []

        current_size = input_size
        for factor in mult:
            next_size = hidden_size * factor
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            if activation=='relu':
                layers.append(nn.ReLU())
            elif activation=='tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())

            layers.append(nn.Dropout(0.5))
            current_size = next_size

        # Add the output layer
        layers.append(nn.Linear(current_size, output_size))

        # Combine all layers into a Sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        return self.layers(x)

def evaluate_model(model, test_loader, criterion, device, record_outputs=False):
    # Ensure the model is in evaluation mode
    model.eval()

    # Initialize variables to calculate test loss
    test_loss_sum = 0.0
    num_test_samples = 0

    # Lists to store all targets and outputs
    all_targets = []
    all_outputs = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, targets in test_loader:
            if len(inputs.shape) == 3:
                inputs = inputs.view(inputs.shape[0], -1)
            if len(targets.shape) == 3:
                targets = targets.view(targets.shape[0], -1)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass to get outputs
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Accumulate test loss
            test_loss_sum += loss.item() * inputs.shape[0]
            num_test_samples += inputs.shape[0]

            # Store the targets and outputs
            if record_outputs:
                all_targets.append(targets.cpu())
                all_outputs.append(outputs.cpu())

    # Calculate the average test loss
    avg_test_loss = test_loss_sum / num_test_samples

    if record_outputs:
        all_targets = torch.cat(all_targets, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)

    if record_outputs:
        return all_targets, all_outputs, avg_test_loss
    else:
        return avg_test_loss

