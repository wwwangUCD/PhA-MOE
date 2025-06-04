import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mult_e=[2, 4], mult_d=[4, 8], activation='leakyrelu', use_batchnorm=True, use_dropout=False):
        super().__init__()
        # Define the activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Encoder layers
        self.encoder_layers = self.build_layers(input_size, mult_e, hidden_size, use_batchnorm, use_dropout)
        self.fc_mu = nn.Linear(mult_e[-1] * hidden_size, hidden_size)
        self.fc_log_var = nn.Linear(mult_e[-1] * hidden_size, hidden_size)

        # Decoder layers
        self.decoder_layers = self.build_layers(hidden_size, mult_d, hidden_size, use_batchnorm, use_dropout)
        self.decoder_layers.add_module('output_layer', nn.Linear(mult_d[-1] * hidden_size, output_size))
        # self.decoder_layers.add_module('output_activation', nn.Tanh())  # Assuming output is in range [-1, 1]
        # with the classic robust preprocessing method it is -1 to 1, but for others it may not.
    def build_layers(self, input_size, mult, hidden_size, use_batchnorm,use_dropout=False):
        layers = []
        current_size = input_size
        for factor in mult:
            next_size = hidden_size * factor
            layers.append(nn.Linear(current_size, next_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(next_size))
            layers.append(self.activation)
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            current_size = next_size
        return nn.Sequential(*layers)
    def encode(self, x):
        x = self.encoder_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder_layers(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return output, mu, log_var


def loss_function(output, x, mu, log_var, kld_weight=0.1):
    # Reconstruction loss (MSE)
    MSE = F.mse_loss(output, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Return combined loss
    return MSE + kld_weight*KLD


def evaluate_model(model, test_loader, device, record_outputs=False, kld_weight=0.1):
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
            outputs, mus, log_vars = model(inputs)

            # Compute the loss
            loss = loss_function(outputs, targets, mus, log_vars, kld_weight=kld_weight)

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



