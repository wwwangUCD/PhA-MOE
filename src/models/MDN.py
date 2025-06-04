# Author: Weiwei Wang
# This MDN code is transfered from https://github.com/BrandonSmithJ/MDN
# where the original version is based on tensorflow and Weiwei rewrote it into the pytorch version

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.distributions import Categorical, MultivariateNormal
class MDN(nn.Module):
    def __init__(self, n_inputs, n_targets, n_mix=5, hidden=[100] * 5, lr=1e-3, l2=1e-3, epsilon=1e-3, activation='relu'):
        super(MDN, self).__init__()

        self.n_inputs = n_inputs
        self.n_targets = n_targets
        self.n_mix = n_mix
        self.hidden = hidden
        self.lr = lr
        self.l2 = l2
        self.epsilon = epsilon
        self.activation = activation

        # Define the model layers
        self.model_layers = nn.ModuleList()
        for inp, out in zip([self.n_inputs] + self.hidden[:-1], self.hidden):
            self.model_layers.append(nn.Linear(inp, out))
            if self.activation == 'relu':
                self.model_layers.append(nn.ReLU())
            elif self.activation == 'sigmoid':
                self.model_layers.append(nn.Sigmoid())
            elif self.activation == 'tanh':
                self.model_layers.append(nn.Tanh())
        # self.model_layers.append(nn.Dropout(0.5))

        # Define the output layer
        self.output_layer = MixtureLayer(n_mix=self.n_mix, n_targets=self.n_targets, epsilon=self.epsilon, insize=self.hidden[-1])

        # Define the optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    def forward(self, x):
        # Forward pass through the model layers
        for layer in self.model_layers:
            x = layer(x)
        # Pass through output layer
        x = self.output_layer(x)
        return x
    def compute_l2_loss(self):
        l2_loss = 0.0
        for layer in self.model_layers:
            if isinstance(layer, nn.Linear):
                l2_loss += torch.sum(layer.weight ** 2)
                if layer.bias is not None:
                    l2_loss += torch.sum(layer.bias ** 2)
        return self.l2 * l2_loss

    def compute_l1_loss(self):
        l1_loss = 0.0
        for layer in self.model_layers:
            if isinstance(layer, nn.Linear):
                l1_loss += torch.sum(torch.abs(layer.weight))
                if layer.bias is not None:
                    l1_loss += torch.sum(torch.abs(layer.bias))
        return self.l2 * l1_loss

    def loss(self, output, y):
        prior, mu, scale_tril = self._parse_outputs(output)

        # Create the mixture components
        mvn_components = [MultivariateNormal(mu[:, i, :], scale_tril=scale_tril[:, i, :, :]) for i in range(self.n_mix)]

        # Compute log probabilities for each component
        log_probs = torch.stack([mvn.log_prob(y) for mvn in mvn_components], dim=1)

        # Compute the log likelihood of the mixture using logsumexp for numerical stability
        weighted_log_probs = log_probs + torch.log(prior)
        mixture_log_likelihood = torch.logsumexp(weighted_log_probs, dim=1)

        # Compute the loss
        loss_value = -torch.mean(mixture_log_likelihood)

        return loss_value+self.compute_l2_loss()

    def get_coefs(self, output):
        prior, mu, scale = self._parse_outputs(output)
        return prior, mu, self._covariance(scale)

    def _predict(self, X, return_coefs=False, **kwargs):
        ''' Generates estimates for the given set. X may be only a subset of the full
            data, which speeds up the prediction process and limits memory consumption.
             '''
        model_out = self.forward(X)
        coefs_out = self.get_coefs(model_out)
        outputs = self.extract_predictions(coefs_out, **kwargs)

        if return_coefs:
            return outputs, [c.cpu().detach().numpy() for c in coefs_out]
        return outputs
    def sample(self, x, n_samples=1):
        output = self.forward(x)
        prior, mu, scale_tril = self.get_coefs(output)

        # Sample from the mixture components
        categorical = torch.distributions.Categorical(prior)
        mvn_components = [MultivariateNormal(mu[:, i, :], scale_tril=scale_tril[:, i, :, :]) for i in range(self.n_mix)]

        samples = []
        for _ in range(n_samples):
            indices = categorical.sample().unsqueeze(-1).expand(x.size(0), self.n_targets)
            component_samples = torch.stack([mvn.sample() for mvn in mvn_components], dim=1)
            samples.append(torch.gather(component_samples, 1, indices.unsqueeze(1)).squeeze(1))

        return torch.stack(samples).mean(0)
    def predict(self, X, return_coefs=False, **kwargs):
        '''Top level interface to get predictions for a given dataset without chunking.

        return_coefs : bool, optional (default=False)
            If True, return the estimated coefficients (prior, mu, sigma) along with the
            other requested outputs. Note that rescaling the coefficients using scalerx/y
            is left up to the user, as calculations involving sigma must be performed in
            the basis learned by the model.
        '''
        # Directly call _predict without chunking
        preds, coefs = self._predict(X, return_coefs=True, **kwargs)

        if return_coefs:
            return preds, coefs
        return preds

    def extract_predictions(self, coefs, confidence_interval=None, threshold=None, avg_est=False):
        '''
        Function used to extract model predictions from the given set of
        coefficients. Users should call the predict() method instead, if
        predictions from input data are needed.

        confidence_interval : float, optional (default=None)
            If a confidence interval value is given, then this function
            returns (along with the predictions) the upper and lower
            {confidence_interval*100}% confidence bounds around the prediction.

        threshold : float, optional (default=None)
            If set, the model outputs the maximum prior estimate when the prior
            probability is above this threshold; and outputs the average estimate
            when below the threshold. Any passed value should be in the range (0, 1],
            though the sign of the threshold can be negative in order to switch the
            estimates (i.e. negative threshold would output average estimate when prior
            is greater than the (absolute) value).

        avg_est : bool, optional (default=False)
            If true, model outputs the prior probability weighted mean as the
            estimate. Otherwise, model outputs the maximum prior estimate.
        '''
        assert (confidence_interval is None or (
                    0 < confidence_interval < 1)), 'confidence_interval must be in the range (0,1)'
        assert (threshold is None or (0 < threshold <= 1)), 'threshold must be in the range (0,1]'

        target = ('avg' if avg_est else 'top') if threshold is None else 'threshold'
        output = getattr(self, f'_get_{target}_estimate')(coefs)

        if confidence_interval is not None:
            assert (threshold is None), f'Cannot calculate confidence on thresholded estimates'
            confidence = getattr(self, f'_get_{target}_confidence')(coefs, confidence_interval)
            upper_bar = output + confidence
            lower_bar = output - confidence
            return output, upper_bar, lower_bar
        return output

    def _parse_outputs(self, output):
        # prior, mu, scale = torch.split(output, [self.n_mix, self.n_mix * self.n_targets, -1], dim=1)
        prior, mu, scale = torch.split(output, [self.n_mix, self.n_mix * self.n_targets, self.n_mix * self.n_targets * self.n_targets], dim=1)
        prior = prior.view(-1, self.n_mix)
        mu = mu.view(-1, self.n_mix, self.n_targets)
        scale = scale.reshape(-1, self.n_mix, self.n_targets, self.n_targets)
        return prior, mu, scale
    def _covariance(self, scale):
        return torch.einsum('abij,abjk->abik', scale.transpose(-1, -2), scale)

    def _calculate_top(self, prior, values):
        # Get the top values and indices
        vals, idxs = torch.topk(prior, k=1, dim=1)

        # Ensure idxs is on the same device
        device = prior.device

        # Create the range tensor and stack with idxs
        range_tensor = torch.arange(idxs.size(0), device=device).unsqueeze(1)
        # Gather values based on idxs
        # top_values = values[range_tensor, idxs].squeeze()
        # top_values = top_values.unsqueeze(1)

        top_values = values[range_tensor, idxs]
        if top_values.shape[0]!=1:
            top_values=top_values.squeeze()
            top_values = top_values.unsqueeze(1)
        else:
            top_values = top_values.view(top_values.shape[0], -1)
        return top_values

    def _get_top_estimate(self, coefs, **kwargs):
        prior, mu, _ = coefs
        return self._calculate_top(prior, mu)

    def _get_avg_estimate(self, coefs, **kwargs):
        prior, mu, _ = coefs
        return torch.sum(mu * prior.unsqueeze(-1), dim=1)

    def _get_threshold_estimate(self, coefs, threshold=0.5):
        top_estimate = self._get_top_estimate(coefs)
        avg_estimate = self._get_avg_estimate(coefs)
        prior, _, _ = coefs
        threshold_mask = (prior.max(dim=1).values / threshold).sign().unsqueeze(-1)
        return torch.where(threshold_mask.unsqueeze(-1), top_estimate, avg_estimate)
    def _calculate_confidence(self, sigma, level=0.9):
        # Calculate confidence based on the given sigma (covariance matrix)
        # For a given confidence level probability p (0<p<1), and number of dimensions d,
        # rho is the error bar coefficient: rho = sqrt(2) * erfinv(p ** (1/d))
        # Ref: https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cs-99-03.pdf
        u, s, v = torch.svd(sigma)
        rho = (2 ** 0.5) * torch.erfinv((level ** (1.0 / self.n_targets)))
        return rho * (2 * s.sqrt())

    def _get_top_confidence(self, coefs, level=0.9):
        prior, mu, sigma = coefs
        top_sigma = self._calculate_top(prior, sigma)
        return self._calculate_confidence(top_sigma, level)

    def _get_avg_confidence(self, coefs, level=0.9):
        prior, mu, sigma = coefs
        avg_estim = self._get_avg_estimate(coefs)
        avg_sigma = torch.sum((prior.unsqueeze(-1).unsqueeze(-1)) * (sigma + torch.matmul((mu - avg_estim.unsqueeze(1)).transpose(-1, -2), (mu - avg_estim.unsqueeze(1)))), dim=1)
        return self._calculate_confidence(avg_sigma, level)


class MixtureLayer(nn.Module):
    def __init__(self, n_mix, n_targets, epsilon, insize):
        super(MixtureLayer, self).__init__()

        # Extract relevant layer kwargs
        # Define the parameters
        self.n_mix = n_mix
        self.n_targets = n_targets
        self.epsilon = epsilon
        self.insize = insize
        self.linear = nn.Linear(self.insize, self.n_outputs)  # Create linear layer

    @property
    def layer_sizes(self):
        ''' Sizes of the prior, mu, and (lower triangle) scale matrix outputs '''
        sizes = [1, self.n_targets, (self.n_targets * (self.n_targets + 1)) // 2]
        return self.n_mix * torch.tensor(sizes)
        # e.g array([5, 5, 5])

    @property
    def n_outputs(self):
        ''' Total output size of the layer object '''
        return sum(self.layer_sizes)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        prior, mu, scale = torch.split(outputs, self.layer_sizes.tolist(), dim=1)

        prior = F.softmax(prior, dim=-1) + 1e-9
        mu = torch.stack(torch.split(mu, self.n_mix, dim=1), dim=2) # TensorShape([278, 5, 1])
        scale = torch.stack(torch.split(scale, self.n_mix, dim=1), dim=2) # TensorShape([278, 5, 1])
        scale = fill_triangular(scale)  # Lower triangle of scale matrix, TensorShape([278, 5, 1, 1])
        norm = torch.eye(self.n_targets).unsqueeze(0).unsqueeze(0) # TensorShape([1, 1, 1, 1])
        sigma = torch.einsum('abij,abjk->abik', scale.transpose(-1, -2), scale)
        norm=norm.to(sigma.device)
        # sigma += self.epsilon * norm
        # scale = torch.linalg.cholesky(sigma)
        # sometimes the result is non PSD, and the reason is the epsilon is too small, so we make it larger to avoid this error

        factor = 1
        while True:
            try:
                sigma += factor * self.epsilon * norm
                scale = torch.linalg.cholesky(sigma)
                break  # Exit the loop if Cholesky decomposition succeeds
            except torch._C._LinAlgError as e:
                print(f"Cholesky decomposition failed. Doubling factor to {factor * 2}")
                factor *= 2  # Increase the factor

        # Reshape outputs
        prior = prior.view(-1, self.n_mix)
        mu = mu.view(-1, self.n_mix * self.n_targets)
        scale = scale.reshape(-1, self.n_mix * (self.n_targets ** 2)) # of shape bs*5*n_targets*n_targets

        return torch.cat([prior, mu, scale], dim=1) # TensorShape([278, 15])


def fill_triangular(tensor, upper=False):
    batch_size, n_mix, num_elements = tensor.shape
    # Calculate the size of the triangular matrix
    n_targets = int((math.sqrt(1 + 8 * num_elements) - 1) / 2)

    # Create an empty tensor to hold the triangular matrices
    triangular = torch.zeros(batch_size, n_mix, n_targets, n_targets, device=tensor.device)

    # Get the indices for the triangular part
    if upper:
        rows, cols = torch.triu_indices(n_targets, n_targets, offset=0)
    else:
        rows, cols = torch.tril_indices(n_targets, n_targets, offset=0)

    # Fill the triangular part with values from the input tensor
    triangular[:, :, rows, cols] = tensor

    return triangular


def evaluate_model(model, test_loader, criterion, device, record_outputs=False):
    predict_kwargs = {
        'avg_est': False,
        'threshold': None,
        'confidence_interval': None,
        'return_coefs': True
    }

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

            estimates, coefs = model.predict(inputs, **predict_kwargs)
            # Store the targets and outputs
            if record_outputs:
                all_targets.append(targets.cpu())
                # all_outputs.append(outputs.cpu())
                # in MDN model the estimation output is
                all_outputs.append(estimates.cpu())
                # Calculate the average test loss
    avg_test_loss = test_loss_sum / num_test_samples

    if record_outputs:
        all_targets = torch.cat(all_targets, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        if len(all_outputs.shape) == 1:
            all_outputs = all_outputs.view(-1, 1)
    if record_outputs:
        return all_targets, all_outputs, avg_test_loss
    else:
        return avg_test_loss


