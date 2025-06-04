"""
The MDSA,SSPB,slope evaluation functions comes from https://github.com/BrandonSmithJ/MDN metrics.py
"""


from scipy import stats
import numpy as np 
import functools, warnings

def ignore_warnings(func):
    ''' Decorator to silence all warnings (Runtime, User, Deprecation, etc.) '''
    @functools.wraps(func)
    def helper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            return func(*args, **kwargs)
    return helper
def validate_shape(func):
	''' Decorator to flatten all function input arrays, and ensure shapes are the same '''
	@functools.wraps(func)
	def helper(*args, **kwargs):
		flat     = [a.flatten() if hasattr(a, 'flatten') else a for a in args]
		flat_shp = [a.shape for a in flat if hasattr(a, 'shape')]
		orig_shp = [a.shape for a in args if hasattr(a, 'shape')]
		assert(all(flat_shp[0] == s for s in flat_shp)), f'Shapes mismatch in {func.__name__}: {orig_shp}'
		return func(*flat, **kwargs)
	return helper  


def only_finite(func):
	''' Decorator to remove samples which are nan in any input array '''
	@validate_shape
	@functools.wraps(func)
	def helper(*args, **kwargs):
		stacked = np.vstack(args)
		valid   = np.all(np.isfinite(stacked), 0)
		assert(valid.sum()), f'No valid samples exist for {func.__name__} metric'
		return func(*stacked[:, valid], **kwargs)
	return helper 


def only_positive(func):
	''' Decorator to remove samples which are zero/negative in any input array '''
	@validate_shape
	@functools.wraps(func)	
	def helper(*args, **kwargs):
		stacked = np.vstack(args)
		valid   = np.all(stacked > 0, 0)
		assert(valid.sum()), f'No valid samples exist for {func.__name__} metric'
		return func(*stacked[:, valid], **kwargs)
	return helper 



def label(name):
	''' Label a function to aid in printing '''
	def wrapper(func):
		func.__name__ = name
		return ignore_warnings(func)
	return wrapper



def calculate_nrmse_per_element(y, y_hat, avg=True):
    """
    Calculate the NRMSE for each element in the vector and then average them.

    Args:
    y (numpy.ndarray): True values, shape (N, M)
    y_hat (numpy.ndarray): Predicted values, shape (N, M)

    Returns:
    float: Average NRMSE across all elements
    """
    n_elements = y.shape[1]

    nrmse_per_element = np.zeros(n_elements)
    for j in range(n_elements):
        mse_j = np.mean(((y[:, j] - y_hat[:, j]) / y[:, j]) ** 2)
        nrmse_per_element[j] = np.sqrt(mse_j)

    average_nrmse = np.mean(nrmse_per_element)
    if avg:
        return average_nrmse
    else:
        return nrmse_per_element

def calculate_nrmse_vector_norm(y, y_hat):
    """
    Calculate the NRMSE using the vector norm method.

    Args:
    y (numpy.ndarray): True values, shape (N, M)
    y_hat (numpy.ndarray): Predicted values, shape (N, M)

    Returns:
    float: NRMSE
    """
    mse = np.mean(np.sum(((y - y_hat) / y) ** 2, axis=1))
    nrmse = np.sqrt(mse)
    return nrmse


def calculate_nrmse_wholeBand(target, output):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) between target and output.

    Parameters:
    - target (np.array): Ground truth values.
    - output (np.array): Predicted values.

    Returns:
    - float: NRMSE value.
    """
    return np.sqrt(np.mean((target - output) ** 2)) / (np.max(target) - np.min(target))


def calculate_log_rmse(y, y_hat, epsilon=1e-6):
    """
    Calculate the RMSE between the logarithms of y and y_hat.

    Args:
    y (numpy.ndarray): True values, shape (N, M)
    y_hat (numpy.ndarray): Predicted values, shape (N, M)
    epsilon (float): Small value to add to avoid taking the log of zero

    Returns:
    float: RMSE between log(y) and log(y_hat)
    """
    y = np.maximum(y, epsilon)
    y_hat = np.maximum(y_hat, epsilon)

    log_y = np.log(y)
    log_y_hat = np.log(y_hat)

    mse_log = np.mean((log_y - log_y_hat) ** 2)
    rmse_log = np.sqrt(mse_log)
    return rmse_log

@only_finite
@only_positive
@label('MdSA')
def mdsa(y, y_hat):
	''' Median Symmetric Accuracy '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	return 100 * (np.exp(np.median(np.abs(np.log(y_hat / y)))) - 1)


@only_finite
@only_positive
@label('NRMSE')
def nrmse(y, y_hat, normalization_mode='relative'):
    ''' Normalized Root Squared Error with different normalization options

    Parameters:
    y: array-like, true values
    y_hat: array-like, predicted values
    normalization_mode: str, normalization method to use
        - 'range': Normalize by the range of y (max(y) - min(y))
        - 'mean': Normalize by the mean of y
        - 'relative': Use relative errors ((y_i - y_hat_i) / y_i) for RMSE calculation
    '''
    mse = np.mean((y_hat - y) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    if normalization_mode == 'range':
        normalization = np.max(y) - np.min(y)  # Range of y
    elif normalization_mode == 'mean':
        normalization = np.mean(y)  # Mean of y
    elif normalization_mode == 'relative':
        relative_errors = (y - y_hat) / y
        rmse = np.sqrt(np.mean(relative_errors ** 2))  # RMSE using relative errors
        normalization = 1  # No further normalization required
    else:
        raise ValueError("Invalid normalization_mode. Choose 'range', 'mean', or 'relative'.")

    return 100 * (rmse / normalization)


@only_finite
@only_positive
@label('SSPB')
def sspb(y, y_hat):
	''' Symmetric Signed Percentage Bias '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	M = np.median( np.log(y_hat / y) )
	return 100 * np.sign(M) * (np.exp(np.abs(M)) - 1)

@only_finite
@only_positive
@label('Slope')
def slope(y, y_hat):
	''' Logarithmic slope '''
	slope_, intercept_, r_value, p_value, std_err = stats.linregress(np.log10(y), np.log10(y_hat))
	return slope_


def calculate_mdsa_wl(y, y_hat, avg=True):
    """
    Calculate the MDSA for each column of the input arrays y and y_hat,
    and then average the result across all columns.

    Args:
    y (numpy.ndarray): True values, shape (N, M)
    y_hat (numpy.ndarray): Predicted values, shape (N, M)

    Returns:
    float: Average MDSA across all columns
    """
    n_columns = y.shape[1]
    mdsa_values = np.zeros(n_columns)

    # Loop through each column and calculate MDSA
    for i in range(n_columns):
        mdsa_values[i] = mdsa(y[:, i], y_hat[:, i])

    # Calculate the average MDSA across all columns
    average_mdsa = np.mean(mdsa_values)
    if avg:
        return average_mdsa
    else:
        return mdsa_values

def calculate_sspb_wl(y, y_hat, avg=True):
    """
    Calculate the SSPB for each column of the input arrays y and y_hat,
    and then average the result across all columns.

    Args:
    y (numpy.ndarray): True values, shape (N, M)
    y_hat (numpy.ndarray): Predicted values, shape (N, M)

    Returns:
    float: Average SSPB across all columns
    """
    n_columns = y.shape[1]
    sspb_values = np.zeros(n_columns)

    # Loop through each column and calculate SSPB
    for i in range(n_columns):
        sspb_values[i] = sspb(y[:, i], y_hat[:, i])

    # Calculate the average SSPB across all columns
    average_sspb = np.mean(np.abs(sspb_values))
    if avg:
        return average_sspb
    else:
        return sspb_values


def calculate_slope_wl(y, y_hat, avg=True):
    """
    Calculate the Slope for each column of the input arrays y and y_hat,
    and then average the result across all columns.

    Args:
    y (numpy.ndarray): True values, shape (N, M)
    y_hat (numpy.ndarray): Predicted values, shape (N, M)

    Returns:
    float: Average Slope across all columns
    """
    n_columns = y.shape[1]
    slope_values = np.zeros(n_columns)
    slope_deviation_values = np.zeros(n_columns)
    # Loop through each column and calculate Slope
    for i in range(n_columns):
        slope_values[i] = slope(y[:, i], y_hat[:, i])
        slope_deviation_values[i] = np.abs(slope_values[i] - 1)
    # Calculate the average Slope across all columns
    average_slope = np.mean(slope_values)
    average_slope_deviation = np.mean(slope_deviation_values)
    if avg:
        return average_slope_deviation
    else:
        return slope_values, slope_deviation_values

def calculate_nrmse_wl(y, y_hat, normalization_mode,avg=True):
    """
    Calculate the MDSA for each column of the input arrays y and y_hat,
    and then average the result across all columns.

    Args:
    y (numpy.ndarray): True values, shape (N, M)
    y_hat (numpy.ndarray): Predicted values, shape (N, M)

    Returns:
    float: Average MDSA across all columns
    """
    n_columns = y.shape[1]
    nrmse_values = np.zeros(n_columns)

    # Loop through each column and calculate MDSA
    for i in range(n_columns):
        nrmse_values[i] = nrmse(y[:, i], y_hat[:, i],normalization_mode)

    # Calculate the average nrmse across all columns
    average_nrmse = np.mean(nrmse_values)
    if avg:
        return average_nrmse
    else:
        return nrmse_values
# Example usage:
if __name__ == "__main__":
    y_true = np.random.rand(100, 52)  # 100 samples, each with length 52
    y_pred = np.random.rand(100, 52)
    y_true = np.abs(y_true) + 1e-6
    y_pred = np.abs(y_pred) + 1e-6
    avg_nrmse = calculate_nrmse_per_element(y_true, y_pred)
    print(f"Average NRMSE (Per Element): {avg_nrmse}")

    nrmse = calculate_nrmse_vector_norm(y_true, y_pred)
    print(f"NRMSE (Vector Norm): {nrmse}")

    log_rmse = calculate_log_rmse(y_true, y_pred)
    print(f"Log RMSE: {log_rmse}")
