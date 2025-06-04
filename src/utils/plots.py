import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from matplotlib import cm
def wavelength_to_rgb(wavelength, gamma=0.8, max_intensity=255):
    """
    Convert a wavelength in the range 380-780 nm to an RGB color.

    Parameters:
    - wavelength (float): Wavelength in nanometers.
    - gamma (float): Gamma correction factor.
    - max_intensity (int): Maximum intensity for RGB components (default is 255).

    Returns:
    - tuple: (R, G, B) values corresponding to the wavelength.
    """
    R, G, B = 0, 0, 0

    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0

    # Intensity factor based on wavelength
    if wavelength > 700:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    elif wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    else:
        factor = 1.0

    # Apply gamma correction and scale to the RGB range
    R = int(max_intensity * (R * factor) ** gamma)
    G = int(max_intensity * (G * factor) ** gamma)
    B = int(max_intensity * (B * factor) ** gamma)

    return (R, G, B)

def plot_metrics(metrics, wavelength_range, colors, metric_str, bar_width=1.0):
    """
    Plot the given metrics as a bar plot with the specified colors, labels, and bar width.

    Parameters:
    - metrics (list or np.array): The values of the metric to plot.
    - wavelength_range (range): The range of wavelengths corresponding to the metrics.
    - colors (list of tuples): The RGB colors to use for each bar, where each tuple is (R, G, B).
    - metric_str (str): The label for the y-axis indicating the metric being plotted.
    - bar_width (float): The width of the bars in the plot.
    """

    # Create the bar plot
    plt.figure(figsize=(14, 8))
    plt.bar(wavelength_range, metrics, 
            color=[(r/255, g/255, b/255) for r, g, b in colors], 
            alpha=1, width=bar_width, edgecolor='none', zorder=3)

    # Labeling
    plt.xlabel('Wavelength (nm)', fontsize=28, fontname='Times New Roman')
    plt.ylabel(metric_str, fontsize=28, fontname='Times New Roman')
    plt.grid(True, which="both", ls="--", zorder=0)
    plt.xticks(fontsize=20, fontname='Times New Roman')
    plt.yticks(fontsize=20, fontname='Times New Roman')

    # Display the plot
    plt.show()


def plot_metrics_new(metrics, wavelength_range, colors, metric_str, bar_width=1.0, fontsize=28):
    """
    Plot the given metrics as a bar plot with the specified colors, labels, and bar width.

    Parameters:
    - metrics (list or np.array): The values of the metric to plot.
    - wavelength_range (range or list): The range of wavelengths corresponding to the metrics.
    - colors (list of tuples): The RGB colors to use for each bar, where each tuple is (R, G, B).
    - metric_str (str): The label for the y-axis indicating the metric being plotted.
    - bar_width (float): The width of the bars in the plot.
    - fontsize (int): Font size for labels and ticks.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object for further processing or saving.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Normalize colors to (0,1) range and create the bar plot
    ax.bar(wavelength_range, metrics,
           color=[(r / 255, g / 255, b / 255) for r, g, b in colors],
           alpha=1, width=bar_width, edgecolor='none', zorder=3)

    # Labeling
    ax.set_xlabel('Wavelength (nm)', fontsize=fontsize, fontname='Times New Roman')
    ax.set_ylabel(metric_str, fontsize=fontsize, fontname='Times New Roman')
    ax.grid(True, which="both", linestyle="--", zorder=0)
    ax.tick_params(axis='x', labelsize=fontsize - 8)
    ax.tick_params(axis='y', labelsize=fontsize - 8)

    return fig  # Return the figure for saving or displaying later

from matplotlib.font_manager import FontProperties

def plot_aphy_comparison(wavelength_range, test_targets, test_outputs, idx, linewidth=2, fontsize=28):
    """
    Plot the ground truth Aphy and the estimated Aphy for a given data point.

    Parameters:
    - wavelength_range (range): The range of wavelengths corresponding to the data points.
    - test_targets (np.array): The ground truth Aphy values (shape: [num_data_points, num_wavelengths]).
    - test_outputs (np.array): The estimated Aphy values (shape: [num_data_points, num_wavelengths]).
    - idx (int): The index of the data point to plot.
    - linewidth (int): The line width for the plot lines. Default is 2.
    - fontsize (int): The font size for labels and title. Default is 28.
    """

    # Extract the specific target and output data for the given index
    aphy_ground_truth = test_targets[idx, :]
    aphy_estimated = test_outputs[idx, :]

    # Plotting the data
    plt.figure(figsize=(14, 8))
    plt.plot(wavelength_range, aphy_ground_truth, label='Aphy Ground Truth', color='blue', linewidth=linewidth)
    plt.plot(wavelength_range, aphy_estimated, label='Estimated Aphy', color='red', linestyle='--', linewidth=linewidth)

    # Font properties for the legend
    font_properties = FontProperties(family='Times New Roman', size=fontsize - 4)

    # Labeling
    plt.xlabel('Wavelength (nm)', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel('Magnitude', fontsize=fontsize, fontname='Times New Roman')
    plt.title(f'Aphy Comparison for Data Point {idx}', fontsize=fontsize + 2, fontname='Times New Roman')
    plt.legend(prop=font_properties)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=fontsize - 4, fontname='Times New Roman')
    plt.yticks(fontsize=fontsize - 4, fontname='Times New Roman')

    # Show the plot
    plt.tight_layout()  # Ensure layout doesn't get cut off
    plt.show()


def plot_aphy_comparison2(wavelength_range, test_targets, test_outputs, test_outputs_map, idx, linewidth=2, fontsize=28, title_str=''):
    """
    Plot the ground truth Aphy and the estimated Aphy from field and PACE Rrs for a given data point.

    Parameters:
    - wavelength_range (range): The range of wavelengths corresponding to the data points.
    - test_targets (np.array): The ground truth Aphy values (shape: [num_data_points, num_wavelengths]).
    - test_outputs (np.array): The estimated Aphy values from field Rrs (shape: [num_data_points, num_wavelengths]).
    - test_outputs_map (np.array): The estimated Aphy values from PACE Rrs (shape: [num_data_points, num_wavelengths]).
    - idx (int): The index of the data point to plot.
    - linewidth (int): The line width for the plot lines. Default is 2.
    - fontsize (int): The font size for labels and title. Default is 28.
    """
    from matplotlib.lines import Line2D
    # Extract the specific target and output data for the given index
    aphy_ground_truth = test_targets[idx, :]
    aphy_estimated = test_outputs[idx, :]
    aphy_estimated_map = test_outputs_map[idx, :]

    # Plotting the data
    plt.figure(figsize=(12, 8))

    # Plot ground truth (Black)
    plt.plot(wavelength_range, aphy_ground_truth, label=r'Ground Truth', color='black', linewidth=linewidth)

    # Plot estimated Aphy from field Rrs (Red)
    plt.plot(wavelength_range, aphy_estimated, label=r'from Field $R_{rs}$', color='red',
             linestyle='--', linewidth=linewidth)

    # Plot estimated Aphy from PACE Rrs (Blue)
    plt.plot(wavelength_range, aphy_estimated_map, label=r'from PACE $R_{rs}$', color='blue',
             linestyle='--', linewidth=linewidth)
    # Font properties for the legend
    font_properties = FontProperties(family='Times New Roman', size=fontsize-6)

    # Labeling
    plt.xlabel('Wavelength (nm)', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$a_{phy}(m^{-1})$', fontsize=fontsize, fontname='Times New Roman')
    if title_str:  # If title_str is not empty
        plt.title(title_str, fontsize=fontsize, fontweight='bold', fontname='Times New Roman')
    else:
        plt.title(f'Aphy Comparison for Data Point {idx}', fontsize=fontsize + 2, fontname='Times New Roman')

    plt.legend(prop=font_properties)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=fontsize-6, fontname='Times New Roman')
    plt.yticks(fontsize=fontsize-6, fontname='Times New Roman')

    # Show the plot
    plt.tight_layout()  # Ensure layout doesn't get cut off
    return plt



def plot_aphy_comparison1(wavelength_range, test_targets, test_outputs, idx, linewidth=2, fontsize=28, title_str=''):
    """
    Plot the ground truth Aphy and the estimated Aphy from PACE Rrs for a given data point.

    Parameters:
    - wavelength_range (range): The range of wavelengths corresponding to the data points.
    - test_targets (np.array): The ground truth Aphy values (shape: [num_data_points, num_wavelengths]).
    - test_outputs (np.array): The estimated Aphy values from Rrs (shape: [num_data_points, num_wavelengths]).
    - idx (int): The index of the data point to plot.
    - linewidth (int): The line width for the plot lines. Default is 2.
    - fontsize (int): The font size for labels and title. Default is 28.
    """
    from matplotlib.lines import Line2D
    # Extract the specific target and output data for the given index
    aphy_ground_truth = test_targets[idx, :]
    aphy_estimated = test_outputs[idx, :]

    # Plotting the data
    plt.figure(figsize=(12, 8))

    # Plot ground truth (Black)
    plt.plot(wavelength_range, aphy_ground_truth, label=r'Ground Truth', color='black', linewidth=linewidth)

    # Plot estimated Aphy from field Rrs (Red)
    plt.plot(wavelength_range, aphy_estimated, label=r'Estimation from $R_{rs}$', color='red',
             linestyle='--', linewidth=linewidth)

    # Font properties for the legend
    font_properties = FontProperties(family='Times New Roman', size=fontsize-6)

    # Labeling
    plt.xlabel('Wavelength (nm)', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$a_{phy}(m^{-1})$', fontsize=fontsize, fontname='Times New Roman')
    if title_str:  # If title_str is not empty
        plt.title(title_str, fontsize=fontsize, fontweight='bold', fontname='Times New Roman')
    else:
        plt.title(f'Index {idx}', fontsize=fontsize + 2, fontname='Times New Roman')
    plt.legend(prop=font_properties)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=fontsize-6, fontname='Times New Roman')
    plt.yticks(fontsize=fontsize-6, fontname='Times New Roman')

    # Show the plot
    plt.tight_layout()  # Ensure layout doesn't get cut off
    return plt



def plot_aphy_comparison1_ID(wavelength_range, test_targets, test_outputs, idx,IDs, linewidth=2, fontsize=28, title_str=''):
    """
    Plot the ground truth Aphy and the estimated Aphy from PACE Rrs for a given data point.

    Parameters:
    - wavelength_range (range): The range of wavelengths corresponding to the data points.
    - test_targets (np.array): The ground truth Aphy values (shape: [num_data_points, num_wavelengths]).
    - test_outputs (np.array): The estimated Aphy values from Rrs (shape: [num_data_points, num_wavelengths]).
    - idx (int): The index of the data point to plot.
    - linewidth (int): The line width for the plot lines. Default is 2.
    - fontsize (int): The font size for labels and title. Default is 28.
    """
    from matplotlib.lines import Line2D
    # Extract the specific target and output data for the given index
    aphy_ground_truth = test_targets[idx, :]
    aphy_estimated = test_outputs[idx, :]

    # Plotting the data
    plt.figure(figsize=(12, 8))

    # Plot ground truth (Black)
    plt.plot(wavelength_range, aphy_ground_truth, label=r'Ground Truth', color='black', linewidth=linewidth)

    # Plot estimated Aphy from field Rrs (Red)
    plt.plot(wavelength_range, aphy_estimated, label=r'Estimation from $R_{rs}$', color='red',
             linestyle='--', linewidth=linewidth)

    # Font properties for the legend
    font_properties = FontProperties(family='Times New Roman', size=fontsize-6)

    # Labeling
    plt.xlabel('Wavelength (nm)', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$a_{phy}(m^{-1})$', fontsize=fontsize, fontname='Times New Roman')
    if title_str is not None:  # If title_str is not empty
        plt.title(f'Index {IDs[idx]}', fontsize=fontsize + 2, fontname='Times New Roman')
    plt.legend(prop=font_properties)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=fontsize-6, fontname='Times New Roman')
    plt.yticks(fontsize=fontsize-6, fontname='Times New Roman')

    # Show the plot
    plt.tight_layout()  # Ensure layout doesn't get cut off
    return plt




def plot_rrs_comparison(wavelength_range, test_input, test_input_map, idx, linewidth=2, fontsize=28, title_str=''):
    """
    Plot the ground truth Rrs and the estimated Rrs from field and PACE Rrs for a given data point.

    Parameters:
    - wavelength_range (range): The range of wavelengths corresponding to the data points.
    - test_input (np.array): The ground truth Rrs values (shape: [num_data_points, num_wavelengths]).
    - test_input_map (np.array): The estimated Rrs values from PACE Rrs (shape: [num_data_points, num_wavelengths]).
    - idx (int): The index of the data point to plot.
    - linewidth (int): The line width for the plot lines. Default is 2.
    - fontsize (int): The font size for labels and title. Default is 28.
    - title_str (str): The title string for the plot.
    """

    # Extract the specific input data for the given index
    rrs_ground_truth = test_input[idx, :]
    rrs_estimated_map = test_input_map[idx, :]

    # Plotting the data
    plt.figure(figsize=(12, 8))

    plt.plot(wavelength_range, rrs_ground_truth, label=r'Field', color='red', linewidth=linewidth)

    # Plot estimated Rrs from PACE (Blue)
    plt.plot(wavelength_range, rrs_estimated_map, label=r'PACE', color='blue',
             linestyle='--', linewidth=linewidth)

    # Font properties for the legend
    font_properties = FontProperties(family='Times New Roman', size=fontsize-6)

    # Labeling
    plt.xlabel('Wavelength (nm)', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$R_{rs}(sr^{-1})$', fontsize=fontsize, fontname='Times New Roman')

    if title_str:  # If title_str is not empty
        plt.title(title_str, fontsize=fontsize, fontweight='bold', fontname='Times New Roman')
    else:
        plt.title(f'Rrs Comparison for Data Point {idx}', fontsize=fontsize + 2, fontname='Times New Roman')

    plt.legend(prop=font_properties)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=fontsize-6, fontname='Times New Roman')
    plt.yticks(fontsize=fontsize-6, fontname='Times New Roman')

    # Show the plot
    plt.tight_layout()  # Ensure layout doesn't get cut off
    return plt

def plot_rrs_comparison1(wavelength_range, test_input, idx, linewidth=2, fontsize=28, title_str=''):
    """
    Plot the ground truth Rrs and the estimated Rrs from field and PACE Rrs for a given data point.

    Parameters:
    - wavelength_range (range): The range of wavelengths corresponding to the data points.
    - test_input (np.array): The ground truth Rrs values (shape: [num_data_points, num_wavelengths]).
    - idx (int): The index of the data point to plot.
    - linewidth (int): The line width for the plot lines. Default is 2.
    - fontsize (int): The font size for labels and title. Default is 28.
    - title_str (str): The title string for the plot.
    """

    # Extract the specific input data for the given index
    rrs_ground_truth = test_input[idx, :]

    # Plotting the data
    plt.figure(figsize=(12, 8))

    plt.plot(wavelength_range, rrs_ground_truth, label=r'Field', color='red', linewidth=linewidth)


    # Font properties for the legend
    font_properties = FontProperties(family='Times New Roman', size=fontsize-6)

    # Labeling
    plt.xlabel('Wavelength (nm)', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$R_{rs}(sr^{-1})$', fontsize=fontsize, fontname='Times New Roman')

    if title_str:  # If title_str is not empty
        plt.title(title_str, fontsize=fontsize, fontweight='bold', fontname='Times New Roman')
    else:
        plt.title(f'Rrs Comparison for Data Point {idx}', fontsize=fontsize + 2, fontname='Times New Roman')

    plt.legend(prop=font_properties)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=fontsize-6, fontname='Times New Roman')
    plt.yticks(fontsize=fontsize-6, fontname='Times New Roman')

    # Show the plot
    plt.tight_layout()  # Ensure layout doesn't get cut off
    return plt

def plot_scatter(y_true, y_pred, nrmse, mdsa, sspb, slope, wavelength, model_type, linewidth=2, fontsize=24, save_dir=None):
    """
    Plot a scatter plot with log-transformed actual and predicted values, along with a regression line, identity line, and KDE contours.

    Parameters:
    - y_true (np.array): Actual values before log transformation.
    - y_pred (np.array): Predicted values before log transformation.
    - nrmse (float): Precomputed Normalized Root Mean Squared Error.
    - mdsa (float): Precomputed Mean Difference Spectral Angle (in percentage).
    - sspb (float): Precomputed Sum of Squared Prediction Biases (in percentage).
    - slope (float): Precomputed slope of the regression line.
    - wavelength (int): The wavelength being analyzed, used in the plot title.
    - model_type (str): The model type (e.g., "MoE-MDN") used in the plot title.
    - linewidth (int, optional): The line width for the plot elements. Default is 2.
    - fontsize (int, optional): The font size for plot labels and title. Default is 24.
    """
    valid_mask = (y_true > 0) & (y_pred > 0)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    # Apply log transformation
    log_actual = np.log10(y_true)
    log_prediction = np.log10(y_pred)

    # Filter valid data points
    valid_mask = np.isfinite(log_actual) & np.isfinite(log_prediction)

    # Prepare data for the regression line
    slope_line, intercept = np.polyfit(log_actual[valid_mask], log_prediction[valid_mask], 1)
    x = np.array([-4, 2])
    y = slope_line * x + intercept

    plt.figure(figsize=(9, 9))  # Increase the figure size

    # Plot the regression line
    plt.plot(x, y, linestyle='--', color='red', linewidth=linewidth)  # Use adjustable line width

    # Plot the identity line (y=x)
    lims = [-4, 2]
    plt.plot(lims, lims, linestyle='-', color='black', linewidth=linewidth)  # Use adjustable line width

    # Scatter plot of the log-transformed actual vs predicted values
    sns.scatterplot(x=log_actual, y=log_prediction, alpha=1, s=140,color='blue')  # Keep dot size at s=70

    # KDE plot for the distribution of data points
    sns.kdeplot(x=log_actual[valid_mask], y=log_prediction[valid_mask], levels=3, color="black", fill=False,
                linewidths=linewidth-1)  # Use adjustable line width
    # Labeling
    # plt.xlabel(r'$a_{phy}$', fontsize=fontsize, fontname='Times New Roman')
    # plt.ylabel(r'$\hat{a}_{phy}$', fontsize=fontsize, fontname='Times New Roman')
    plt.xlabel(r'$\log_{10}(a_{phy})$', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$\log_{10}(\hat{a}_{phy})$', fontsize=fontsize, fontname='Times New Roman')

    # Set axis limits
    plt.xlim(-4, 2)
    plt.ylim(-4, 2)

    # # Adjust the axis ticks to be in powers of 10
    # plt.xticks(ticks=[-4, -3, -2, -1, 0, 1, 2],
    #            labels=[r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'],
    #            fontsize=fontsize-4, fontname='Times New Roman')  # Reduce font size for ticks
    # plt.yticks(ticks=[-4, -3, -2, -1, 0, 1, 2],
    #            labels=[r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'],
    #            fontsize=fontsize-4, fontname='Times New Roman')  # Reduce font size for ticks
    # Adjust the axis ticks to only show specific labels (-4, -2, 0, 2)
    # plt.xticks(ticks=[-4, -2, 0, 2],
    #            labels=[r'$10^{-4}$', r'$10^{-2}$', r'$10^{0}$', r'$10^{2}$'],
    #            fontsize=fontsize - 2, fontname='Times New Roman')
    #
    # plt.yticks(ticks=[-4, -2, 0, 2],
    #            labels=[r'$10^{-4}$', r'$10^{-2}$', r'$10^{0}$', r'$10^{2}$'],
    #            fontsize=fontsize - 2, fontname='Times New Roman')
    plt.xticks(ticks=[-4, -2, 0, 2],
               labels=[r'$-4$', r'$-2$', r'$0$', r'$2$'],
               fontsize=fontsize, fontname='Times New Roman')

    plt.yticks(ticks=[-4, -2, 0, 2],
               labels=[r'$-4$', r'$-2$', r'$0$', r'$2$'],
               fontsize=fontsize, fontname='Times New Roman')
    # Add grid
    plt.grid(True, which="both", ls="--")

    # Add legend with detailed metrics and larger font size
    # plt.legend(title=(f'NRMSE = {nrmse:.2f}, MDSA = {mdsa:.2f}%, \n '
    #                   f'SSPB = {sspb:.2f}%, Slope = {slope:.2f}'),
    #             title_fontsize=fontsize-6, prop={'family': 'Times New Roman'}, loc='upper left')
    # plt.legend(title=(f'NRMSE = {nrmse:.2f},  \n MDSA = {mdsa:.2f}%, \n '
    #                   f'SSPB = {sspb:.2f}%,  \n Slope = {slope:.2f}'),
    #             title_fontsize=fontsize-6, prop={'family': 'Times New Roman'}, loc='upper left',framealpha=0.2)
    # Add NRMSE and MDSA in the top left corner
    # plt.text(0.02, 0.98, f'NRMSE = {nrmse:.2f}\nMDSA = {mdsa:.2f}%',
    #          transform=plt.gca().transAxes, fontsize=fontsize-2, fontname='Times New Roman',
    #          verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    #
    # # Add SSPB and Slope in the bottom right corner
    # plt.text(0.98, 0.02, f'SSPB = {sspb:.2f}%\nSlope = {slope:.2f}',
    #          transform=plt.gca().transAxes, fontsize=fontsize-2, fontname='Times New Roman',
    #          verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.text(0.02, 0.98, f'NRMSE = {nrmse:.2f}\nε = {mdsa:.2f}%',
             transform=plt.gca().transAxes, fontsize=fontsize-6, fontname='Times New Roman',
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.text(0.98, 0.02, f'β = {sspb:.2f}%\nS = {slope:.2f}',
             transform=plt.gca().transAxes, fontsize=fontsize-6, fontname='Times New Roman',
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Set plot title including wavelength and model type
    plt.title(f'{model_type} - {wavelength} nm', fontsize=fontsize, fontweight='bold', fontname='Times New Roman')

    # Adjust layout to avoid text being cropped
    plt.tight_layout()

    # # Display the plot
    # plt.show()
    return plt


# def plot_scatter(y_true, y_pred, nrmse, mdsa, sspb, slope, wavelength, model_type):
#     """
#     Plot a scatter plot with log-transformed actual and predicted values, along with a regression line, identity line, and KDE contours.
#
#     Parameters:
#     - y_true (np.array): Actual values before log transformation.
#     - y_pred (np.array): Predicted values before log transformation.
#     - nrmse (float): Precomputed Normalized Root Mean Squared Error.
#     - mdsa (float): Precomputed Mean Difference Spectral Angle (in percentage).
#     - sspb (float): Precomputed Sum of Squared Prediction Biases (in percentage).
#     - slope (float): Precomputed slope of the regression line.
#     - wavelength (int): The wavelength being analyzed, used in the plot title.
#     - model_type (str): The model type (e.g., "MoE-MDN") used in the plot title.
#     """
#
#     # Apply log transformation
#     log_actual = np.log10(y_true)
#     log_prediction = np.log10(y_pred)
#
#     # Filter valid data points
#     valid_mask = np.isfinite(log_actual) & np.isfinite(log_prediction)
#
#     # Prepare data for the regression line
#     slope_line, intercept = np.polyfit(log_actual[valid_mask], log_prediction[valid_mask], 1)
#     x = np.array([-4, 2])
#     y = slope_line * x + intercept
#
#     plt.figure(figsize=(9, 9))  # Increase the figure size
#
#     # Plot the regression line
#     plt.plot(x, y, linestyle='--', color='blue', linewidth=2)  # Increase line width
#
#     # Plot the identity line (y=x)
#     lims = [-4, 2]
#     plt.plot(lims, lims, linestyle='-', color='black', linewidth=2)  # Increase line width
#
#     # Scatter plot of the log-transformed actual vs predicted values
#     sns.scatterplot(x=log_actual, y=log_prediction, alpha=0.6, s=70)  # Increase dot size with s=70
#
#     # KDE plot for the distribution of data points
#     sns.kdeplot(x=log_actual[valid_mask], y=log_prediction[valid_mask], levels=3, color="black", fill=False,
#                 linewidths=2)  # Increase line width
#
#     # Labeling
#     plt.xlabel(r'Actual $a_{phy}$ Values', fontsize=24, fontname='Times New Roman')
#     plt.ylabel(r'Predicted $a_{phy}$ Values', fontsize=24, fontname='Times New Roman')
#
#     # Set axis limits
#     plt.xlim(-4, 2)
#     plt.ylim(-4, 2)
#
#     # Adjust the axis ticks to be in powers of 10
#     plt.xticks(ticks=[-4, -3, -2, -1, 0, 1, 2],
#                labels=[r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'],
#                fontsize=20, fontname='Times New Roman')
#     plt.yticks(ticks=[-4, -3, -2, -1, 0, 1, 2],
#                labels=[r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'],
#                fontsize=20, fontname='Times New Roman')
#
#     # Add grid
#     plt.grid(True, which="both", ls="--")
#
#     # Add legend with detailed metrics and larger font size
#     plt.legend(title=(f'NRMSE = {nrmse:.2f}, MDSA = {mdsa:.2f}%, \n '
#                       f'SSPB = {sspb:.2f}%, Slope = {slope:.2f}'),
#                 title_fontsize=20, prop={'family': 'Times New Roman'}, loc='upper left')
#
#     # Set plot title including wavelength and model type
#     plt.title(f'{model_type} - {wavelength} nm', fontsize=24, fontweight='bold', fontname='Times New Roman')
#
#     # Adjust layout to avoid text being cropped
#     plt.tight_layout()
#
#     # Display the plot
#     plt.show()
    
def plot_scatter2(y_true, y_pred, nrmse, mdsa, sspb, slope, wavelength, model_type, linewidth=2, fontsize=24, save_dir=None):
    """
    Plot a scatter plot with log-transformed actual and predicted values, along with a regression line, identit y line, and KDE contours.
    """
    # Apply log transformation
    log_actual = np.log10(y_true)
    log_prediction = np.log10(y_pred)

    # Filter valid data points
    valid_mask = np.isfinite(log_actual) & np.isfinite(log_prediction)

    # Prepare data for the regression line
    slope_line, intercept = np.polyfit(log_actual[valid_mask], log_prediction[valid_mask], 1)
    x = np.array([-2, 1])
    y = slope_line * x + intercept

    plt.figure(figsize=(9, 9))  # Increase the figure size

    # Plot the regression line and add a label
    plt.plot(x, y, linestyle='--', color='red', linewidth=linewidth, label='Regression line')  # Use adjustable line width

    # Plot the identity line (y=x) and add a label
    lims = [-2, 1]
    plt.plot(lims, lims, linestyle='-', color='black', linewidth=linewidth, label='Identity line')  # Use adjustable line width

    # Scatter plot of the log-transformed actual vs predicted values, with a label
    sns.scatterplot(x=log_actual, y=log_prediction, alpha=0.6, s=70, label='Data points')  # Keep dot size at s=70

    # KDE plot for the distribution of data points
    sns.kdeplot(x=log_actual[valid_mask], y=log_prediction[valid_mask], levels=3, color="black", fill=False,
                linewidths=linewidth)  # Use adjustable line width

    # Labeling
    plt.xlabel(r'Actual $a_{phy}$ Values', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'Predicted $a_{phy}$ Values', fontsize=fontsize, fontname='Times New Roman')

    # Set axis limits
    plt.xlim(-2, 1)
    plt.ylim(-2, 1)

    # Adjust the axis ticks to only show specific labels (-2, 0, 1)
    plt.xticks(ticks=[-2, 0, 1],
               labels=[r'$10^{-2}$', r'$10^{0}$', r'$10^{1}$'],
               fontsize=fontsize - 8, fontname='Times New Roman')

    plt.yticks(ticks=[-2, 0, 1],
               labels=[r'$10^{-2}$', r'$10^{0}$', r'$10^{1}$'],
               fontsize=fontsize - 8, fontname='Times New Roman')

    # Add grid
    plt.grid(True, which="both", ls="--")

    # Add legend with custom metrics and labels from plot
    plt.legend(title=(f'NRMSE = {nrmse:.2f},  \n MDSA = {mdsa:.2f}%, \n '
                      f'SSPB = {sspb:.2f}%,  \n Slope = {slope:.2f}'),
               title_fontsize=fontsize-6, prop={'family': 'Times New Roman'}, loc='upper left', framealpha=0.2)

    # Set plot title including wavelength and model type
    plt.title(f'{model_type} - {wavelength} nm', fontsize=fontsize, fontweight='bold', fontname='Times New Roman')

    # Adjust layout to avoid text being cropped
    plt.tight_layout()
    
    plt.show()
    return plt

def plot_scatter3(y_true, y_pred, wavelengths, nrmse, mdsa, sspb, slope,linewidth=2, fontsize=24, save_dir=None, title_str=''):
    """
    Plot a scatter plot with log-transformed actual and predicted values for all wavelengths, 
    coloring the points based on their wavelength using a 'jet' colormap.

    Parameters:
    - y_true (np.array): Actual values before log transformation (2D array: samples x wavelengths).
    - y_pred (np.array): Predicted values before log transformation (2D array: samples x wavelengths).
    - wavelengths (list): List of wavelength values (1D array).
    - linewidth (int, optional): The line width for the plot elements. Default is 2.
    - fontsize (int, optional): The font size for plot labels and title. Default is 24.
    """

    # Flatten the data (assumes y_true and y_pred are 2D arrays)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    wavelengths_flat = np.tile(wavelengths, y_true.shape[0])
    valid_mask = (y_true_flat > 0) & (y_pred_flat > 0)
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]
    # Apply log transformation
    log_actual = np.log10(y_true_flat)
    log_prediction = np.log10(y_pred_flat)

    # Filter valid data points (in case of invalid values after log transformation)
    valid_mask = np.isfinite(log_actual) & np.isfinite(log_prediction)
    # Prepare color mapping using the jet colormap
    cmap = cm.get_cmap('jet')
    norm = plt.Normalize(400, 700)  # Normalize for wavelength range from 400 to 700
    colors = cmap(norm(wavelengths_flat[valid_mask]))

    plt.figure(figsize=(10, 7.3))  # Increase the figure size

    # Plot the identity line (y=x)
    lims = [-2, 0]
    plt.plot(lims, lims, linestyle='-', color='black', linewidth=linewidth)  # Use adjustable line width

    # Scatter plot with color based on wavelengths
    plt.scatter(log_actual[valid_mask], log_prediction[valid_mask], c=colors, alpha=0.8, s=140)

    # Labeling
    # plt.xlabel(r'Actual $a_{phy}$ Values', fontsize=fontsize+2, fontname='Times New Roman')
    # plt.ylabel(r'Predicted $a_{phy}$ Values', fontsize=fontsize+2, fontname='Times New Roman')
    plt.xlabel(r'$\log_{10}(a_{phy})$', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$\log_{10}(\hat{a}_{phy})$', fontsize=fontsize, fontname='Times New Roman')
    # Set axis limits based on the given range (-2, -1, 0)
    plt.xlim(-2, 0)
    plt.ylim(-2, 0)

    # # Adjust the axis ticks to match the new limits (-2, -1, 0)
    # plt.xticks(ticks=[-2, -1, 0],
    #            labels=[r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'],
    #            fontsize=fontsize - 6, fontname='Times New Roman')
    # 
    # plt.yticks(ticks=[-2, -1, 0],
    #            labels=[r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'],
    #            fontsize=fontsize - 6, fontname='Times New Roman')
    # Adjust the axis ticks to match the new limits (-2, -1, 0)
    plt.xticks(ticks=[-2, -1, 0],
               labels=[r'$-2$', r'$-1$', r'$0$'],
               fontsize=fontsize-4, fontname='Times New Roman')

    plt.yticks(ticks=[-2, -1, 0],
               labels=[r'$-2$', r'$-1$', r'$0$'],
               fontsize=fontsize-4, fontname='Times New Roman')
    # Add grid
    plt.grid(True, which="both", ls="--")
    # plt.legend(title=(f'NRMSE = {nrmse:.2f},  \nMDSA = {mdsa:.2f}%, \n'
    #                   f'SSPB = {sspb:.2f}%,  \nSlope = {slope:.2f}'),
    #            title_fontsize=fontsize-6, prop={'family': 'Times New Roman'}, loc='upper left', framealpha=0.2)

    plt.text(0.02, 0.98, f'NRMSE = {nrmse:.2f}\nε = {mdsa:.2f}%',
             transform=plt.gca().transAxes, fontsize=fontsize-2, fontname='Times New Roman',
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.text(0.98, 0.02, f'β = {sspb:.2f}%\nS = {slope:.2f}',
             transform=plt.gca().transAxes, fontsize=fontsize-2, fontname='Times New Roman',
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add colorbar for the wavelength color mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need this to make ScalarMappable work
    cbar = plt.colorbar(sm)
    cbar.set_label('Wavelength (nm)', fontsize=fontsize - 4)
    cbar.ax.tick_params(labelsize=fontsize-4)
    # Set plot title
    # plt.title('Predicted vs Actual $a_{phy}$ Values Across Wavelengths', fontsize=fontsize, fontweight='bold', fontname='Times New Roman')
    if title_str:  # If title_str is not empty
        plt.title(title_str, fontsize=fontsize, fontweight='bold', fontname='Times New Roman')

    # Adjust layout to avoid text being cropped
    plt.tight_layout()

    return plt



def plot_scatter3_Rrs(y_true, y_pred, wavelengths, nrmse, mdsa, sspb, slope,linewidth=2, fontsize=24, save_dir=None, title_str=''):
    """
    Plot a scatter plot with log-transformed actual and predicted values for all wavelengths,
    coloring the points based on their wavelength using a 'jet' colormap.

    Parameters:
    - y_true (np.array): Actual values before log transformation (2D array: samples x wavelengths).
    - y_pred (np.array): Predicted values before log transformation (2D array: samples x wavelengths).
    - wavelengths (list): List of wavelength values (1D array).
    - linewidth (int, optional): The line width for the plot elements. Default is 2.
    - fontsize (int, optional): The font size for plot labels and title. Default is 24.
    """

    # Flatten the data
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    wavelengths_flat = np.tile(wavelengths, y_true.shape[0])  # (13 * 144 = 1872)

    # First valid mask: where both y_true and y_pred are positive
    valid_mask = (y_true_flat > 0) & (y_pred_flat > 0)

    # Apply the mask to all three arrays
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    wavelengths_valid = wavelengths_flat[valid_mask]

    # Apply log transformation
    log_actual = np.log10(y_true_valid)
    log_prediction = np.log10(y_pred_valid)

    # Second valid mask: filter out NaNs or -infs
    final_mask = np.isfinite(log_actual) & np.isfinite(log_prediction)

    # Final filtered data
    log_actual = log_actual[final_mask]
    log_prediction = log_prediction[final_mask]
    wavelengths_final = wavelengths_valid[final_mask]

    # Prepare color mapping
    cmap = cm.get_cmap('jet')
    norm = plt.Normalize(400, 700)
    colors = cmap(norm(wavelengths_final))

    plt.figure(figsize=(10, 7.3))  # Increase the figure size

    # Plot the identity line (y=x)
    lims = [-5, -1]
    plt.plot(lims, lims, linestyle='-', color='black', linewidth=linewidth)  # Use adjustable line width

    # Scatter plot with color based on wavelengths
    plt.scatter(log_actual, log_prediction, c=colors, alpha=0.8, s=140)

    # Labeling
    # plt.xlabel(r'Actual $a_{phy}$ Values', fontsize=fontsize+2, fontname='Times New Roman')
    # plt.ylabel(r'Predicted $a_{phy}$ Values', fontsize=fontsize+2, fontname='Times New Roman')
    plt.xlabel(r'$\log_{10}(R_{rs})$', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'$\log_{10}(\hat{R}_{rs})$', fontsize=fontsize, fontname='Times New Roman')
    # Set axis limits based on the given range (-2, -1, 0)
    plt.xlim(-5, -1)
    plt.ylim(-5, -1)

    # # Adjust the axis ticks to match the new limits (-2, -1, 0)
    # plt.xticks(ticks=[-2, -1, 0],
    #            labels=[r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'],
    #            fontsize=fontsize - 6, fontname='Times New Roman')
    #
    # plt.yticks(ticks=[-2, -1, 0],
    #            labels=[r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'],
    #            fontsize=fontsize - 6, fontname='Times New Roman')
    # Adjust the axis ticks to match the new limits (-2, -1, 0)
    plt.xticks(ticks=[-5,-4, -3,-2, -1],
               labels=[r'$-5$',r'$-4$',r'$-3$', r'$-2$', r'$-1$'],
               fontsize=fontsize-4, fontname='Times New Roman')

    plt.yticks(ticks=[-5,-4, -3,-2, -1],
               labels=[r'$-5$',r'$-4$',r'$-3$', r'$-2$', r'$-1$'],
               fontsize=fontsize-4, fontname='Times New Roman')
    # Add grid
    plt.grid(True, which="both", ls="--")
    # plt.legend(title=(f'NRMSE = {nrmse:.2f},  \nMDSA = {mdsa:.2f}%, \n'
    #                   f'SSPB = {sspb:.2f}%,  \nSlope = {slope:.2f}'),
    #            title_fontsize=fontsize-6, prop={'family': 'Times New Roman'}, loc='upper left', framealpha=0.2)

    plt.text(0.02, 0.98, f'NRMSE = {nrmse:.2f}\nε = {mdsa:.2f}%',
             transform=plt.gca().transAxes, fontsize=fontsize-2, fontname='Times New Roman',
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.text(0.98, 0.02, f'β = {sspb:.2f}%\nS = {slope:.2f}',
             transform=plt.gca().transAxes, fontsize=fontsize-2, fontname='Times New Roman',
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add colorbar for the wavelength color mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need this to make ScalarMappable work
    cbar = plt.colorbar(sm)
    cbar.set_label('Wavelength (nm)', fontsize=fontsize - 4)
    cbar.ax.tick_params(labelsize=fontsize-4)
    # Set plot title
    # plt.title('Predicted vs Actual $a_{phy}$ Values Across Wavelengths', fontsize=fontsize, fontweight='bold', fontname='Times New Roman')
    if title_str:  # If title_str is not empty
        plt.title(title_str, fontsize=fontsize, fontweight='bold', fontname='Times New Roman')

    # Adjust layout to avoid text being cropped
    plt.tight_layout()

    return plt



def plot_scatter4(y_true, y_pred, wavelengths, linewidth=2, fontsize=24, save_dir=None):
    """
    Plot a scatter plot with actual and predicted values for all wavelengths, 
    coloring the points based on their wavelength using a 'jet' colormap.

    Parameters:
    - y_true (np.array): Actual values (2D array: samples x wavelengths).
    - y_pred (np.array): Predicted values (2D array: samples x wavelengths).
    - wavelengths (list): List of wavelength values (1D array).
    - linewidth (int, optional): The line width for the plot elements. Default is 2.
    - fontsize (int, optional): The font size for plot labels and title. Default is 24.
    """

    # Flatten the data (assumes y_true and y_pred are 2D arrays)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    wavelengths_flat = np.tile(wavelengths, y_true.shape[0])

    # Filter valid data points (in case of invalid values)
    valid_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)

    # Prepare color mapping using the jet colormap
    cmap = cm.get_cmap('jet')
    norm = plt.Normalize(400, 700)  # Normalize for wavelength range from 400 to 700
    colors = cmap(norm(wavelengths_flat[valid_mask]))

    plt.figure(figsize=(9, 9))  # Increase the figure size

    # Plot the identity line (y=x)
    lims = [0, 0.3]  # Based on the range of y_true and y_pred
    plt.plot(lims, lims, linestyle='-', color='black', linewidth=linewidth)  # Use adjustable line width

    # Scatter plot with color based on wavelengths
    plt.scatter(y_true_flat[valid_mask], y_pred_flat[valid_mask], c=colors, alpha=0.6, s=70)

    # Labeling
    plt.xlabel(r'Actual $a_{phy}$ Values', fontsize=fontsize, fontname='Times New Roman')
    plt.ylabel(r'Predicted $a_{phy}$ Values', fontsize=fontsize, fontname='Times New Roman')

    # Set axis limits based on the given range (0 to 0.3)
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.3)

    # Adjust the axis ticks to match the new limits
    plt.xticks(fontsize=fontsize - 8, fontname='Times New Roman')
    plt.yticks(fontsize=fontsize - 8, fontname='Times New Roman')

    # Add grid
    plt.grid(True, which="both", ls="--")

    # Add colorbar for the wavelength color mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need this to make ScalarMappable work
    cbar = plt.colorbar(sm)
    cbar.set_label('Wavelength (nm)', fontsize=fontsize - 4)

    # Set plot title
    plt.title('Predicted vs Actual $a_{phy}$ Values Across Wavelengths', fontsize=fontsize, fontweight='bold', fontname='Times New Roman')

    # Adjust layout to avoid text being cropped
    plt.tight_layout()

    return plt
