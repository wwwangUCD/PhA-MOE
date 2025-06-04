# Plotting Utilities Documentation

This document summarizes the plotting functions defined in `utils/plots.py`.

---
## Function: `plot_metrics`

**Purpose:**  
Creates a bar plot for a metric (e.g., error or correlation) across wavelengths using RGB-colored bars.

**Notes:**  
- Displays the plot directly using `plt.show()`.
- Does not return the figure.


## Function: `plot_metrics_new`

**Purpose:**  
Improved version of `plot_metrics` with better customization and return support.

**Differences from `plot_metrics`:**
- Returns the figure object (`fig`) instead of displaying it.
- Supports customizable `fontsize`.
- Uses the `ax`/`fig` object-oriented interface instead of the `plt` global interface.


## Function: `plot_aphy_comparison`

**Purpose:**  
Plots the ground truth and estimated Aphy for a single data point.

**Notes:**  
- Only compares two curves: ground truth and estimation.
- Uses default label formatting without LaTeX-style notation.


## Function: `plot_aphy_comparison1`

**Purpose:**  
Plots the ground truth and estimated Aphy (from Rrs) for a given index.

**Differences from `plot_aphy_comparison`:**
- Uses LaTeX-style labels (e.g., `$R_{rs}$`).
- Adds optional `title_str` argument for flexible titles.
- Plot styling and fonts are slightly more polished.


## Function: `plot_aphy_comparison1_ID`

**Purpose:**  
Same as `plot_aphy_comparison1`, but adds sample ID labeling in the title.

**Differences from `plot_aphy_comparison1`:**
- Adds `IDs` argument to show a human-readable ID instead of numeric index.


## Function: `plot_aphy_comparison2`

**Purpose:**  
Plots three curves: ground truth, estimated Aphy from **field** Rrs, and estimated Aphy from **PACE** Rrs.

**Differences from `plot_aphy_comparison1`:**
- Adds `test_outputs_map` as a third curve.
- Labeling distinguishes "Field" and "PACE" estimates.
- Intended for direct comparison of two sources of estimated Aphy.


## Function: `plot_rrs_comparison`

**Purpose:**  
Plots ground truth Rrs and estimated Rrs from PACE for a given data point.

**Notes:**  
- Compares two curves: field Rrs vs. PACE-estimated Rrs.
- Used to evaluate spectral differences between field measurements and PACE retrievals.


## Function: `plot_rrs_comparison1`

**Purpose:**  
Plots only the field Rrs for a given data point.

**Differences from `plot_rrs_comparison`:**
- Only displays the field Rrs (no comparison with PACE).
- Useful for visualizing individual spectra without overlaying prediction.

## Function: `plot_scatter`

**Purpose:**  
Plots a log-log scatter plot of predicted vs. actual \( a_{phy} \), including regression and identity lines, KDE contours, and annotated metrics.

**Notes:**
- Applies log10 transformation.
- Annotates NRMSE, MDSA (ε), SSPB (β), and slope (S).
- Uses fixed axis limits and tick formatting.


## Function: `plot_scatter2`

**Purpose:**  
A simplified version of `plot_scatter` with basic labeling, contouring, and embedded metric annotation.

**Differences from `plot_scatter`:**
- Simpler layout and axis formatting.
- Uses smaller axis range (`[-2, 1]`).
- Shows legend box instead of text-annotated corners.


## Function: `plot_scatter3`

**Purpose:**  
Scatter plot of predicted vs. actual \( a_{phy} \) with log10 values, where each point is colored by wavelength.

**Differences from `plot_scatter`:**
- Supports multiband input and uses wavelength-based coloring.
- Adds colorbar indicating wavelengths.
- Used for global spectral evaluation across all bands.


## Function: `plot_scatter3_Rrs`

**Purpose:**  
Same as `plot_scatter3`, but used for evaluating \( R_{rs} \) predictions instead of \( a_{phy} \).

**Differences from `plot_scatter3`:**
- Axis labels are for \( R_{rs} \) rather than \( a_{phy} \).
- Log range is extended to `[-5, -1]` for lower signal values.


## Function: `plot_scatter4`

**Purpose:**  
Scatter plot of predicted vs. actual values without log transformation, colored by wavelength.

**Differences from `plot_scatter3`:**
- Uses raw values instead of log10.
- Suitable for high-SNR or low-dynamic-range data like filtered results.
- Uses fixed linear axes and raw \( a_{phy} \) values.

