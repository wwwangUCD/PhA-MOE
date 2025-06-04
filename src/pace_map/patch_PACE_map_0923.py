"""
code to plot the patched map for Rrs
"""
import hypercoast
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import rasterio
from PIL import Image
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
bas_dir=r'D:\Research\HyperCoast\PACEmap\4Seasons'

# filepath = "PACE_OCI.20240730T181157.L2.OC_AOP.V2_0.NRT.nc"
# filepath="PACE_OCI.20240515T182840.L2.OC_AOP.V2_0.NRT.nc"
filepath="PACE_OCI.20240923T184221.L2.OC_AOP.V2_0.NRT.nc"
bas_dir2=r'D:\Research\HyperCoast\PACEmap\4Seasons'

tiff_file = os.path.join(bas_dir, "20240415_RGB.tif")

# Open the TIFF file
with rasterio.open(tiff_file) as src:
    img_rgb = src.read()  # Reads the TIFF file into a NumPy array

rgb_image = img_rgb[:3, :, :]

# Transpose to shape (2908, 4177, 3) for plotting
rgb_image = np.transpose(rgb_image, (1, 2, 0))
rgb_image_normalized = rgb_image / 255.0
dataset = hypercoast.read_pace(os.path.join(bas_dir2,filepath))
da = dataset["Rrs"]

wl =da.wavelength.values
Rrs=da.values
latitude =da.latitude.values
longitude =da.longitude.values

indices = np.where((wl >= 401) & (wl <= 699))[0]
filtered_Rrs = Rrs[:, :, indices]

# Filtered wl
filtered_wl = wl[indices]



rgb_wavelengths = [650, 560, 400]

# Find the indices of the wavelengths closest to 650, 560, and 400 nm
rgb_indices = [np.argmin(np.abs(filtered_wl - wl)) for wl in rgb_wavelengths]

# Extract the R, G, and B channels from the filtered Rrs data
R_channel = filtered_Rrs[:, :, rgb_indices[0]]
G_channel = filtered_Rrs[:, :, rgb_indices[1]]
B_channel = filtered_Rrs[:, :, rgb_indices[2]]

# Normalize the R, G, and B channels between vmin and vmax
vmin, vmax = 0, 0.02
R_channel = np.clip(R_channel, vmin, vmax)
G_channel = np.clip(G_channel, vmin, vmax)
B_channel = np.clip(B_channel, vmin, vmax)

# Normalize each channel to 0-1 range for plotting
R_channel = (R_channel - vmin) / (vmax - vmin)
G_channel = (G_channel - vmin) / (vmax - vmin)
B_channel = (B_channel - vmin) / (vmax - vmin)

# Stack the R, G, and B channels to create the rgb_Rrs image
rgb_Rrs = np.stack([R_channel, G_channel, B_channel], axis=-1)

rgb_height, rgb_width, _ = rgb_image_normalized.shape

lon_min, lon_max = -90.6697, -89.5225
lat_min, lat_max = 29.7754, 30.5740

rgb_lons = np.linspace(lon_min, lon_max, rgb_width)
rgb_lats = np.linspace( lat_max,lat_min, rgb_height)

rgb_lon_grid, rgb_lat_grid = np.meshgrid(rgb_lons, rgb_lats)


# Step 4: Update the plot_channel function to handle the patching inside it
def plot_channel(rgb_image_normalized, rgb_Rrs, longitude, latitude,
                 rgb_lon_grid, rgb_lat_grid,
                 rgb_wavelengths=(443, 555, 670),
                 vmin=0, vmax=0.2, fontsize=24):
    """
    Interpolates the Rrs RGB composite onto the original RGB image grid and returns a consistent styled figure.
    """
    # Flatten RGB bands and coordinates
    R_flat = rgb_Rrs[:, :, 0].flatten()
    G_flat = rgb_Rrs[:, :, 1].flatten()
    B_flat = rgb_Rrs[:, :, 2].flatten()

    lon_flat = longitude.flatten()
    lat_flat = latitude.flatten()

    # Interpolate R, G, B channels
    R_interpolated = griddata((lon_flat, lat_flat), R_flat, (rgb_lon_grid, rgb_lat_grid), method='nearest')
    G_interpolated = griddata((lon_flat, lat_flat), G_flat, (rgb_lon_grid, rgb_lat_grid), method='nearest')
    B_interpolated = griddata((lon_flat, lat_flat), B_flat, (rgb_lon_grid, rgb_lat_grid), method='nearest')

    # Combine channels
    rgb_Rrs_interpolated = np.stack([R_interpolated, G_interpolated, B_interpolated], axis=-1)

    # Patch missing values with the original RGB background
    mask = np.isnan(R_interpolated).astype(int)
    patched_image = np.where(mask[:, :, None] == 1, rgb_image_normalized, rgb_Rrs_interpolated)

    # Plot
    fig = plt.figure(figsize=(10.5, 7.3))
    ax = fig.add_subplot(111)
    im = ax.imshow(patched_image, extent=[
        rgb_lon_grid.min(), rgb_lon_grid.max(),
        rgb_lat_grid.min(), rgb_lat_grid.max()
    ], origin='upper')

    # Title and labels
    # ax.set_title(
    #     rf'$R_{{rs}}$ map (R: {rgb_wavelengths[0]}nm, G: {rgb_wavelengths[1]}nm, B: {rgb_wavelengths[2]}nm)',
    #     fontsize=fontsize
    # )
    ax.set_xlabel('Longitude', fontsize=fontsize)
    ax.set_ylabel('Latitude', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize - 2)

    return fig

plt = plot_channel(
    rgb_image_normalized=rgb_image_normalized,  # The original RGB image
    rgb_Rrs=rgb_Rrs,  # The RGB image created from Rrs data
    longitude=longitude,  # Longitudes for the Rrs data
    latitude=latitude,  # Latitudes for the Rrs data
    rgb_lon_grid=rgb_lon_grid,  # Longitude grid for the RGB image
    rgb_lat_grid=rgb_lat_grid,  # Latitude grid for the RGB image
    vmin=0,  # Minimum reflectance value (not relevant here, can be ignored)
    vmax=0.02,  # Maximum reflectance value (not relevant here, can be ignored)
    fontsize=28  # Font size for the plot labels
)
save_path = r'D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\Map2\PACE_rrs.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
