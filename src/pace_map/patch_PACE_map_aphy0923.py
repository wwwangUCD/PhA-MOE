"""
code to plot the patched map for aphy
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
bas_dir2=r'D:\Research\HyperCoast\PACEmap\4Seasons\npyData'



# date='0425'
# filepath="PACE_OCI.20240425T181836.L2.OC_AOP.V2_0.NRT.nc"
# date='0607'
# filepath="PACE_OCI.20240607T184245.L2.OC_AOP.V2_0.NRT.nc"
date='0923'
filepath="PACE_OCI.20240923T184221.L2.OC_AOP.V2_0.NRT.nc"
#
# date='0929'
# filepath="PACE_OCI.20240929T185124.L2.OC_AOP.V2_0.NRT.nc"

# date='1229'
# filepath="PACE_OCI.20241229T183314.L2.OC_AOP.V2_0.NRT.nc"

tiff_file = os.path.join(bas_dir, "20240415_RGB.tif")
save_dir='D:\Research\EnvironmentalData\MyPapers\MoE_letter\RemoteSensing\Map'
# Open the TIFF file
with rasterio.open(tiff_file) as src:
    img_rgb = src.read()  # Reads the TIFF file into a NumPy array

rgb_image = img_rgb[:3, :, :]

# Transpose to shape (2908, 4177, 3) for plotting
rgb_image = np.transpose(rgb_image, (1, 2, 0))

# rgb_image_normalized = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image)) * 255
#
# # Convert to an integer type for proper plotting
# rgb_image_normalized = rgb_image_normalized.astype(np.uint8)
#
# # Plot the normalized RGB image
# plt.figure(figsize=(10, 10))
# plt.imshow(rgb_image_normalized)
# plt.title('Normalized RGB Image (0-255)')
# plt.axis('off')
# plt.show()


dataset = hypercoast.read_pace(os.path.join(bas_dir,filepath))
da = dataset["Rrs"]

wl =da.wavelength.values
Rrs=da.values
latitude =da.latitude.values
longitude =da.longitude.values

indices = np.where((wl >= 401) & (wl <= 699))[0]
filtered_Rrs = Rrs[:, :, indices]

# Filtered wl
filtered_wl = wl[indices]

# aphy_PACE=np.load('aphy_PACE_0923_nft.npy') # 'nft' means no fine tunning
aphy_PACE=np.load(os.path.join(bas_dir2,f'Aphy_All_{date}.npy')) # fig 4 in paper is the map on fine tunned model
mask=np.load(os.path.join(bas_dir2,f'Rrs_All_nan_mask_{date}.npy'))



selected_wavelengths = [440, 620, 673]
selected_indices = [np.argmin(np.abs(filtered_wl - wl)) for wl in selected_wavelengths]
nearest_selected_wavelengths = np.rint(filtered_wl[selected_indices]).astype(int)

selected_aphy_PACE=aphy_PACE[:,:,selected_indices]

valid_selected_aphy_PACE = selected_aphy_PACE[mask == 1]


vmin=0
vmax=0.5

rgb_height, rgb_width, _ = rgb_image.shape

# Define the latitude and longitude bounds for the RGB image based on your data
lon_min, lon_max = -90.6697, -89.5225
lat_min, lat_max = 29.7754, 30.5740

rgb_lons = np.linspace(lon_min, lon_max, rgb_width)
rgb_lats = np.linspace( lat_max,lat_min, rgb_height)

rgb_lon_grid, rgb_lat_grid = np.meshgrid(rgb_lons, rgb_lats)

def plot_channel(data_channel, longitude, latitude, rgb_image, rgb_lon_grid, rgb_lat_grid, vmin=0, vmax=0.2, wl=440,
                 fontsize=24,cmap_str='jet'):
    """
    Function to interpolate and plot a data channel (reflectance or other) on top of an RGB image.

    Args:
        data_channel: 2D array (height, width) of the data to be plotted (e.g., reflectance data).
        longitude: Flattened array of the longitudes corresponding to data points.
        latitude: Flattened array of the latitudes corresponding to data points.
        rgb_image: The RGB image to use as background for patching NaN values.
        rgb_lon_grid: 2D array of longitudes for the RGB image grid.
        rgb_lat_grid: 2D array of latitudes for the RGB image grid.
        vmin: Minimum value for normalization of the data.
        vmax: Maximum value for normalization of the data.
        wl: Wavelength value to be used in the plot title.
        fontsize: Font size for the plot labels.

    Returns:
        None
    """
    # Flatten the data for interpolation
    data_flat = data_channel.flatten()
    lon_flat = longitude.flatten()
    lat_flat = latitude.flatten()

    # Interpolate data to the grid defined by the RGB image
    reflectance_interpolated = griddata(
        (lon_flat, lat_flat),  # Original (longitude, latitude) pairs
        data_flat,  # Corresponding reflectance values
        (rgb_lon_grid, rgb_lat_grid),  # Target grid for interpolation
        method='nearest'  # Choose 'nearest' or 'linear' depending on data needs
    )

    # Handle NaN values and apply colormap
    mask = np.isnan(reflectance_interpolated).astype(int)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_str)

    # Create RGB representation of the reflectance values
    reflectance_rgb = cmap(norm(reflectance_interpolated))[:, :, :3]  # Ignore the alpha channel
    reflectance_rgb = (reflectance_rgb * 255).astype(np.uint8)

    # Patch NaN values with the original RGB image
    patched_image = np.where(mask[:, :, None] == 1, rgb_image, reflectance_rgb)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10.5, 7.3))
    # Display the patched image with geographic extent
    im = ax.imshow(patched_image, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')

    # Set title with wavelength and fontsize
    ax.set_title(r'$a_{phy}$ map, wavelength=' + f'{wl}nm', fontsize=fontsize)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add color bar for the reflectance
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Adjust the color bar to match the image height
    # cbar = fig.colorbar(sm, ax=ax)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'$a_{phy}$', fontsize=fontsize)

    # Set axis labels for latitude and longitude
    ax.set_xlabel('Longitude', fontsize=fontsize)
    ax.set_ylabel('Latitude', fontsize=fontsize)

    # Set tick parameters for the axes and color bar
    ax.tick_params(axis='both', labelsize=fontsize - 2)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    # Show the plot
    # plt.show()
    return fig

fig1=plot_channel(
    data_channel=selected_aphy_PACE[:, :, 0],  # Your data channel
    longitude=longitude,                      # Flattened longitudes
    latitude=latitude,                        # Flattened latitudes
    rgb_image=rgb_image,                      # Background RGB image
    rgb_lon_grid=rgb_lon_grid,                # Longitude grid for the RGB image
    rgb_lat_grid=rgb_lat_grid,                # Latitude grid for the RGB image
    vmin=0,                                   # Minimum reflectance value
    vmax=0.5,                                 # Maximum reflectance value
    wl=440,                                   # Wavelength value for the title
    fontsize=28                               # Font size for the plot labels
)
file_name = f"PACE_aphy_440nm_{date}.pdf"
save_path = os.path.join(save_dir, file_name)
fig1.savefig(save_path, dpi=300, bbox_inches='tight')

fig2=plot_channel(
    data_channel=selected_aphy_PACE[:, :, 1],  # Your data channel
    longitude=longitude,                      # Flattened longitudes
    latitude=latitude,                        # Flattened latitudes
    rgb_image=rgb_image,                      # Background RGB image
    rgb_lon_grid=rgb_lon_grid,                # Longitude grid for the RGB image
    rgb_lat_grid=rgb_lat_grid,                # Latitude grid for the RGB image
    vmin=0,                                   # Minimum reflectance value
    vmax=0.5,                                 # Maximum reflectance value
    wl=620,                                   # Wavelength value for the title
    fontsize=28                               # Font size for the plot labels
)
file_name = f"PACE_aphy_620nm_{date}.pdf"
save_path = os.path.join(save_dir, file_name)
fig2.savefig(save_path, dpi=300, bbox_inches='tight')

fig3=plot_channel(
    data_channel=selected_aphy_PACE[:, :, 2],  # Your data channel
    longitude=longitude,                      # Flattened longitudes
    latitude=latitude,                        # Flattened latitudes
    rgb_image=rgb_image,                      # Background RGB image
    rgb_lon_grid=rgb_lon_grid,                # Longitude grid for the RGB image
    rgb_lat_grid=rgb_lat_grid,                # Latitude grid for the RGB image
    vmin=0,                                   # Minimum reflectance value
    vmax=0.5,                                 # Maximum reflectance value
    wl=673,                                   # Wavelength value for the title
    fontsize=28                               # Font size for the plot labels
)

file_name = f"PACE_aphy_673nm_{date}.pdf"
save_path = os.path.join(save_dir, file_name)
fig3.savefig(save_path, dpi=300, bbox_inches='tight')
