import hypercoast
import numpy as np
import os
from scipy.interpolate import griddata
bas_dir=f'D:\Research\HyperCoast\PACEmap\PACEdata-20241203T182106Z-001\PACEdata'

filepath="PACE_OCI.20240923T184221.L2.OC_AOP.V2_0.NRT.nc"
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

# stations used to evaluate the finetuned model
# # Station list for eval_data.npy
# station_list = ['C4', 'Hyper2', 'Hyper3', 'LP8', 'LP10', 'N2', 'T12']
# Corresponding latitude and longitude pairs
# specific_lat_lon_pairs = np.array([
#     [30.201519, -90.134024],
#     [30.153197, -90.220664],
#     [30.341124, -90.179677],
#     [30.105944, -90.318889],
#     [30.137639, -90.263056],
#     [30.296599, -90.277498],
#     [30.096039, -90.346776]
# ])

# Station list for ft_data.npy
station_list = ['C7', 'Hyper4', 'LP9', 'LP11']
specific_lat_lon_pairs = np.array([
    [30.07,	-90.158431],
    [30.142291,	-89.999967],
    [30.073828,	-90.287775],
    [30.178806,	-90.19725]
])


lat_flat = latitude.flatten()  # Your flattened latitude array (shape: [n,])
lon_flat = longitude.flatten()  # Your flattened longitude array (shape: [n,])
Rrs_flat = filtered_Rrs.reshape(-1, filtered_Rrs.shape[2])  # Shape: [n, 144], flatten Rrs keeping bands

Rrs_interpolated = griddata(
    (lon_flat, lat_flat), Rrs_flat,  # Input: original coordinates and values
    (specific_lat_lon_pairs[:, 1], specific_lat_lon_pairs[:, 0]),  # Target: lon, lat pairs
    method='nearest'  # Can also try 'linear' or 'cubic' depending on data
)

print(Rrs_interpolated.shape)
Rrs_interpolated_np = np.array(Rrs_interpolated)

save_path = r'D:\Research\HyperCoast\PACEmap\PACEdata-20241203T182106Z-001\PACEdata\PACEmap_Dec2_ft\Rrs_Stations_0923.npy'
np.save(save_path, Rrs_interpolated_np)
print(f"Shape of Rrs is {Rrs_interpolated_np.shape}")
print(f"Array saved to {save_path}")
