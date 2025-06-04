import hypercoast
import numpy as np
import os
from scipy.interpolate import griddata
bas_dir=f'D:\Research\HyperCoast\PACEmap\PACEdata-20241203T182106Z-001\PACEdata'

filepath="PACE_OCI.20241023T175219.L2.OC_AOP.V2_0.NRT.nc"
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

# Station list for eval_data.npy
station_list = ['b03', 'b05', 'b07', 'b09', 'b11', 'b15', 'T3', 'T5', 'T15', 'TC8']
# Corresponding latitude and longitude pairs
specific_lat_lon_pairs = np.array([
    [29.33401, -89.9226],
    [29.37323, -89.9451],
    [29.41868, -89.9299],
    [29.36688, -89.90676],
    [29.3628, -89.8509],
    [29.27755, -89.95436],
    [29.18584845, -90.31021563],
    [29.13392247, -90.31021563],
    [29.21700403, -90.48676394],
    [29.1322, -90.3812]
])

# Station list for f_data.npy
# station_list = ['b02', 'b04', 'b06', 'b08', 'b10', 'b13', 'T2', 'T4', 'T14', 'TC7']

# Corresponding latitude and longitude pairs
# specific_lat_lon_pairs = np.array([
#     [29.32108, -89.9449],
#     [29.34731, -89.94508],
#     [29.39965, -89.94478],
#     [29.39236, -89.9157],
#     [29.38461, -89.8755],
#     [29.26266, -89.89625],
#     [29.23777442, -90.34137121],
#     [29.1766, -90.581],
#     [29.14430767, -90.47637875],
#     [29.2154, -90.4102]
# ])

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

# save_path = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\ft_Rrs_Stations_1023.npy'
save_path = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\ft_train_Rrs_Stations_1023.npy'


np.save(save_path, Rrs_interpolated_np)
print(f"Shape of Rrs is {Rrs_interpolated_np.shape}")
print(f"Array saved to {save_path}")
