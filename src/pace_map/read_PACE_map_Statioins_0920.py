import hypercoast
import numpy as np
import os
from scipy.interpolate import griddata
bas_dir=f'D:\Research\HyperCoast\PACEmap\PACEdata-20241203T182106Z-001\PACEdata'

filepath="PACE_OCI.20240920T183746.L2.OC_AOP.V2_0.NRT.nc"
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
station_list=['C1','C4','Hyper3','N2']
specific_lat_lon_pairs = np.array([
[30.31472,	-90.10945999],
[30.201519,	-90.134024],
[30.341124,	-90.179677],
[30.296599,	-90.277498],
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

# save_path = r'D:\Research\HyperCoast\PACEmap\PACEdata-20241203T182106Z-001\PACEdata\PACEmap_Dec2_ft\Rrs_Stations_0920.npy'

save_path = r'D:\Research\EnvironmentalData\BenchmarkEvaluation_r3\papers\ft_train_Rrs_Stations_0920.npy'


np.save(save_path, Rrs_interpolated_np)
print(f"Shape of Rrs is {Rrs_interpolated_np.shape}")
print(f"Array saved to {save_path}")
