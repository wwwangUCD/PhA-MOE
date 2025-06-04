import hypercoast
import numpy as np
import os
bas_dir=r'D:\Research\HyperCoast\PACEmap\4Seasons'
date='0929'
filepath="PACE_OCI.20240929T185124.L2.OC_AOP.V2_0.NRT.nc"
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

save_dir = r'D:\Research\HyperCoast\PACEmap\4Seasons\npyData'
save_path=os.path.join(save_dir,f'Rrs_All_{date}.npy')
np.save(save_path, filtered_Rrs)

# Create a mask that is 1 where all wavelengths for a given pixel have non-NaN values, and 0 otherwise
mask = np.all(~np.isnan(filtered_Rrs), axis=2).astype(int)

save_path=os.path.join(save_dir,f'Rrs_All_nan_mask_{date}.npy')
np.save(save_path, mask)

