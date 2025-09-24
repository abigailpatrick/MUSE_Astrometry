import argparse
import os
import numpy as np
import pandas as pd
from astropy.io import fits




df = pd.read_csv("/home/apatrick/P1/slurm/1537_wave.csv")

# print the min and max wavelength
print(df["slice_wavelength"].min(), df["slice_wavelength"].max())

# print the difference between the max and min wavelength
print(df["slice_wavelength"].max() - df["slice_wavelength"].min())

# print the median wavelength
print(np.median(df["slice_wavelength"].values))
