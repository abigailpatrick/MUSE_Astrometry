from astropy.io import fits
 

path = "/cephfs/apatrick/musecosmos/reduced_cubes/norm/DATACUBE_FINAL_Autocal3692946a_1_ZAP_norm.fits"

with fits.open(path) as hdul:
    hdul.info()