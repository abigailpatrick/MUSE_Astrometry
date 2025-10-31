from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from scipy.ndimage import rotate
import numpy as np

def plot_fits_image(fits_file):
    # Open FITS file and extract image data
    with fits.open(fits_file, ignore_missing_simple=True) as hdul:
        data = hdul[0].data

    # Handle NaNs
    data = np.nan_to_num(data)

    # Rotate 20 degrees clockwise (negative = clockwise)
    data_rot = rotate(data, 19.5, reshape=True)

    # Normalize for better contrast
    norm = simple_norm(data_rot, 'sqrt', percent=99.5)

    # Create figure with transparent background
    fig = plt.figure(frameon=False)
    fig.set_size_inches(6, 6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Plot image in grayscale
    ax.imshow(data_rot, cmap='gray', origin='lower', norm=norm)

    # Save with truly transparent background
    output_file = fits_file.replace('.fits', '.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    print(f" Saved rotated, transparent image to {output_file}")

# Example usage
plot_fits_image("/cephfs/apatrick/musecosmos/scripts/aligned/masked_exposures/DATACUBE_FINAL_Autocal3693592a_2_ZAP_img_aligned_masked0.0p.fits")
