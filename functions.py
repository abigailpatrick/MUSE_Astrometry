
# imports

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from numpy import ma
import numpy as np
from astropy.table import Table, Column, MaskedColumn, pprint
from astropy.io import fits
from astropy.wcs import WCS
from scipy import integrate
from astropy.coordinates import SkyCoord


# Imports necessary for Source Catalog
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from scipy import ndimage
from scipy.ndimage import binary_erosion
from numpy import loadtxt
from photutils.aperture import CircularAperture, aperture_photometry, EllipticalAperture
from astropy.coordinates import match_coordinates_sky

# Imports necessary for visulisation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import simple_norm


from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, Moffat2DKernel
import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import Cube, Image

# Imports for cutouts and convolution

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, Moffat2DKernel
import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import Cube, Image
import os

# functions


def source_catalog(data, wcs_muse,photplam,pixel_scale,npixels=10, radii=[3.0, 4.0, 5.0],fout=f'outputs/source_catalog_MUSE.fits'):


    """  

    Takes an image from the mpdaf bandpass_image function 
    (a 2d image convolved with a bandpass filter) and returns a source catalog
    of the sources in the image. 

    Parameters
    ----------
    data : np.ndarray
        The data array of the image

    wcs_muse: astropy.wcs.WCS
        The WCS information extracted from the MUSE image.

    npixels : int
        The number of pixels that define the minimum source size. Default is 10.
    
    radii : list of float
        List of aperture radii for aperture photometry. Default is [3.0, 4.0, 5.0].

    pixel_scale : float
        The pixel scale of the image in arcseconds per pixel.
    
    fout : str
        The output file path for the FITS file.

    

    Returns
    -------
    data : np.ndarray
        The data array of the image

    tbl : astropy.table.Table
        A table containing the source catalog

    segment_map : np.ndarray
        The segmentation map of the image

    segm_deblend : np.ndarray
        The deblended segmentation map of the image

    cat : photutils.segmentation.SourceCatalog
        The source catalog of the image

    aperture_phot_tbl : astropy.table.Table
        A table containing the aperture photometry results for each radius


    """

    
    # Replace all 0.0 values with NaN
    data = np.where((data == 0.0), np.nan, data)

    # Erosion of the image
    data_e = ndimage.binary_erosion(data, iterations=20)
    data = np.where(data_e == False, np.nan, data)

    
    # Background substraction
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (30, 30), filter_size=(3, 3),
                    bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    # Select threshold above background at x-sigma  pixel noise level
    threshold = 2 * bkg.background_rms

    # Convolve the data with a 2d gaussian kernel
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)

    # Detect sources in the convolved data
    segment_map = detect_sources(convolved_data, threshold, npixels) # Should all the npixels be the same?
    
    # Deblend the sources
    segm_deblend = deblend_sources(convolved_data, segment_map,
                                npixels, nlevels=32, contrast=0.001,
                                progress_bar=False)

    # Find the sources
    finder = SourceFinder(npixels, progress_bar=False)
    segment_map = finder(convolved_data, threshold)


    # Create a source catalog
    cat = SourceCatalog(data, segm_deblend, convolved_data=data, wcs=wcs_muse)
    
    # Create a table of the source catalog
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'

    # Add the Kron radius to the table
    tbl['kron_radius'] = cat.kron_radius

    # Add the source fwhm to the table 
    tbl['fwhm'] = cat.fwhm


    # Add semi-major and semi-minor axes to the table
    tbl['semi_major_axis'] = cat.semimajor_sigma
    tbl['semi_minor_axis'] = cat.semiminor_sigma

    # Calculate the area of the apertures
    aperture_areas = np.pi * tbl['semi_major_axis'] * tbl['semi_minor_axis'] * tbl['kron_radius']**2
    tbl['aperture_area'] = aperture_areas


    # Convert Kron radius from pixels to arcseconds
    kron_radius_arcsec = tbl['kron_radius'] * pixel_scale
    tbl['kron_radius_arcsec'] = kron_radius_arcsec


    print (tbl['aperture_area'])
    # Perform aperture photometry for multiple radii
    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
    apertures = [CircularAperture(positions, r=r) for r in radii]
    aperture_phot_tbl = aperture_photometry(data, apertures)

    # Add aperture photometry results to the table
    for i, r in enumerate(radii):
        colname = f'aperture_sum_{r}'
        tbl[colname] = aperture_phot_tbl[f'aperture_sum_{i}']

    # Convert pixel coordinates (xcentroid, ycentroid) to RA and Dec
    ra, dec = wcs_muse.all_pix2world(tbl['xcentroid'], tbl['ycentroid'], 0)

    # Create SkyCoord object with units in degrees
    sky_coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    # Add RA and Dec columns to the table 
    tbl['RA'] = sky_coords.ra
    tbl['Dec'] = sky_coords.dec

    aperture_col = f'kron_flux'

    # Convert insert physical units of MUSE
    arr_es = tbl[aperture_col] * 10**-20 * u.erg / (u.s * u.cm**2 * u.AA)
    
    # Convert to flux density (Jy)
    arr_jy = arr_es.to(u.Jy, equivalencies=u.spectral_density(photplam * u.AA)) 
    
    # Convert Jy to µJy
    arr_ujy = arr_jy.to(u.microjansky)
    
    # Save the new values in the table
    new_col_name = f'{aperture_col}_uJy'
    tbl[new_col_name] = arr_ujy  # Store in the table
    

    print(tbl.colnames)
    print(tbl)

    # Save the table as a FITS file
    tbl.write(fout, format='fits', overwrite=True)

    #Get catalog for one source
    #cat_source = get_label(tbl, source)

    #print(cat_source)
    
    return data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl




def visulisation(segment_map, data, segm_deblend, cat, fout='outputs/visualisation.pdf'):

    """
    Generates plots of the data, segmentation map and deblended segmentation map
    of the sources in the image. It also plots the sources selected in the source catalog.  
    
    This gives the visulisation of the sources selected so the selection critieria can be adjusted by eye.

    Parameters
    ----------
    segment_map : np.ndarray
        The segmentation map of the image

    data : np.ndarray   
        The data array of the image
    
    segm_deblend : np.ndarray       
        The deblended segmentation map of the image

    cat : photutils.segmentation.SourceCatalog
        The source catalog of the image
    
    Returns
    -------
    plt.show() : matplotlib.pyplot
        The plots of the data, segmentation map and deblended segmentation map of the sources in the image


    """
    pdf_pages = PdfPages(fout)
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Background-subtracted Data')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image')

    norm = ImageNormalize(stretch=SqrtStretch())
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    ax.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
              interpolation='nearest')
    ax.set_title('Deblended Segmentation Image')
    plt.tight_layout()

    norm = simple_norm(data, 'sqrt')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Data')
    ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)

    return pdf_pages.savefig(), pdf_pages.close()  

def get_wcs_info(im_muse, wcs_muse):
    """
    Get the WCS information from the MUSE image to dictate the shape of  
    the cutout.
    
    Parameters
    ----------
    im_muse : np.ndarray   
        The data array of the muse image (probably after bandpass filter applied).

    wcs_muse: astropy.wcs.WCS
        The WCS information extracted from the MUSE image.
    
    Returns
    -------
    shape_muse : tuple
        The shape of the MUSE image.
    
    wcs_muse : astropy.wcs.WCS
        The WCS information extracted from the MUSE image.

    central_pixel : tuple
        The central pixel coordinates of the MUSE image.

    ra_muse : float
        The RA of the central pixel of the MUSE image.

    dec_muse : float
        The Dec of the central pixel of the MUSE image.

    width_deg : float
        The width of the MUSE image in degrees.

    height_deg : float
        The height of the MUSE image in degrees.

    fwhm_nyquist : float astropy.u 
        The FWHM for Nyquist sampling in arcesecs. 
    
    
    """
    # Get the shape of the MUSE image
    shape_muse = im_muse.shape

    # Calculate the central pixel coordinates of the MUSE image
    central_pixel = (shape_muse[0] // 2, shape_muse[1] // 2)

    # Convert the central pixel coordinates to world coordinates (RA, Dec)
    central_coords = wcs_muse.all_pix2world(central_pixel[1], central_pixel[0], 0)
    ra_muse, dec_muse = central_coords[0], central_coords[1]

    # Calculate the size of the MUSE image in world coordinates (degrees)
    # Convert the pixel corners to world coordinates and calculate the differences
    corner1 = wcs_muse.all_pix2world(0, 0, 0)
    corner2 = wcs_muse.all_pix2world(shape_muse[1], shape_muse[0], 0)
    width_deg = abs(corner2[0] - corner1[0])
    height_deg = abs(corner2[1] - corner1[1])

    print(f"Central RA: {ra_muse} degrees")
    print(f"Central Dec: {dec_muse} degrees")
    print(f"Width: {width_deg} degrees")
    print(f"Height: {height_deg} degrees")  
    print(f"central pixel: {central_pixel}")
   
    # Get cd object
    cd_matrix = wcs_muse.wcs.cd

    # Extract the CD1_1 value
    cd1_1 = abs(cd_matrix[0, 0])

    print("CD1_1 value:", cd1_1)

    # Unit conversion to arcseconds
    pixscale_deg = cd1_1 * u.deg
    pixscale_arcs = pixscale_deg.to(u.arcsec)

    print("Value in arcseconds:", pixscale_arcs)

    # Calculate the pixscale for Nyquist sampling (2 times the pixel scale)
    pixscale_nyquist = 2 * pixscale_arcs #have this as an output and input it into the convolution function

    
    print(f"FWHM for Nyquist sampling: {pixscale_nyquist}")

    
    return shape_muse, wcs_muse, central_pixel, ra_muse, dec_muse, width_deg, height_deg, pixscale_nyquist

def create_cutout(image_data_hst, wcs_hst, width_deg, height_deg, ra_muse, dec_muse, fout='outputs/hst_cutout.fits'):
    """
    Create a cutout of the HST image using the WCS information from the MUSE image.

    Prior to this need to read in hst image date (hdul[0].data) and wcs (hdul[0].header)

    Parameters
    ----------
    image_data_hst : np.ndarray
        The data array of the HST image.

    wcs_hst : astropy.wcs.WCS
        The WCS information of the HST image.

    width_deg : float
        The width of the cutout in degrees.
    
    height_deg : float
        The height of the cutout in degrees.

    ra_muse : float
        The RA of the central pixel of the MUSE image.

    dec_muse : float
        The Dec of the central pixel of the MUSE image.

    Returns
    -------
    cutout : astropy.nddata.Cutout2D
        The cutout of the HST image.

    cutout_wcs : astropy.wcs.WCS
        The WCS information of the cutout.

    Also saves the cutout to a new FITS file 'hst_cutout.fits'.
    
    """
    # 
    # Convert the size from degrees to pixel units in the HST image
    pixel_scale_hst = wcs_hst.proj_plane_pixel_scales()
    width_pix = (width_deg / pixel_scale_hst[0]).value
    height_pix = (height_deg / pixel_scale_hst[1]).value   

    # Convert the central world coordinates to pixel coordinates in the HST image
    central_pixel_hst = wcs_hst.world_to_pixel_values(ra_muse, dec_muse) 

    # Converrt the central pixel coordinates to RA and DEC
    central_coord_hst = wcs_hst.all_pix2world(central_pixel_hst[0], central_pixel_hst[1], 0)
    ra_hst, dec_hst = central_coord_hst[0], central_coord_hst[1]

    print(f"Central RA: {ra_hst} degrees")
    print(f"Central Dec: {dec_hst} degrees")

    # Create the cutout with the calculated size in pixels
    size_cutout = (int(height_pix), int(width_pix))
    cutout = Cutout2D(image_data_hst, position=central_pixel_hst, size=size_cutout, wcs=wcs_hst)

    # Get the WCS object from the Cutout2D
    cutout_wcs = cutout.wcs

    # Save the cutout to a new FITS file
    hdu = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header())
    hdu.writeto(fout, overwrite=True)

    cutout = cutout.data

   


    return cutout, cutout_wcs



def plot_cutout(cutout, wcs_header, fout='outputs/hst_cutout.pdf'):
    """
    Plot the cutout of the HST image with axes in arcseconds.

    Parameters
    ----------
    cutout : 2d array
        The cutout of the HST image.
    wcs_header : astropy.io.fits.Header
        The WCS header from the original FITS file.
    fout : str
        The output file path for the PDF.

    Returns
    -------
    None
    """
    pdf_pages = PdfPages(fout)
    wcs = wcs_header

    # Compute the percentiles to adjust contrast
    vmin = np.percentile(cutout, 5)
    vmax = np.percentile(cutout, 95)

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)
    im = ax.imshow(cutout, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, orientation='vertical')

    ax.set_title('Cutout from HST Image')
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')

    # Set the format of the tick labels to degrees
    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_major_formatter('d.ddd')
    lat.set_major_formatter('d.ddd')

    plt.show()

    return pdf_pages.savefig(), pdf_pages.close()


def convolve_image(image_data_hst, fwhm, gamma):
    """
    Convolve the HST image with a Moffat kernel.

    Parameters
    ----------
    image_data_hst : np.ndarray
        The data array of the HST image.

    fwhm : float
        The full width at half maximum of the Moffat kernel.

    gamma : float
        The gamma parameter of the Moffat kernel.

    Returns
    -------
    convolved_image : np.ndarray
        The convolved image.

    """

    # Calculate alpha
    alpha = fwhm / (2 * np.sqrt(2**(1/gamma) - 1))  

    # Create the Moffat kernel
    kernel_M = Moffat2DKernel(gamma=gamma, alpha=alpha)

    # Convolve the image with the kernel
    convolved_image = convolve(image_data_hst, kernel_M)

    return convolved_image

def plot_cutout_convolved(convolved_image, fout='outputs/hst_convolved_cutout.pdf'):

    """
    Plot the convolved cutout of the HST image.

    Parameters
    ----------
    convolved_image : np.ndarray
        The convolved image.

    Returns
    -------
    pdf_pages.savefig() :
        Saves the plot to a pdf file called 'cutout.pdf'.

    pdf_pages.close() :
        Closes the pdf file 'cutout.pdf'.

    """

    pdf_pages = PdfPages(fout)

    # Compute the percentiles to adjust contrast
    vmin = np.percentile(convolved_image, 5)
    vmax = np.percentile(convolved_image, 95)

    # Plot the cutout
    plt.imshow(convolved_image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Convolved Cutout from HST Image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.colorbar()
    plt.show()

    return pdf_pages.savefig(), pdf_pages.close()  



def source_catalog_HST(data, wcs_hst, photflam, photplam, radii, npixels=10, fout='outputs/source_catalog_HST.fits'):


    """  

    Takes an image from the mpdaf bandpass_image function 
    (a 2d image convolved with a bandpass filter) and returns a source catalog
    of the sources in the image. 

    Parameters
    ----------
    data : np.ndarray
        The data array of the image

    wcs_hst: astropy.wcs.WCS
        The WCS information extracted from the HST image.

    npixels : int
        The number of pixels that define the minimum source size. Default is 10.

    radii : list of float
        List of aperture radii for aperture photometry.


    Returns
    -------
    data : np.ndarray
        The data array of the image

    tbl : astropy.table.Table
        A table containing the source catalog

    segment_map : np.ndarray
        The segmentation map of the image

    segm_deblend : np.ndarray
        The deblended segmentation map of the image

    cat : photutils.segmentation.SourceCatalog
        The source catalog of the image

    aperture_phot_tbl : astropy.table.Table
        A table containing the aperture photometry results for each radius


    """

    
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (30, 30), filter_size=(3, 3),
                    bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    # Select threshold above background at x-sigma  pixel noise level
    threshold = 3 * bkg.background_rms

    # Convolve the data with a 2d gaussian kernel
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    data = convolve(data, kernel)

    # Detect sources in the convolved data
    segment_map = detect_sources(data, threshold, npixels) # Should all the npixels be the same?
    
    # Deblend the sources
    segm_deblend = deblend_sources(data, segment_map,
                                npixels, nlevels=32, contrast=0.001,
                                progress_bar=False)

    # Find the sources
    finder = SourceFinder(npixels, progress_bar=False)
    segment_map = finder(data, threshold)

    # Create a source catalog
    cat = SourceCatalog(data, segm_deblend, convolved_data=data, wcs=wcs_hst)
    
    # Create a table of the source catalog
    tbl = cat.to_table()
    
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f' #units?

    # Add the source fwhm to the table 
    tbl['fwhm'] = cat.fwhm
    

    # Perform aperture photometry for multiple radii
    positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
    apertures = [CircularAperture(positions, r=r) for r in radii]
    aperture_phot_tbl = aperture_photometry(data, apertures)

    # Add aperture photometry results to the table
    for i, r in enumerate(radii):
        colname = f'aperture_sum_{r}'
        tbl[colname] = aperture_phot_tbl[f'aperture_sum_{i}']
        # BUNIT   = 'ELECTRONS/S'          / Units of image data

    # Convert pixel coordinates (xcentroid, ycentroid) to RA and Dec
    ra, dec = wcs_hst.all_pix2world(tbl['xcentroid'], tbl['ycentroid'], 0)

    # Create SkyCoord object with units in degrees
    sky_coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    # Add RA and Dec columns to the table 
    tbl['RA'] = sky_coords.ra
    tbl['Dec'] = sky_coords.dec

    #The below other than print statements is an experiment in converting the flux to microjansky
    
    # Conversion factor from steradians to square arcseconds
    #pixar_sr = ((0.03 * u.arcsec)**2).to(u.steradian)  # Assuming pixel scale of 0.03 arcsec/pixel, update if different

    
    # Conversion process for the specified band - star
    for i, r in enumerate(radii):
        aperture_col = f'aperture_sum_{r}'
        # Convert ELECTRONS/S to physical units (erg/cms/s/AA)
        arr_es = tbl[aperture_col] * (u.electron / u.s)
        arr_pflam = arr_es * (photflam * u.erg / u.cm**2 / u.AA / u.electron)
    
        # Convert to flux density (Jy)
        arr_jy = arr_pflam.to(u.Jy, equivalencies=u.spectral_density(photplam * u.AA)) 
    
        # Convert Jy to µJy
        arr_ujy = arr_jy.to(u.microjansky)
    
        # Save the new values in the table
        new_col_name = f'{aperture_col}_uJy'
        tbl[new_col_name] = arr_ujy  # Store in the table

    aperture_col = f'kron_flux'

    # Convert ELECTRONS/S to physical units (erg/cms/s/AA)
    arr_es = tbl[aperture_col] * (u.electron / u.s)
    arr_pflam = arr_es * (photflam * u.erg / u.cm**2 / u.AA / u.electron)
    
    # Convert to flux density (Jy)
    arr_jy = arr_pflam.to(u.Jy, equivalencies=u.spectral_density(photplam * u.AA)) 
    
    # Convert Jy to µJy
    arr_ujy = arr_jy.to(u.microjansky)
    
    # Save the new values in the table
    new_col_name = f'{aperture_col}_uJy'
    tbl[new_col_name] = arr_ujy  # Store in the table
    

    print(tbl.colnames)
    print(tbl)



    # Save the table as a FITS file
    tbl.write(fout, format='fits', overwrite=True)

    
    return data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl 


def visulisation_HST(segment_map, data, segm_deblend, cat ,fout='outputs/visualisation_HST.pdf'):

    """
    Generates plots of the data, segmentation map and deblended segmentation map
    of the sources in the image. It also plots the sources selected in the source catalog.  
    
    This gives the visulisation of the sources selected so the selection critieria can be adjusted by eye.

    Parameters
    ----------
    segment_map : np.ndarray
        The segmentation map of the image

    data : np.ndarray   
        The data array of the image
    
    segm_deblend : np.ndarray       
        The deblended segmentation map of the image

    cat : photutils.segmentation.SourceCatalog
        The source catalog of the image
    
    Returns
    -------
    plt.show() : matplotlib.pyplot
        The plots of the data, segmentation map and deblended segmentation map of the sources in the image


    """

    # Compute the percentiles to adjust contrast
    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 95)

    pdf_pages = PdfPages(fout)
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
    ax1.set_title('Background-subtracted Data')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image')

    norm = ImageNormalize(stretch=SqrtStretch())
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    ax.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
              interpolation='nearest')
    ax.set_title('Deblended Segmentation Image')
    plt.tight_layout()

    norm = simple_norm(data, 'sqrt')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
    ax1.set_title('Data')
    ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)


    return pdf_pages.savefig(), pdf_pages.close()




def plot_flux_vs_aperture(tblconv,tblraw, radii ,source_index=0, fout='outputs/flux_vs_aperture.pdf'):
    """
    Plots the flux in microjanskys against aperture radius for a given source.

    Parameters
    ----------
    tblconv : astropy.table.Table
        The convolved source catalog table returned by source_catalog_HST.

    tblraw : astropy.table.Table
        The raw source catalog table returned by source_catalog_HST.

    source_index : int, optional
        Index of the source to plot. Default is 0.

    radii : list of float, optional
        List of aperture radii for which the flux is plotted. Default is [3.0, 4.0, 5.0].

    fout : str, optional
        The output file name. Default is 'outputs/flux_vs_aperture.pdf'.


    Returns
    -------
    pdf_pages.savefig() :
        Saves the plot to a pdf file called 'flux_vs_aperture.pdf'.

    pdf_pages.close() :
        Closes the pdf file 'flux_vs_aperture.pdf'.

    """
    # Select the source from the table
    sourceconv = tblconv[source_index]
    sourceraw = tblraw[source_index]

    # Extract flux values in microjanskys for different aperture radii
    flux_uJy_conv = [sourceconv[f'aperture_sum_{r}_uJy'].value for r in radii]
    flux_uJy_raw = [sourceraw[f'aperture_sum_{r}_uJy'].value for r in radii]


    # Normalize the flux values by dividing by the total flux
    total_flux_conv = sum(flux_uJy_conv)
    total_flux_raw = sum(flux_uJy_raw)

    normalized_flux_conv = [flux / total_flux_conv for flux in flux_uJy_conv]
    normalized_flux_raw = [flux / total_flux_raw for flux in flux_uJy_raw]

    pdf_pages = PdfPages(fout)

    # Plot the flux values against the aperture radii
    plt.figure(figsize=(8, 6))
    plt.plot(radii, normalized_flux_conv, marker='o', linestyle='-', color='b', label='Convolved')
    plt.plot(radii, normalized_flux_raw, marker='o', linestyle='-', color='r', label='Raw')
    plt.xlabel('Aperture Radius (pixels)')
    plt.ylabel('Normalized Flux')
    plt.title(f'Flux vs. Aperture Radius for Source {source_index}')
    plt.grid(True)
    plt.legend()
    plt.show()

    return pdf_pages.savefig(), pdf_pages.close()


def crossmatch_catalogs(catalog1, catalog2, tolerance_arcsec=1.0):
    """
    Crossmatch two catalogs of sources based on their RA and Dec coordinates.

    Parameters
    ----------
    catalog1 : astropy.table.Table - is it or do I need to convert it to this?
        The first catalog of sources.
    
    catalog2 : astropy.table.Table
        The second catalog of sources.

    tolerance_arcsec : float
        The matching tolerance in arcseconds. Default is 1.0.

    Returns
    -------
    matched_catalog : astropy.table.Table
        A new catalog with only the matched sources.
    
    """
   
    
   # Extract RA and Dec with case-insensitive column names
    ra1 = catalog1[[col for col in catalog1.dtype.names if col.lower() == 'ra'][0]]
    dec1 = catalog1[[col for col in catalog1.dtype.names if col.lower() == 'dec'][0]]

    ra2 = catalog2[[col for col in catalog2.dtype.names if col.lower() == 'ra'][0]]
    dec2 = catalog2[[col for col in catalog2.dtype.names if col.lower() == 'dec'][0]]

    
    # Create SkyCoord objects
    coords1 = SkyCoord(ra1, dec1, unit='deg')
    coords2 = SkyCoord(ra2, dec2, unit='deg')
    
    # Perform the crossmatch
    idx, d2d, d3d = match_coordinates_sky(coords1, coords2)
    
    # Convert tolerance to degrees
    tolerance = tolerance_arcsec * u.arcsec
    
    # Find matches within the specified tolerance
    matched_mask = d2d <= tolerance
    
    # Get the indices of matched sources in catalog1 and catalog2
    idx1 = np.arange(len(catalog1))[matched_mask]
    idx2 = idx[matched_mask]
    
    # Extract the matched rows from both catalogs
    matched_catalog1 = catalog1[matched_mask]
    matched_catalog2 = catalog2[idx2]

    # Create a new structured array with combined columns from both catalogs
    # Prefix field names from each catalog to ensure uniqueness
    combined_dtype = [(f'cat1_{name}', dtype) for name, dtype in matched_catalog1.dtype.descr]
    combined_dtype += [(f'cat2_{name}', dtype) for name, dtype in matched_catalog2.dtype.descr if f'cat2_{name}' not in combined_dtype]

    combined_catalog = np.empty(matched_catalog1.shape, dtype=combined_dtype)

    # Fill the new structured array with data from both matched catalogs
    # Use the prefixed field names
    for name in matched_catalog1.dtype.names:
        combined_catalog[f'cat1_{name}'] = matched_catalog1[name]
    for name in matched_catalog2.dtype.names:
        combined_catalog[f'cat2_{name}'] = matched_catalog2[name]

    matched_catalog = Table(combined_catalog)

    # Print the column names
    #print("Columns in matched catalog:")
    #for colname in matched_catalog.colnames:
        #print(colname)

    return matched_catalog

def add_offsets(matched_catalog):
    """
    Add columns to the matched catalog with the RA and Dec offsets between the two catalogs.

    Parameters
    ----------
    matched_catalog : astropy.table.Table
        The matched catalog of sources.

    Returns
    -------
    matched_catalog : astropy.table.Table
        The matched catalog with additional columns for RA and Dec offsets.

    """
    # Calculate the RA and Dec offsets
    ra_offset = matched_catalog['cat1_RA'] - matched_catalog['cat2_RA']
    dec_offset = matched_catalog['cat1_Dec'] - matched_catalog['cat2_Dec']

    # Add the offsets to the matched catalog
    matched_catalog['RA_offset'] = ra_offset
    matched_catalog['Dec_offset'] = dec_offset

    # Convert the offsets from degrees to arcseconds
    ra_offset_arcsec = ra_offset * 3600
    dec_offset_arcsec = dec_offset * 3600

    # Add the offsets in arcseconds to the matched catalog
    matched_catalog['RA_offset_arcsec'] = ra_offset_arcsec
    matched_catalog['Dec_offset_arcsec'] = dec_offset_arcsec

    return matched_catalog

def plot_offset(matched_catalog, fout='outputs/offset.pdf'):
    """
    Plot the RA and Dec offsets between the two catalogs.

    Parameters
    ----------
    matched_catalog : astropy.table.Table
        The matched catalog of sources with RA and Dec offsets.

    Returns
    -------
    pdf_pages.savefig() :
        Saves the plot to a pdf file called 'offset.pdf'.

    pdf_pages.close() :
        Closes the pdf file 'offset.pdf'.

    """
    pdf_pages = PdfPages(fout)

    # Calculate statistics
    ra_median = np.median(matched_catalog['RA_offset_arcsec'])
    dec_median = np.median(matched_catalog['Dec_offset_arcsec'])
    ra_16th = np.percentile(matched_catalog['RA_offset_arcsec'], 16)
    ra_84th = np.percentile(matched_catalog['RA_offset_arcsec'], 84)
    dec_16th = np.percentile(matched_catalog['Dec_offset_arcsec'], 16)
    dec_84th = np.percentile(matched_catalog['Dec_offset_arcsec'], 84)

    ra_err = [[ra_median - ra_16th], [ra_84th - ra_median]]
    dec_err = [[dec_median - dec_16th], [dec_84th - dec_median]]

    # Create a figure with a grid of plots
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)

    fig.suptitle('RA and Dec Offsets between Catalogs', fontsize=16)

    
    # Main scatter plot
    main_ax = fig.add_subplot(grid[1:, :-1])
    main_ax.plot(matched_catalog['RA_offset_arcsec'], matched_catalog['Dec_offset_arcsec'], 'ob', markersize=3)
    main_ax.set_xlim(-0.85, 0.85)
    main_ax.set_ylim(-0.85, 0.85)
    main_ax.set_xlabel('RA Offset (arcsec)')
    main_ax.set_ylabel('Dec Offset (arsec)')
    main_ax.grid(True)

    # Plot median point with error bars
    main_ax.errorbar(ra_median, dec_median, xerr=ra_err, yerr=dec_err, fmt='or', ecolor='r', elinewidth=2, capsize=4)


    # Ensure tick labels are set for the main scatter plot
    main_ax.tick_params(axis='both', which='major', labelsize=10)
    main_ax.tick_params(axis='both', which='minor', labelsize=10)

    # Histogram for RA offsets (top plot)
    x_hist = fig.add_subplot(grid[0, :-1], sharex=main_ax)
    x_hist.hist(matched_catalog['RA_offset_arcsec'], bins=50, color='blue', alpha=0.7, density = True)
    x_hist.set_xlim(main_ax.get_xlim())
    x_hist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis ticks and labels
    x_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis ticks and labels for the histogram
    

    # Histogram for Dec offsets (right plot)
    y_hist = fig.add_subplot(grid[1:, -1], sharey=main_ax)
    y_hist.hist(matched_catalog['Dec_offset_arcsec'], bins=50, orientation='horizontal', color='blue', alpha=0.7, density = True)
    y_hist.set_ylim(main_ax.get_ylim())
    y_hist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis ticks and labels for the histogram
    y_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis ticks and labels
   

    # Adjust layout to make sure plots fit well together
    plt.tight_layout()

    return pdf_pages.savefig(), pdf_pages.close()

def plot_offset_comp(matched_catalog1, matched_catalog2, fout='outputs/offset.pdf'):
    """
    Plot the RA and Dec offsets between the two catalogs.

    Parameters
    ----------
    matched_catalog1 : astropy.table.Table
        The matched catalog of sources with RA and Dec offsets.

    matched_catalog2 : astropy.table.Table
        The matched catalog of sources with RA and Dec offsets.

    Returns
    -------
    pdf_pages.savefig() :
        Saves the plot to a pdf file called 'offset.pdf'.

    pdf_pages.close() :
        Closes the pdf file 'offset.pdf'.

    """
    pdf_pages = PdfPages(fout)

    # Calculate statistics
    ra_median1 = np.median(matched_catalog1['RA_offset_arcsec'])
    dec_median1 = np.median(matched_catalog1['Dec_offset_arcsec'])
    ra_16th1 = np.percentile(matched_catalog1['RA_offset_arcsec'], 16)
    ra_84th1 = np.percentile(matched_catalog1['RA_offset_arcsec'], 84)
    dec_16th1 = np.percentile(matched_catalog1['Dec_offset_arcsec'], 16)
    dec_84th1 = np.percentile(matched_catalog1['Dec_offset_arcsec'], 84)

    print (ra_median1, dec_median1, ra_16th1, ra_84th1, dec_16th1, dec_84th1)
    ra_err1 = [[ra_median1 - ra_16th1], [ra_84th1 - ra_median1]]
    dec_err1 = [[dec_median1 - dec_16th1], [dec_84th1 - dec_median1]]

    # Calculate statistics
    ra_median2 = np.median(matched_catalog2['RA_offset_arcsec'])
    dec_median2 = np.median(matched_catalog2['Dec_offset_arcsec'])
    ra_16th2 = np.percentile(matched_catalog2['RA_offset_arcsec'], 16)
    ra_84th2 = np.percentile(matched_catalog2['RA_offset_arcsec'], 84)
    dec_16th2 = np.percentile(matched_catalog2['Dec_offset_arcsec'], 16)
    dec_84th2 = np.percentile(matched_catalog2['Dec_offset_arcsec'], 84)

    print (ra_median2, dec_median2, ra_16th2, ra_84th2, dec_16th2, dec_84th2)
    ra_err2 = [[ra_median2 - ra_16th2], [ra_84th2 - ra_median2]]
    dec_err2 = [[dec_median2 - dec_16th2], [dec_84th2 - dec_median2]]

    # Create a figure with a grid of plots
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)

    fig.suptitle('RA and Dec Offsets between Catalogs', fontsize=16)

    
    # Main scatter plot
    main_ax = fig.add_subplot(grid[1:, :-1])
    main_ax.plot(matched_catalog1['RA_offset_arcsec'], matched_catalog1['Dec_offset_arcsec'], 'ob', markersize=3, label = '814')
    main_ax.plot(matched_catalog2['RA_offset_arcsec'], matched_catalog2['Dec_offset_arcsec'], 'or', markersize=3, label = '606')
    main_ax.set_xlim(-1, 1)
    main_ax.set_ylim(-1, 1)
    main_ax.set_xlabel('RA Offset (arcsec)')
    main_ax.set_ylabel('Dec Offset (arsec)')
    main_ax.grid(True)
    main_ax.legend()

    # Plot median point with error bars
    main_ax.errorbar(ra_median1, dec_median1, xerr=ra_err1, yerr=dec_err1, fmt='ob', ecolor='b', elinewidth=2, capsize=4)
    main_ax.errorbar(ra_median2, dec_median2, xerr=ra_err2, yerr=dec_err2, fmt='or', ecolor='r', elinewidth=2, capsize=4)

    # Ensure tick labels are set for the main scatter plot
    main_ax.tick_params(axis='both', which='major', labelsize=10)
    main_ax.tick_params(axis='both', which='minor', labelsize=10)

    # Histogram for RA offsets (top plot)
    x_hist = fig.add_subplot(grid[0, :-1], sharex=main_ax)
    x_hist.hist(matched_catalog1['RA_offset_arcsec'], bins=50, color='blue', alpha=0.7, density = True)
    x_hist.hist(matched_catalog2['RA_offset_arcsec'], bins=50, color='red', alpha=0.7, density = True)
    x_hist.set_xlim(main_ax.get_xlim())
    x_hist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis ticks and labels
    x_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis ticks and labels for the histogram
    

    # Histogram for Dec offsets (right plot)
    y_hist = fig.add_subplot(grid[1:, -1], sharey=main_ax)
    y_hist.hist(matched_catalog1['Dec_offset_arcsec'], bins=50, orientation='horizontal', color='blue', alpha=0.7, density = True)
    y_hist.hist(matched_catalog2['Dec_offset_arcsec'], bins=50, orientation='horizontal', color='red', alpha=0.7, density = True)
    y_hist.set_ylim(main_ax.get_ylim())
    y_hist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis ticks and labels for the histogram
    y_hist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis ticks and labels
   

    # Adjust layout to make sure plots fit well together
    plt.tight_layout()
    

    return pdf_pages.savefig(), pdf_pages.close()