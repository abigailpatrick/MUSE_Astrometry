import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from mpdaf.obj import Cube, Image
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground


def mkcol(b, v, r, ff, gamma, xlo, xhi, ylo, yhi):
    # (keep function exactly as your supervisor sent)
    tmpb = b[ylo:yhi, xlo:xhi]
    tmpv = v[ylo:yhi, xlo:xhi]
    tmpr = r[ylo:yhi, xlo:xhi]

    bmean,bmedian,bstd,bmax = getstats(tmpb, ff)
    vmean,vmedian,vstd,vmax = getstats(tmpv, ff)
    rmean,rmedian,rstd,rmax = getstats(tmpr, ff)

    print('rescaling...')
    bmin, vmin, rmin = bmean, vmean, rmean

    gdb = np.where((b != 0) & (b != np.nan))
    gdv = np.where((v != 0) & (v != np.nan))
    gdr = np.where((r != 0) & (r != np.nan))

    b[gdb] = (b[gdb]-bmin)/(bmax-bmin)
    v[gdv] = (v[gdv]-vmin)/(vmax-vmin)
    r[gdr] = (r[gdr]-rmin)/(rmax-rmin)

    lo, hi = 0., 1.
    b[b <= lo] = 0; b[b >= hi] = 1
    v[v <= lo] = 0; v[v >= hi] = 1
    r[r <= lo] = 0; r[r >= hi] = 1

    b = b**gamma; v = v**gamma; r = r**gamma

    print('writing to array')
    sz = b.shape
    col = np.zeros((sz[0], sz[1], 3))
    col[:,:,0] = r
    col[:,:,1] = v
    col[:,:,2] = b
    print('mkcol complete.')
    return col

def getstats(img,ff):
   #from photutils import make_source_mask
   gd = np.where((img != 0) & np.isfinite(img))
   print('there are ',len(gd[0]),' elements')
   arr = img[gd]
   arr = sorted(arr)
   n = len(arr)
   print('array is ',n,' elements')
   i = round(ff*n)
   vmax = arr[i]
   print(ff,' signal range value is ',vmax)

   print('making mask')
   #mask = make_source_mask(img, nsigma=2, npixels=5, dilate_size=11)
   mask = make_source_mask(img)
   print('calculating stats')
   vmean, vmedian, vstd = sigma_clipped_stats(img, sigma=3.0, mask=mask, mask_value=0)
   print('mean: ',vmean)
   print('median: ',vmedian)
   print('sigma: ',vstd)
   return vmean,vmedian,vstd,vmax

def make_source_mask(img):

   bkg_estimator = MedianBackground()
   
   bkg = Background2D(img, (50, 50), filter_size=(3, 3),bkg_estimator=bkg_estimator)
   imgb = img.copy()
   imgb = img - bkg.background  # subtract the background
   #amstools.writefits_nohdr('imgb.fits',imgb)
   threshold = 2 * bkg.background_rms
   segment_map = detect_sources(imgb, threshold, npixels=10)
   segment_map = segment_map.data
   #segment_map = segment_map.astype(float)
   bad = np.where(segment_map >= 1)
   gd = np.where(segment_map == 0)
   segment_map[bad] = True
   segment_map[gd] = False
   #amstools.writefits_nohdr('seg.fits',segment_map)
   return(segment_map)

def wl(fits_path, out_fits, out_png, out_png_rgb, out_png_rgb3d=None):
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        # channels kept (as in your code)
        kept_idx = np.r_[0:800, 1000:3680]
        data = data[kept_idx, :, :]

        header_f = hdul[1].header
        print("Full header info:")  
        print(repr(header_f))

        white_data = np.sum(data, axis=0, dtype=np.float32)
        blue = np.sum(data[0:843, :, :], axis=0)
        green = np.sum(data[843:843+1175, :, :], axis=0)
        red = np.sum(data[843+1175:, :, :], axis=0)

    hdu = fits.PrimaryHDU(data=white_data, header=header)
    hdu.writeto(out_fits, overwrite=True)
    print(f"White-light image saved to {out_fits}")

    # White-light plot
    plt.figure(figsize=(8, 6))
    norm = simple_norm(white_data, 'sqrt', percent=99)
    plt.imshow(white_data, origin='lower', cmap='viridis', norm=norm)
    plt.colorbar(label='Flux')
    plt.title("White-light Image from FITS Cube")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"White-light image plot saved to {out_png}")

    # RGB composite
    print("Creating color image with mkcol...")
    ff = 0.99
    gamma = 0.5
    xlo, xhi, ylo, yhi = 0, white_data.shape[1], 0, white_data.shape[0]
    col = mkcol(blue.copy(), green.copy(), red.copy(), ff, gamma, xlo, xhi, ylo, yhi)

    plt.figure(figsize=(8, 6))
    plt.imshow(col, origin='lower')
    plt.title("MUSE RGB Composite")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(out_png_rgb, dpi=300)
    plt.close()
    print(f"RGB composite image saved to {out_png_rgb}")

    # 3D IFU block (optional)
    if out_png_rgb3d is not None:
        save_ifu_rgb_block(
            col=col,
            header=header,
            kept_channels_idx=kept_idx,
            out_png_3d=out_png_rgb3d,
            max_dim=900,       # increase to ~1200 if you have RAM/CPU
            n_slices=12,
            depth_ratio=0.28,  # a touch deeper for poster drama
            tint_strength=0.25,
            alpha_bg_threshold=0.01,
            elev=22, azim=-60,
            dpi=300
        )

import matplotlib
from matplotlib import cm

def save_ifu_rgb_block(
    col,
    header,
    kept_channels_idx,
    out_png_3d,
    *,
    max_dim=900,          # downsample max dimension (pixels) for plotting speed
    n_slices=12,          # number of stacked slices into the page
    depth_ratio=0.25,     # depth relative to max(width,height)
    tint_strength=0.25,   # how much the λ gradient tints interior slices
    alpha_bg_threshold=0.01, # threshold to make background transparent
    elev=22, azim=-60,    # camera angles for 3D view
    dpi=300               # export DPI for poster
):
    """
    Turn a 2D RGB image (col) into a 3D 'IFU block' with wavelength axis.

    Parameters
    ----------
    col : (H, W, 3) float in [0,1]
        The RGB composite (from mkcol).
    header : fits.Header
        FITS header containing spectral WCS (CRVAL3/CDELT3[/CD3_3]/CUNIT3).
    kept_channels_idx : 1D array-like of int
        Indices of spectral channels retained (after your cut).
        Used to compute λmin and λmax from header.
    out_png_3d : str
        Output image path (PNG recommended).
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Determine wavelength range from header + kept channels
    def _get_cdelt3(h):
        if 'CDELT3' in h:
            return float(h['CDELT3'])
        # fallback for old WCS style
        for key in ('CD3_3', 'PC3_3'):
            if key in h:
                return float(h[key])
        # final fallback
        return 1.0

    crval3 = float(header.get('CRVAL3', 0.0))
    cdelt3 = _get_cdelt3(header)
    cunit3 = (header.get('CUNIT3', 'Angstrom') or 'Angstrom').lower().strip()

    # Convert to Angstrom if needed
    unit_scale = 1.0
    if cunit3 in ('nm', 'nanometer', 'nanometre'):
        unit_scale = 10.0
    elif cunit3 in ('m', 'meter', 'metre'):
        unit_scale = 1e10
    elif cunit3 in ('micron', 'micrometer', 'micrometre', 'um', 'µm'):
        unit_scale = 1e4
    # else assume Angstrom

    kept_channels_idx = np.asarray(kept_channels_idx)
    i0, i1 = int(kept_channels_idx.min()), int(kept_channels_idx.max())

    lam0 = (crval3 + cdelt3 * i0) * unit_scale
    lam1 = (crval3 + cdelt3 * i1) * unit_scale
    lam_mid = 0.5 * (lam0 + lam1)

    # 2) Downsample image for 3D plotting efficiency
    H, W, _ = col.shape
    step = max(1, int(max(H, W) / max_dim))
    cold = col[::step, ::step, :]
    Hd, Wd, _ = cold.shape

    # Build alpha mask from intensity to preserve irregular footprint
    intensity = cold.mean(axis=2)
    # Normalize and threshold to get a clean alpha (transparent background)
    if intensity.max() > 0:
        a = (intensity - alpha_bg_threshold) / max(1e-6, intensity.max() - alpha_bg_threshold)
    else:
        a = np.zeros_like(intensity)
    a = np.clip(a, 0.0, 1.0)

    # 3) Build coordinates
    y = np.linspace(0, H, Hd)
    x = np.linspace(0, W, Wd)
    X, Y = np.meshgrid(x, y)
    depth = depth_ratio * max(W, H)

    # 4) Plot stacked slices with spectral tint
    fig = plt.figure(figsize=(9, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # alpha ramp for interior slices (front strong, back faint)
    alpha_ramp = np.linspace(1.0, 0.18, n_slices)

    for k, z in enumerate(np.linspace(0.0, depth, n_slices)):
        t = 0.0 if depth == 0 else z / depth
        tint_rgb = np.array(cm.Spectral(t))[:3]
        # Tint only interior slices; keep the front slice closer to original colours
        blend = tint_strength if k > 0 else tint_strength * 0.2
        col_tinted = np.clip((1 - blend) * cold + blend * tint_rgb, 0, 1)

        fcolors = np.dstack([col_tinted, a * alpha_ramp[k]])
        ax.plot_surface(
            X, Y, z * np.ones_like(X),
            rstride=1, cstride=1,
            facecolors=fcolors,
            shade=False, linewidth=0, antialiased=False
        )

    # 5) Axes, labels, ticks
    ax.set_xlabel('RA (pixels)')
    ax.set_ylabel('Dec (pixels)')
    ax.set_zlabel('λ (Å)')

    ax.set_zticks([0, depth/2, depth])
    ax.set_zticklabels([f'{lam0:.0f}', f'{lam_mid:.0f}', f'{lam1:.0f}'])

    # Colorbar showing λ gradient
    sm = cm.ScalarMappable(cmap=cm.Spectral,
                           norm=matplotlib.colors.Normalize(vmin=lam0, vmax=lam1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, shrink=0.6)
    cbar.set_label('Wavelength λ (Å)')

    # Aspect and camera
    try:
        ax.set_box_aspect((W, H, depth))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    # Lighter grid for a clean poster look
    ax.grid(False)
    for pane in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        try:
            pane.set_pane_color((1,1,1,0))  # transparent panes, newer MPL
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(out_png_3d, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"3D IFU block saved to {out_png_3d}")


if __name__ == "__main__":
    cube_path = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits"
    out_fits = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_wl.fits"
    out_png  = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_wl_image.png"
    out_png_rgb = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_rgb.png"

    wl(cube_path, out_fits, out_png, out_png_rgb)



#subcube = mcube[:, 500:1500, 500:1500]
# (wavelength, y, x)
#print ("Subcube 500:1500,500:1500 extracted")
#subcube.info()
#subcube.write("/cephfs/apatrick/musecosmos/dataproducts/extractions/subcube_500_1500_500_1500.fits")

"""

# Load subcube from file to verify
#subcube = Cube("/cephfs/apatrick/musecosmos/dataproducts/extractions/subcube_500_1500_500_1500.fits", ext=1)
##print ("Subcube re-loaded from file")
#subcube.info()

# save primary and extension headers to text file
with open("/cephfs/apatrick/musecosmos/dataproducts/extractions/new_headers.txt", "w") as f:
    with fits.open("/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits") as hdul:
        f.write("Primary Header:\n")
        f.write(repr(hdul[0].header))
        f.write("\n\nExtension Header:\n")
        f.write(repr(hdul[1].header))   
print (f"Subcube headers saved to /cephfs/apatrick/musecosmos/dataproducts/extractions/new_headers.txt")


#wcs = subcube.wcs



# White-light image (sum over all wavelengths to get 2D image)
white_img = subcube.sum(axis=0)  # axis=0 = wavelength axis
plt.figure(figsize=(8, 6))
plt.imshow(white_img.data, origin='lower', cmap='viridis')
plt.colorbar(label='Flux')
plt.title("White-light Subcube Image")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.savefig("/cephfs/apatrick/musecosmos/dataproducts/extractions/wl_subcube_image.png", dpi=300)
print (f"White-light subcube image saved to /cephfs/apatrick/musecosmos/dataproducts/extractions/wl_subcube_image.png")
plt.close()

# 2. Full-cube 1D spectrum (sum over all spatial pixels to get 1D spectrum)
full_spectrum = subcube.sum(axis=(1, 2))  # collapse x and y
plt.figure(figsize=(8, 4))
plt.plot(full_spectrum.wave.coord(), full_spectrum.data, color='k', lw=1)
plt.xlabel("Wavelength [Å]")
plt.ylabel("Flux")
plt.title("Full-cube Spectrum")
plt.grid(alpha=0.3)
plt.savefig("/cephfs/apatrick/musecosmos/dataproducts/extractions/full_subcube_spectrum.png", dpi=300)
print (f"Full-cube spectrum saved to /cephfs/apatrick/musecosmos/dataproducts/extractions/full_subcube_spectrum.png")
plt.close()

"""