from astropy.io import fits
import os

fits_dir = "/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full"
start_id, end_id = 0, 3680

bad_files = []

for i in range(start_id, end_id + 1):
    fname = f"mosaic_slice_{i}.fits"
    fpath = os.path.join(fits_dir, fname)
    try:
        with fits.open(fpath, ignore_missing_simple=True) as hdul:
            _ = hdul[0].data  # try reading data to ensure file is valid
    except Exception as e:
        bad_files.append((fname, str(e)))

print("==== FITS Validation Check ====")
if not bad_files:
    print("✅ All FITS files opened successfully!")
else:
    print(f"❌ {len(bad_files)} corrupt or unreadable FITS files:")
    for fname, err in bad_files[:10]:
        print(f"   {fname} → {err}")
    if len(bad_files) > 10:
        print("   ...")
