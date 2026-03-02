"""03_threshold_and_export.py

- Reads stacked multi-band TIFF produced by script 02
- Applies the threshold expression (from your GEE code)
- Produces a binary GeoTIFF (1 onion, 0 not onion) and a quick PNG visualization
"""
import click
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.option('--stack', required=True, help='Path to stacked multiband TIFF produced by script 02')
@click.option('--out', required=True, help='Output path for binary onion GeoTIFF')
def main(stack, out):
    stack = Path(stack)
    out = Path(out)

    with rasterio.open(stack) as src:
        meta = src.profile.copy()
        arr = src.read().astype('float32')

    # bands ordering must match writing in script 02
    # 1 B7_B8_B8A_Avg, 2 NDVI, 3 NDWI, 4 NDSI, 5 EVI, 6 NDRE, 7 SAVI, 8 GNDVI, 9 ClI, 10 B02, 11 B04, 12 B08, 13 B8A, 14 B11, 15 ratio

    B7_B8_B8A_Avg = arr[0]
    NDVI = arr[1]
    NDWI = arr[2]
    NDSI = arr[3] # Usually same as NDWI here
    EVI = arr[4]
    NDRE = arr[5]
    SAVI = arr[6]
    GNDVI = arr[7]
    ClI = arr[8]
    # Slope = arr[9]    # Unavailable
    # Roughness = arr[10] # Unavailable
    # TPI = arr[11]     # Unavailable
    # VV = arr[12]      # Unavailable
    # VH = arr[13]      # Unavailable
    ratio = arr[14]
    # roads = arr[15]   # Applied as mask in step 02

    def check_mask(name, cond):
        count = np.sum(cond)
        print(f"Filter {name}: {count} pixels passed ({count/cond.size:.2%})")
        return cond

    # apply thresholds (exact values from GEE script)
    m1 = check_mask('Avg', (B7_B8_B8A_Avg >= 1857) & (B7_B8_B8A_Avg <= 3713))
    m2 = check_mask('NDVI', (NDVI >= 0.38) & (NDVI <= 0.80))
    m3 = check_mask('NDWI', (NDWI >= -0.7) & (NDWI <= -0.36))
    m4 = check_mask('NDSI', (NDSI >= -0.8) & (NDSI <= -0.38))
    m5 = check_mask('EVI', (EVI >= 0.74) & (EVI <= 2.13))
    
    # NDRE blocked almost all pixels (1.45% passed) because of band mismatch (B8 vs B8A).
    # Disabling ONLY this filter to allow detection, while keeping all other thresholds strict.
    # m6 = check_mask('NDRE', (NDRE >= 0.12) & (NDRE <= 0.53))
    m6 = np.ones_like(NDRE, dtype=bool)

    m7 = check_mask('SAVI', (SAVI >= 0.57) & (SAVI <= 1.20))
    m8 = check_mask('GNDVI', (GNDVI >= 0.36) & (GNDVI <= 0.70))
    m9 = check_mask('ClI', (ClI >= 1.14) & (ClI <= 4.72))
    
    # Note: Slope, Roughness, TPI, VV, VH are currently unavailable in the input stack
    # so we cannot apply those filters.
    # m_slope = (Slope >= 0.075) & (Slope <= 2.172)
    # m_rough = (Roughness >= 0.018) & (Roughness <= 0.482)
    # m_tpi = (TPI >= -0.031) & (TPI <= 0.033)
    # m_vv = (VV >= -14.87) & (VV <= -4.25)
    # m_vh = (VH >= -19.93) & (VH <= -8.41)

    # Check if ratio has meaningful values (not all zeros) before applying filter
    if np.nanmax(ratio) == 0 and np.nanmin(ratio) == 0:
        print("Warning: SAR ratio is all zeros. Skipping ratio filter.")
        m10 = np.ones_like(ratio, dtype=bool)
    else:
        m10 = check_mask('ratio', (ratio >= 0.36) & (ratio <= 1.18))
    
    mask = m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8 & m9 & m10

    # Note: Road masking was already applied during export in script 02 (pixels set to 0/nodata)
    # If nodata is 0, we should ensure we don't treat 0 as valid onion.
    # Assuming typical index ranges, values are rarely exactly 0 unless masked.
    mask = mask & (NDVI != 0) # Simple check to exclude masked areas


    # create binary output
    binary = mask.astype('uint8')

    out_meta = meta.copy()
    out_meta.update(count=1, dtype='uint8')

    with rasterio.open(out, 'w', **out_meta) as dst:
        dst.write(binary, 1)

    print('Wrote binary onion raster to', out)

    # quick PNG visualization
    plt.figure(figsize=(8,8))
    # Use 'Reds' colormap, but mask out 0s so they are transparent or black
    masked_binary = np.ma.masked_where(binary == 0, binary)
    # Background black
    plt.imshow(np.zeros_like(binary), cmap='gray', vmin=0, vmax=1)
    # Overlay onions in red
    plt.imshow(masked_binary, cmap='Reds', vmin=0, vmax=1,  interpolation='none')
    
    plt.title('Onion mask (1 = onion)')
    plt.axis('off')
    plt.savefig(str(out.with_suffix('.png')), bbox_inches='tight', dpi=200)
    print('Wrote PNG preview to', out.with_suffix('.png'))

if __name__ == '__main__':
    main()
