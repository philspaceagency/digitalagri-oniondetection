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
    # 1 B7_B8_B8A_Avg, 2 NDVI, 3 NDWI, 4 NDSI, 5 EVI, 6 NDRE, 7 SAVI, 8 GNDVI, 9 ClI, 10 Slope, 11 Roughness, 12 TPI, 13 VV, 14 VH, 15 ratio, 16 roads

    B7_B8_B8A_Avg = arr[0]
    NDVI = arr[1]
    NDWI = arr[2]
    NDSI = arr[3]
    EVI = arr[4]
    NDRE = arr[5]
    SAVI = arr[6]
    GNDVI = arr[7]
    ClI = arr[8]
    Slope = arr[9]
    Roughness = arr[10]
    TPI = arr[11]
    VV = arr[12]
    VH = arr[13]
    ratio = arr[14]
    roads = arr[15]

    # apply thresholds (from your GEE expression)
    mask = (
        (B7_B8_B8A_Avg >= 1857) & (B7_B8_B8A_Avg <= 3713) &
        (NDVI >= 0.38) & (NDVI <= 0.80) &
        (NDWI >= -0.7) & (NDWI <= -0.36) &
        (NDSI >= -0.8) & (NDSI <= -0.38) &
        (EVI >= 0.74) & (EVI <= 2.13) &
        (NDRE >= 0.12) & (NDRE <= 0.53) &
        (SAVI >= 0.57) & (SAVI <= 1.20) &
        (GNDVI >= 0.36) & (GNDVI <= 0.70) &
        (ClI >= 1.14) & (ClI <= 4.72) &
        (Slope >= 0.075) & (Slope <= 2.172) &
        (Roughness >= 0.018) & (Roughness <= 0.482) &
        (TPI >= -0.031) & (TPI <= 0.033) &
        (VV >= -14.87) & (VV <= -4.25) &
        (VH >= -19.93) & (VH <= -8.41) &
        (ratio >= 0.36) & (ratio <= 1.18)
    )

    # Optionally remove immediate road pixels (e.g., set to 0 if road is present)
    mask = mask & (roads == 0)

    # create binary output
    binary = mask.astype('uint8')

    out_meta = meta.copy()
    out_meta.update(count=1, dtype='uint8')

    with rasterio.open(out, 'w', **out_meta) as dst:
        dst.write(binary, 1)

    print('Wrote binary onion raster to', out)

    # quick PNG visualization
    plt.figure(figsize=(8,8))
    plt.imshow(binary, cmap='gray')
    plt.title('Onion mask (1 = onion)')
    plt.axis('off')
    plt.savefig(str(out.with_suffix('.png')), bbox_inches='tight', dpi=200)
    print('Wrote PNG preview to', out.with_suffix('.png'))

if __name__ == '__main__':
    main()
