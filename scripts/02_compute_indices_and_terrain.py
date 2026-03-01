"""
02_compute_indices_and_terrain.py

Processes:
- Load metadata JSON output from Script 01
- Select first Sentinel-2 & Sentinel-1 items
- Compute vegetation indices + SAR metrics
- Load terrain variables from precomputed IfSAR rasters
- Load crop and road masks from shapefiles
- Rasterize masks to match reference grid
- Apply mask (crop only, exclude roads)
- Export aligned multiband GeoTIFF clipped to AOI
"""

import json
from pathlib import Path
import numpy as np
import rasterio
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import rasterize
import click


OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------------ Index Helpers ------------------ #
def ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)

def ndwi(green, swir):
    return (green - swir) / (green + swir + 1e-10)

def ndsi(green, swir):
    return (green - swir) / (green + swir + 1e-10)

def evi(nir, red, blue):
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

def ndre(nir, rededge):
    return (nir - rededge) / (nir + rededge + 1e-10)

def savi(nir, red, L=0.5):
    return ((nir - red) / (nir + red + L)) * (1 + L)

def gndvi(nir, green):
    return (nir - green) / (nir + green + 1e-10)

def cli(b8, b3):
    return (b8 / (b3 + 1e-10)) - 1


@click.command()
@click.option("--meta", required=True)
@click.option("--outdir", default=str(OUTDIR))
def main(meta, outdir):

    outdir = Path(outdir)

    # ------------------ Load Metadata ------------------ #
    with open(meta) as f:
        m = json.load(f)

    aoi_geom = shape(m["geometry"])
    gdf_aoi = gpd.GeoDataFrame({"geometry": [aoi_geom]}, crs="EPSG:4326")

    s2_item = m["sentinel2_items"][0]
    s1_item = m["sentinel1_items"][0]


    # ------------------ Asset Finder ------------------ #
    def find_asset(item, keywords):
        for k, v in item["assets"].items():
            href = v.get("href", "")
            if any(tok in k.upper() or tok in href.upper() for tok in keywords):
                return href
        return None


    # ------------------ Sentinel-2 ------------------ #
    bands = {
        "B02": find_asset(s2_item, ["B02"]),
        "B03": find_asset(s2_item, ["B03"]),
        "B04": find_asset(s2_item, ["B04"]),
        "B07": find_asset(s2_item, ["B07"]),
        "B08": find_asset(s2_item, ["B08"]),
        "B8A": find_asset(s2_item, ["B8A"]),
        "B11": find_asset(s2_item, ["B11"]),
    }

    xrds = {}
    for k, v in bands.items():
        da = rxr.open_rasterio(v, masked=True).squeeze()
        da = da.rio.clip(gdf_aoi.geometry, gdf_aoi.crs)
        xrds[k] = da

    ref = xrds["B04"]

    for k in xrds:
        xrds[k] = xrds[k].rio.reproject_match(ref).astype("float32")

    B2, B3, B4, B7, B8, B8A, B11 = [xrds[b].data for b in bands.keys()]


    # ------------------ Indices ------------------ #
    NDVI = ndvi(B8, B4)
    NDWI = ndwi(B3, B11)
    NDSI = ndsi(B3, B11)
    EVI = evi(B8, B4, B2)
    NDRE = ndre(B8, B8A)
    SAVI = savi(B8, B4)
    GNDVI = gndvi(B8, B3)
    ClI = cli(B8, B3)
    B7_B8_B8A_Avg = (B7 + B8 + B8A) / 3.0


    # ------------------ Sentinel-1 ------------------ #
    VV = rxr.open_rasterio(find_asset(s1_item, ["VV"]), masked=True).squeeze()
    VH = rxr.open_rasterio(find_asset(s1_item, ["VH"]), masked=True).squeeze()

    VV = VV.rio.clip(gdf_aoi.geometry, gdf_aoi.crs).rio.reproject_match(ref).data.astype("float32")
    VH = VH.rio.clip(gdf_aoi.geometry, gdf_aoi.crs).rio.reproject_match(ref).data.astype("float32")

    ratio = np.where(VH == 0, 0, VV / (VH + 1e-10))


    # ------------------ Load IfSAR Terrain ------------------ #
    terrain_paths = {
        "Slope": "data/Bongabon_Slope_wgs84.tif",
        "Roughness": "data/Bongabon_Roughness_wgs84.tif",
        "TPI": "data/Bongabon_TopographicPositionIndex_wgs84.tif"
    }

    terrain = {}

    for name, path in terrain_paths.items():
        da = rxr.open_rasterio(path, masked=True).squeeze()
        da = da.rio.clip(gdf_aoi.geometry, gdf_aoi.crs)
        da = da.rio.reproject_match(ref)
        terrain[name] = da.data.astype("float32")

    Slope = terrain["Slope"]
    Roughness = terrain["Roughness"]
    TPI = terrain["TPI"]


    # ------------------ Load & Rasterize Shapefile Masks ------------------ #
    print("Rasterizing crop and road masks...")

    crop_gdf = gpd.read_file("data/Crop_mask/cropmask.shp").to_crs(ref.rio.crs)
    road_gdf = gpd.read_file("data/Road_mask/Bongabon_roads.shp").to_crs(ref.rio.crs)

    crop_shapes = [(geom, 1) for geom in crop_gdf.geometry]
    road_shapes = [(geom, 1) for geom in road_gdf.geometry]

    transform = ref.rio.transform()
    out_shape = ref.shape

    crop_mask = rasterize(
        crop_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    road_mask = rasterize(
        road_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    valid_mask = (crop_mask == 1) & (road_mask == 0)

    def apply_mask(arr):
        return np.where(valid_mask, arr, np.nan)


    arrays = [
        B7_B8_B8A_Avg, NDVI, NDWI, NDSI, EVI,
        NDRE, SAVI, GNDVI, ClI,
        Slope, Roughness, TPI,
        VV, VH, ratio
    ]

    arrays = [apply_mask(a) for a in arrays]


    # ------------------ Export ------------------ #
    stack = np.stack(arrays)

    out_path = outdir / f"stack_{m['province']}_{m['start']}_{m['end']}.tif"

    profile = ref.rio.to_rasterio().profile
    profile.update(count=stack.shape[0], dtype="float32")

    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i].astype("float32"), i + 1)

    print(f"Saved output: {out_path}")


if __name__ == "__main__":
    main()