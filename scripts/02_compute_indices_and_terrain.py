"""
02_compute_indices_and_terrain.py

Processes:
- Load metadata JSON output from Script 01
- Determine AOI from saved province geometry (GADM)
- Select first Sentinel-2 & Sentinel-1 items
- Compute vegetation indices + SAR metrics
- Compute terrain derivatives from DEM
- Download & rasterize OSM roads
- Export all as aligned multiband GeoTIFF
"""

import json
from pathlib import Path
import numpy as np
import rasterio
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import shape, box
import osmnx as ox
from rasterio.features import rasterize
from scipy import ndimage
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


# ------------------ DEM Derivatives ------------------ #
def compute_slope(dem, transform):
    dx = transform[0]
    dy = -transform[4] if transform[4] != 0 else 1.0
    x, y = np.gradient(dem, dx, dy)
    return np.sqrt(x*x + y*y)

def compute_roughness(dem):
    return ndimage.generic_filter(dem, np.std, size=3)

def compute_tpi(dem):
    local_mean = ndimage.uniform_filter(dem, size=11)
    return dem - local_mean


@click.command()
@click.option("--meta", required=True)
@click.option("--outdir", default=str(OUTDIR))
def main(meta, outdir):

    outdir = Path(outdir)

    with open(meta) as f:
        m = json.load(f)

    # ✅ AOI from GADM geometry stored in JSON from Script #1
    aoi_geom = shape(m["geometry"])
    gdf = gpd.GeoDataFrame({"geometry": [aoi_geom]}, crs="EPSG:4326")

    # ✅ Extract first S2 + S1 items
    if not m["sentinel2_items"]:
        raise RuntimeError("❌ No Sentinel-2 items in metadata!")
    if not m["sentinel1_items"]:
        raise RuntimeError("❌ No Sentinel-1 items in metadata!")

    s2_item = m["sentinel2_items"][0]
    s1_item = m["sentinel1_items"][0]

    print("✅ Loaded first Sentinel-2 + Sentinel-1 scenes")


    # ------------------ Sentinel-2 Asset Finder ------------------ #
    def find_asset(item, keywords):
        for k, v in item["assets"].items():
            href = v.get("href", "")
            if any(tok in k.upper() or tok in href.upper() for tok in keywords):
                return href
        return None

    bands = {
        "B02": find_asset(s2_item, ["B02"]),
        "B03": find_asset(s2_item, ["B03"]),
        "B04": find_asset(s2_item, ["B04"]),
        "B07": find_asset(s2_item, ["B07"]),
        "B08": find_asset(s2_item, ["B08"]),
        "B8A": find_asset(s2_item, ["B8A"]),
        "B11": find_asset(s2_item, ["B11"]),
    }

    print("📌 Loaded S2 bands:", bands)

    xrds = {k: rxr.open_rasterio(v, masked=True).squeeze()
            for k, v in bands.items()}

    ref = xrds["B04"]
    for k in xrds:
        xrds[k] = xrds[k].rio.reproject_match(ref).astype("float32")

    B2, B3, B4, B7, B8, B8A, B11 = [xrds[b].data for b in bands.keys()]

    # ------------------ Compute Indices ------------------ #
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
    def find_s1(item, pol):
        return find_asset(item, [pol])

    VV = rxr.open_rasterio(find_s1(s1_item, "VV")).squeeze()
    VH = rxr.open_rasterio(find_s1(s1_item, "VH")).squeeze()

    VV = VV.rio.reproject_match(ref).data.astype("float32")
    VH = VH.rio.reproject_match(ref).data.astype("float32")

    ratio = np.where(VH == 0, 0, VV / (VH + 1e-10))


    # ------------------ DEM via Local SRTM ------------------ #
    dem_path = Path("data/dem/srtm.tif")
    if dem_path.exists():
        dem_da = rxr.open_rasterio(str(dem_path)).squeeze().rio.reproject_match(ref)
        dem = dem_da.data.astype("float32")
        transform = dem_da.rio.transform()
    else:
        print("⚠️ DEM missing — Filling with zeros")
        dem = np.zeros_like(B4)
        transform = ref.rio.transform()

    Slope = compute_slope(dem, transform)
    Roughness = compute_roughness(dem)
    TPI = compute_tpi(dem)


    # ------------------ OSM Roads ------------------ #
    print("🚧 Downloading OSM Roads...")
    minx, miny, maxx, maxy = gdf.total_bounds
    roads_graph = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type="drive")
    roads = ox.graph_to_gdfs(roads_graph, nodes=False, edges=True)
    roads_geom = [(geom, 1) for geom in roads.geometry]

    roads_raster = rasterize(
        roads_geom,
        out_shape=B4.shape,
        transform=ref.rio.transform(),
        fill=0,
        dtype="uint8"
    )


    # ------------------ Multiband Export ------------------ #
    stack = np.stack([
        B7_B8_B8A_Avg, NDVI, NDWI, NDSI, EVI, NDRE, SAVI, GNDVI, ClI,
        Slope, Roughness, TPI, VV, VH, ratio, roads_raster
    ])

    out_path = outdir / f"stack_{m['province']}_{m['start']}_{m['end']}.tif"

    profile = ref.rio.to_rasterio().profile
    profile.update(count=stack.shape[0], dtype="float32")

    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i].astype("float32"), i + 1)

    print(f"✅ Saved output: {out_path}")


if __name__ == "__main__":
    main()
