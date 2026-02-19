"""
02_compute_indices_and_terrain.py

Processes:
- Load metadata JSON output from Script 01
- Determine AOI from saved province geometry (GADM)
- Select first Sentinel-2 & Sentinel-1 items
- Compute vegetation indices + SAR metrics

- Export all as aligned multiband GeoTIFF
"""

import json
from pathlib import Path
import numpy as np
import rasterio
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import shape, box
from rasterio.features import rasterize
from scipy import ndimage
import click


OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------------ Sentinel-2 Indices ------------------ #

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


# ------------------ Terrain Variables ------------------ #
def get_slope(dem, transform):
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
            candidates = []
            for k, v in item["assets"].items():
                href = v.get("href", "")
                if any(tok in k.upper() or tok in href.upper() for tok in keywords):
                    candidates.append(href)
            if not candidates:
                return None
            # Prefer GeoTIFF assets when available
            for c in candidates:
                if str(c).lower().endswith('.tif') or str(c).lower().endswith('.tiff'):
                    return c
            return candidates[0]

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

    def safe_open(path, name=None, masked=True):
        try:
            display_name = name or str(path)
            print(f"Opening raster: {display_name}")
            # If opening from S3 public bucket, allow unsigned (no AWS creds)
            if isinstance(path, str) and path.startswith('s3://'):
                import rasterio
                with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
                    da = rxr.open_rasterio(path, masked=masked).squeeze()
            else:
                da = rxr.open_rasterio(path, masked=masked).squeeze()
            return da
        except Exception as e:
            print(f"ERROR: failed to open raster '{path}' ({display_name}): {e}")
            raise

    xrds = {}
    for k, v in bands.items():
        if v is None:
            raise RuntimeError(f"Missing asset for band {k}")
        xrds[k] = safe_open(v, name=k)

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

    vv_href = find_s1(s1_item, "VV")
    vh_href = find_s1(s1_item, "VH")
    if not vv_href or not vh_href:
        raise RuntimeError("Sentinel-1 VV/VH assets not found in metadata item")

    VV = safe_open(vv_href, name="VV")
    VH = safe_open(vh_href, name="VH")

    # attempt reprojection; if the source is corrupt, fallback to zeros
    try:
        VV = VV.rio.reproject_match(ref).data.astype("float32")
    except Exception as e:
        print(f"⚠️ reprojection of VV failed ({e}) – using zeros")
        VV = np.zeros_like(B4, dtype="float32")
    try:
        VH = VH.rio.reproject_match(ref).data.astype("float32")
    except Exception as e:
        print(f"⚠️ reprojection of VH failed ({e}) – using zeros")
        VH = np.zeros_like(B4, dtype="float32")

    ratio = np.where(VH == 0, 0, VV / (VH + 1e-10))

    # ------------------ Load Precomputed Terrain Rasters ------------------ #
    # These rasters will be reprojected/matched to the Sentinel-2 reference grid
    def load_and_match(path, name):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Terrain file missing: {p}")
        da = rxr.open_rasterio(str(p)).squeeze().rio.reproject_match(ref)
        print(f"✅ Loaded {name}: {p}")
        return da.data.astype("float32")

    Slope = load_and_match("data/Bongabon_Slope_wgs84.tif", "Slope")
    Roughness = load_and_match("data/Bongabon_Roughness_wgs84.tif", "Roughness")
    TPI = load_and_match("data/Bongabon_TopographicPositionIndex_wgs84.tif", "TPI")

    # ------------------ Load and Apply Crop and Road Masks (GEE-style) ------------------ #
    transform = ref.rio.transform()
    out_shape = B4.shape
    crs = ref.rio.crs

    # Load crop mask and rasterize (keep pixels inside crop)
    try:
        crop_gdf = gpd.read_file("data/Crop mask/cropmask.shp").to_crs(crs)
        crop_geoms = [(g, 1) for g in crop_gdf.geometry if g is not None]
        if crop_geoms:
            crop_raster = rasterize(crop_geoms, out_shape=out_shape, transform=transform, fill=0, dtype="uint8")
            print("✅ Rasterized crop mask")
        else:
            print("⚠️ Crop mask contains no geometries; using full extent")
            crop_raster = np.ones(out_shape, dtype="uint8")
    except Exception as e:
        print(f"⚠️ Could not load crop mask: {e}; using full extent")
        crop_raster = np.ones(out_shape, dtype="uint8")

    # Load roads, buffer by 10m, and rasterize; then invert (keep non-road pixels)
    try:
        roads_gdf = gpd.read_file("data/Road mask/Bongabon_roads.shp").to_crs(crs)
        # Buffer roads by 10 meters
        roads_buffered = roads_gdf.copy()
        roads_buffered['geometry'] = roads_buffered.geometry.buffer(10)
        roads_geoms = [(g, 1) for g in roads_buffered.geometry if g is not None]
        if roads_geoms:
            roads_raster = rasterize(roads_geoms, out_shape=out_shape, transform=transform, fill=0, dtype="uint8")
            # Invert: 1 where NO roads, 0 where roads exist (like GEE .not())
            roads_mask = 1 - roads_raster
            print("✅ Rasterized and inverted road mask (10m buffer)")
        else:
            print("⚠️ Road mask contains no geometries; keeping all pixels")
            roads_mask = np.ones(out_shape, dtype="uint8")
    except UnicodeDecodeError:
        print("⚠️ Unicode error reading roads mask; keeping all pixels")
        roads_mask = np.ones(out_shape, dtype="uint8")
    except Exception as e:
        print(f"⚠️ Could not load roads mask: {e}; keeping all pixels")
        roads_mask = np.ones(out_shape, dtype="uint8")

    # Combined mask: keep pixels inside crop AND not on roads
    combined_mask = crop_raster * roads_mask


    # ------------------ Stack Export ------------------ #
    # Compose stack: S2 indices + terrain + S1 bands
    stack = np.stack([
        B7_B8_B8A_Avg, NDVI, NDWI, NDSI, EVI, NDRE, SAVI, GNDVI, ClI,
        Slope, Roughness, TPI, VV, VH, ratio
    ])

    # Mask out crop and road areas (keep only pixels inside crop AND not on roads)
    print("🔒 Applying crop and road masks to composite")
    for i in range(stack.shape[0]):
        stack[i] = stack[i] * combined_mask  # 0 where outside crop or on roads

    out_path = outdir / f"stack_{m['province']}_{m['start']}_{m['end']}.tif"

    # Build rasterio profile robustly
    try:
        profile = ref.rio.to_rasterio().profile
    except Exception:
        print("⚠️ rioxarray.to_rasterio() not available — constructing profile manually")
        transform = ref.rio.transform()
        crs = ref.rio.crs
        height, width = ref.shape
        profile = {
            "driver": "GTiff",
            "height": int(height),
            "width": int(width),
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }
    profile.update(count=stack.shape[0], dtype="float32")

    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i].astype("float32"), i + 1)

    print(f"✅ Saved output: {out_path}")


if __name__ == "__main__":
    main()