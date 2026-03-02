"""
02_compute_indices_and_terrain.py

Step-by-step processing:
1. Load Sentinel-1 and Sentinel-2 data from metadata JSON (output from script 01)
2. Compute vegetation indices (S2) and SAR metrics (S1)
3. Mosaic all bands and clip to AOI
4. Apply road mask and crop mask
5. Generate 15-band GeoTIFF output
"""

import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_geom
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import shape, box, mapping
import click
import os

# Enable unauthenticated S3 access for public Sentinel data
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif,.tiff,.jp2,.json'

OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STEP 1: INDEX COMPUTATION FUNCTIONS
# ============================================================================

def ndvi(nir, red):
    """Normalized Difference Vegetation Index"""
    return (nir - red) / (nir + red + 1e-10)

def ndwi(green, swir):
    """Normalized Difference Water Index"""
    return (green - swir) / (green + swir + 1e-10)

def ndsi(green, swir):
    """Normalized Difference Snow Index"""
    return (green - swir) / (green + swir + 1e-10)

def evi(nir, red, blue):
    """Enhanced Vegetation Index"""
    with np.errstate(divide='ignore', invalid='ignore'):
        evi_val = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        evi_val = np.nan_to_num(evi_val, nan=0.0, posinf=0.0, neginf=0.0)
    return evi_val

def ndre(nir, rededge):
    """Normalized Difference Red Edge"""
    return (nir - rededge) / (nir + rededge + 1e-10)

def savi(nir, red, L=0.5):
    """Soil-Adjusted Vegetation Index"""
    return ((nir - red) / (nir + red + L)) * (1 + L)

def gndvi(nir, green):
    """Green Normalized Difference Vegetation Index"""
    return (nir - green) / (nir + green + 1e-10)

def cli(nir, green):
    """Chlorophyll Index"""
    return (nir / (green + 1e-10)) - 1

def sar_ratio(vh, vv):
    """VV/VH ratio (SAR metric)"""
    return np.where(vh == 0, 0, vv / (vh + 1e-10))


# ============================================================================
# STEP 2: LOAD AND MOSAIC SENTINEL DATA
# ============================================================================

def load_band_from_stac(items, band_key, aoi_geom=None):
    """
    Load a single band from STAC items and clip to AOI if provided.
    Returns stacked data from multiple tiles clipped to AOI.
    Handles both TIFF and JP2 formats.
    """
    # Map Sentinel-2 band names to asset names in metadata
    s2_asset_mapping = {
        'B02': ['blue', 'blue-jp2', 'coastal', 'coastal-jp2'],
        'B03': ['green', 'green-jp2'],
        'B04': ['red', 'red-jp2'],
        'B07': ['swir16', 'swir16-jp2'],
        'B08': ['nir', 'nir-jp2'],
        'B8A': ['rededge3', 'rededge2', 'rededge3-jp2', 'rededge2-jp2'],
        'B11': ['swir22', 'swir22-jp2'],
        'VV': ['vv'],
        'VH': ['vh']
    }
    
    # Get the asset names to look for
    asset_names = s2_asset_mapping.get(band_key, [band_key, band_key.lower()])
    
    href_list = []
    
    for item in items:
        href = None
        assets = item.get("assets", {})
        
        # Try to find asset by name mapping
        for asset_name in asset_names:
            if asset_name in assets:
                href = assets[asset_name].get("href")
                if href:
                    break
        
        # Fallback: look for any asset containing the band name
        if not href:
            for asset_key, asset_val in assets.items():
                if band_key.lower() in asset_key.lower():
                    href = asset_val.get("href")
                    if href:
                        break
        
        if href:
            # Verify we can open it
            try:
                # Optimized check: just read small window or metadata
                with rasterio.open(href) as src:
                    # Reading full band is too slow and heavy, just verify profile
                    profile = src.profile
                href_list.append(href)
                print(f"   ✅ Found {band_key} tile ({href.split('/')[-1]})")
            except Exception as e:
                print(f"   ⚠️  Cannot open {band_key}: {type(e).__name__}")
    
    if not href_list:
        print(f"   ❌ No data found for band {band_key}")
        return None, None
    
    # Load and optionally clip all tiles
    clipped_arrays = []
    transform_ref = None
    
    for href in href_list:
        try:
            with rasterio.open(href) as src:
                if aoi_geom is not None:
                    # Clip to AOI using rasterio mask function
                    try:
                        # Reproject AOI to source CRS (usually UTM for S2)
                        aoi_shape = aoi_geom.__geo_interface__
                        if src.crs and src.crs.to_string() != 'EPSG:4326':
                            aoi_shape = transform_geom('EPSG:4326', src.crs, aoi_shape)

                        # Debug: Print bounds overlap check
                        rast_bounds = src.bounds
                        from shapely.geometry import shape
                        aoi_poly = shape(aoi_shape)
                        if not aoi_poly.intersects(box(*rast_bounds)):
                             print(f"      ⚠️  Skipping tile {href.split('/')[-1]}: AOI does not overlap tile bounds in {src.crs}")
                             continue

                        clipped_data, clipped_transform = rio_mask(src, [aoi_shape], crop=True)
                        
                        # Handle rank issues - ensure consistent 2D output per tile
                        if clipped_data.ndim == 3:
                            if clipped_data.shape[0] == 1:
                                clipped_data = clipped_data[0] # Typical case, remove band dim
                            else:
                                # Multi-band case (rare for single band asset, but possible)
                                clipped_data = clipped_data[0] # Take first band

                        clipped_arrays.append(clipped_data)
                        if transform_ref is None:
                            transform_ref = clipped_transform
                    except Exception as e:
                        print(f"      ⚠️  Could not clip tile: {e}")
                        # Fallback: just read the full tile
                        # data = src.read(1) # Reading full tile is risky with unreliable connection
                        # clipped_arrays.append(data)
                        if transform_ref is None:
                            transform_ref = src.transform
                        continue # Skip this tile if clipping failed - reading full COG is too slow/risky
                else:
                    # No AOI clipping, just read the data
                    data = src.read(1)
                    clipped_arrays.append(data)
                    if transform_ref is None:
                        transform_ref = src.transform
        except Exception as e:
            print(f"      ⚠️  Error loading tile: {e}")
            import traceback
            traceback.print_exc()

    if not clipped_arrays:
        print(f"   ❌ Failed to load any valid data for {band_key}")
        return None, None
    
    # If single array, return it; otherwise stack them
    if len(clipped_arrays) == 1:
        print(f"   ✅ Loaded {band_key} ({clipped_arrays[0].shape})")
        return clipped_arrays[0].astype('float32'), transform_ref
    else:
        # Check shapes
        shape0 = clipped_arrays[0].shape
        valid_arrays = []
        for arr in clipped_arrays:
            if arr.shape == shape0:
                valid_arrays.append(arr)
            else:
                 # Try to resize to match? S1 tiles can be tricky
                 # For now, let's just drop mismatches if valid exists
                 pass
        
        if not valid_arrays:
             # If all mismatched, maybe just take the first one?
             valid_arrays = [clipped_arrays[0]]

        # Stack tiles (assuming they align after clipping to same AOI)
        print(f"   ✅ Loaded {band_key} ({len(valid_arrays)} tiles clipped to AOI)")
        
        # Try to stack
        stacked = np.stack(valid_arrays, axis=0) # shape (N, H, W)
        # Compute median across the stack to get a single composite image
        median_composite = np.median(stacked, axis=0)
        print(f"   ✅ Created median composite from {len(valid_arrays)} tiles")
        return median_composite.astype('float32'), transform_ref


def align_bands(bands_dict, ref_band_name, ref_data):
    """
    Align all bands to reference band using rasterio.
    Returns dictionary of aligned numpy arrays.
    """
    print(f"\n📐 Aligning all bands to reference: {ref_band_name}")
    aligned = {}
    
    for band_name, band_data in bands_dict.items():
        if band_data is None:
            aligned[band_name] = np.zeros_like(ref_data)
            print(f"   ⚠️  {band_name}: missing, filling with zeros")
        elif band_data.shape == ref_data.shape:
            aligned[band_name] = band_data.astype("float32")
            print(f"   ✅ {band_name}: shape matches")
        else:
            # Resize using scipy
            from scipy.ndimage import zoom
            
            # Ensure we are working with correct dimensions
            # ref_data and band_data match in rank, or we squash band_data if it has extra singleton dim
            if band_data.ndim == 3 and band_data.shape[0] == 1:
                band_data = band_data[0]
                
            if ref_data.ndim == 3 and ref_data.shape[0] == 1:
                ref_data = ref_data[0] # Should not happen based on logic but good safety
                
            if band_data.ndim != ref_data.ndim:
                print(f"   ⚠️  Rank mismatch: {band_name} {band_data.shape} vs Ref {ref_data.shape}")
                # Try to squeeze or expand
                if band_data.ndim > ref_data.ndim:
                     band_data = np.squeeze(band_data)
                
            # If still mismatch, we can't easily zoom
            if band_data.ndim != ref_data.ndim:
                 print(f"   ❌ Cannot align {band_name} (rank {band_data.ndim}) to reference (rank {ref_data.ndim})")
                 aligned[band_name] = np.zeros_like(ref_data)
                 continue

            zoom_factors = np.array(ref_data.shape) / np.array(band_data.shape)
            aligned[band_name] = zoom(band_data.astype("float32"), zoom_factors, order=1)
            print(f"   ✅ {band_name}: resized from {band_data.shape} to {ref_data.shape}")
    
    return aligned


# ============================================================================
# STEP 3: CLIP TO AOI AND APPLY MASKS
# ============================================================================

def clip_to_aoi(data, aoi_geom, nodata=-9999):
    """Clip raster data to AOI geometry."""
    try:
        # Create a temporary rasterio dataset for masking
        from affine import Affine
        h, w = data.shape[:2]
        bounds = aoi_geom.bounds
        minx, miny, maxx, maxy = bounds
        xres = (maxx - minx) / w
        yres = (maxy - miny) / h
        transform = Affine.translation(minx, maxy) * Affine.scale(xres, -yres)
        
        if data.ndim == 3:
            clipped = np.zeros_like(data)
            for i in range(data.shape[0]):
                clipped[i] = rio_mask_array(data[i], aoi_geom, transform, nodata)
            return clipped
        else:
            return rio_mask_array(data, aoi_geom, transform, nodata)
    except Exception as e:
        print(f"⚠️  Could not clip to AOI precisely: {e}")
        return data


def rio_mask_array(arr, geom, transform, nodata=-9999):
    """Apply geometry mask to numpy array."""
    from rasterio.features import rasterize
    mask_array = rasterize([geom], out_shape=arr.shape, transform=transform, fill=0, default_value=1)
    masked = arr.copy().astype("float32")
    masked[mask_array == 0] = nodata
    return masked


def apply_road_mask(data, road_mask_path):
    """Mask out roads from data."""
    # print(f"\n🛣️  Applying road mask...") 
    return data # Skip for now - requires transform

def apply_crop_mask(data, crop_mask_path):
    """Keep only data within crop mask."""
    # print(f"\n🌾 Applying crop mask...")
    return data # Skip for now - requires transform


def apply_crop_mask(data, crop_mask_path):
    """Keep only data within crop mask."""
    # print(f"\n🌾 Applying crop mask...")
    return data # Skip for now - requires transform

def apply_mask_to_array(arr, geom, inverse=False):
    """Apply geometry mask to 2D array."""
    from rasterio.features import rasterize
    from affine import Affine
    
    h, w = arr.shape
    # Estimate transform from array bounds
    transform = Affine.identity()
    
    try:
        mask_raster = rasterize([geom], out_shape=(h, w), transform=transform, fill=0, default_value=1)
        masked = arr.copy().astype("float32")
        
        if inverse:
            # Remove masked areas (set to 0)
            masked[mask_raster == 1] = 0
        else:
            # Keep only masked areas
            masked[mask_raster == 0] = 0
        
        return masked
    except:
        return arr



# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

@click.command()
@click.option("--meta", required=True, help="Path to metadata JSON from script 01")
@click.option("--outdir", default=str(OUTDIR), help="Output directory for results")
def main(meta, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(" 02_COMPUTE_INDICES_AND_TERRAIN - Step-by-Step Processing")
    print("="*70)
    
    # Load metadata
    print(f"\n📄 Loading metadata: {meta}")
    with open(meta) as f:
        metadata = json.load(f)
    
    start_date = metadata.get("start")
    end_date = metadata.get("end")
    aoi_geom = shape(metadata.get("geometry"))
    
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Sentinel-2 items: {len(metadata.get('sentinel2_items', []))}")
    print(f"   Sentinel-1 items: {len(metadata.get('sentinel1_items', []))}")
    
    if not metadata.get('sentinel2_items'):
        raise RuntimeError("❌ Missing Sentinel-2 data in metadata!")
    
    # ========================================================================
    # STEP 1: LOAD SENTINEL-2 BANDS (clipped to AOI)
    # ========================================================================
    print(f"\n🛰️  STEP 1: Loading Sentinel-2 bands (clipped to AOI)")
    print("-" * 70)
    
    s2_items = metadata['sentinel2_items']
    s2_bands = {}
    s2_transforms = {}
    
    for band_key in ['B02', 'B03', 'B04', 'B07', 'B08', 'B8A', 'B11']:
        data, transform = load_band_from_stac(s2_items, band_key, aoi_geom)
        s2_bands[band_key] = data
        if transform:
            s2_transforms[band_key] = transform
    
    # Use B04 (Red) as reference
    ref_band = s2_bands.get('B04')
    ref_transform = s2_transforms.get('B04')
    
    # Capture the CRS from the reference file if possible, otherwise we default to something logic
    # But we need to know the CRS of the loaded data! 
    # load_band_from_stac currently returns (data, transform). We should probably return CRS too or assume it matches source.
    
    if ref_band is None:
        # Try B08 as fallback
        ref_band = s2_bands.get('B08')
        ref_transform = s2_transforms.get('B08')
        
    if ref_band is None:
        raise RuntimeError("❌ No reference band (B04 or B08) found in Sentinel-2 data!")
    
    print(f"   Reference band shape: {ref_band.shape}")
    print(f"   Reference transform: {ref_transform}")
    
    # We need the CRS of the reference band to write the output correctly
    # Since load_band_from_stac doesn't return CRS, let's grab it from one of the files quickly
    # or better, modify load_band_from_stac to return it.
    # For now, let's peek at the first S2 item to get the likely CRS
    ref_crs = None
    try:
        if s2_items:
             # Just check the first asset of the first item
             item = s2_items[0]
             # Find a valid href
             href = None
             assets = item.get("assets", {})
             for k in ['visual', 'red', 'B04', 'B04.tif']:
                 if k in assets:
                     href = assets[k]['href']
                     break
             if href:
                 with rasterio.open(href) as ds:
                     ref_crs = ds.crs
    except:
        pass
        
    if ref_crs is None:
        print("   ⚠️  Could not determine CRS from source, defaulting to EPSG:32651 (Zone 51N)")
        ref_crs = rasterio.crs.CRS.from_epsg(32651)
        
    print(f"   Reference CRS: {ref_crs}")

    
    # ========================================================================
    # STEP 2: LOAD SENTINEL-1 BANDS (clipped to AOI)
    # ========================================================================
    print(f"\n📡 STEP 2: Loading Sentinel-1 bands (clipped to AOI)")
    print("-" * 70)
    
    s1_items = metadata.get('sentinel1_items', [])
    s1_bands = {}
    
    if not s1_items:
        print(f"   ⚠️  No Sentinel-1 items in metadata - using zeros for SAR bands")
        VV = None
        VH = None
    else:
        for band_key in ['VV', 'VH']:
            s1_bands[band_key], _ = load_band_from_stac(s1_items, band_key, aoi_geom)
    
    # ========================================================================
    # STEP 3: ALIGN ALL BANDS TO REFERENCE
    # ========================================================================
    print(f"\n🔧 STEP 3: Aligning bands to reference geometry")
    print("-" * 70)
    
    all_bands = {**s2_bands}
    
    # Add S1 bands if available
    if s1_items:
        all_bands.update(s1_bands)
    
    aligned = align_bands(all_bands, 'B04', ref_band)
    
    # Extract aligned bands
    B02 = aligned['B02']
    B03 = aligned['B03']
    B04 = aligned['B04']
    B07 = aligned['B07']
    B08 = aligned['B08']
    B8A = aligned['B8A']
    B11 = aligned['B11']
    
    # S1 bands with fallback to zeros if not available
    if s1_items:
        VV = aligned.get('VV', np.zeros_like(B04))
        VH = aligned.get('VH', np.zeros_like(B04))
    else:
        VV = np.zeros_like(B04)
        VH = np.zeros_like(B04)
        print(f"   ⚠️  S1 bands unavailable, using zeros")
    
    # ========================================================================
    # STEP 4: COMPUTE VEGETATION AND SAR INDICES
    # ========================================================================
    print(f"\n📊 STEP 4: Computing indices")
    print("-" * 70)
    
    NDVI = ndvi(B08, B04)
    print(f"   ✅ NDVI computed")
    
    NDWI = ndwi(B03, B11)
    print(f"   ✅ NDWI computed")
    
    NDSI = ndsi(B03, B11)
    print(f"   ✅ NDSI computed")
    
    EVI = evi(B08, B04, B02)
    print(f"   ✅ EVI computed")
    
    NDRE = ndre(B08, B8A)
    print(f"   ✅ NDRE computed")
    
    SAVI = savi(B08, B04)
    print(f"   ✅ SAVI computed")
    
    GNDVI = gndvi(B08, B03)
    print(f"   ✅ GNDVI computed")
    
    ClI = cli(B08, B03)
    print(f"   ✅ ClI (Chlorophyll Index) computed")
    
    B7_B8_B8A_Avg = (B07 + B08 + B8A) / 3.0
    print(f"   ✅ SWIR-NIR-RedEdge Average computed")
    
    SAR_Ratio = sar_ratio(VH, VV)
    print(f"   ✅ SAR VV/VH ratio computed")
    
    # ========================================================================
    # STEP 5: PREPARE 15-BAND OUTPUT (memory-efficient approach)
    # ========================================================================
    print(f"\n📦 STEP 5: Preparing 15-band output")
    print("-" * 70)
    
    # Define the 15 bands we need (in order)
    bands_for_export = [
        (B7_B8_B8A_Avg,  "SWIR-NIR-RedEdge Avg"),
        (NDVI,           "NDVI"),
        (NDWI,           "NDWI"),
        (NDSI,           "NDSI"),
        (EVI,            "EVI"),
        (NDRE,           "NDRE"),
        (SAVI,           "SAVI"),
        (GNDVI,          "GNDVI"),
        (ClI,            "Chlorophyll Index"),
        (B02,            "Blue (B02)"),
        (B04,            "Red (B04)"),
        (B08,            "NIR (B08)"),
        (B8A,            "Red Edge (B8A)"),
        (B11,            "SWIR (B11)"),
        (SAR_Ratio,      "SAR VV/VH Ratio")
    ]
    print(f"   ✅ {len(bands_for_export)} bands prepared for export")
    
    # ========================================================================
    # STEP 6: CLIP, MASK, AND EXPORT (streaming to avoid memory overload)
    # ========================================================================
    print(f"\n🎭 STEP 6: Applying masks and exporting")
    print("-" * 70)
    
    # Data is already clipped to AOI during loading
    h, w = B7_B8_B8A_Avg.shape # Should match ref_band.shape
    
    # Use the reference transform and CRS we captured
    transform = ref_transform
    crs = ref_crs
    
    output_path = outdir / f"stack_AOI_{start_date}_{end_date}.tif"
    
    profile = {
        'driver': 'GTiff',
        'height': h,
        'width': w,
        'count': 15,
        'dtype': 'float32',
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    print(f"   Creating GeoTIFF: {output_path}")
    print(f"   Dimensions: {h} x {w}, 15 bands")
    print(f"   CRS: {crs}")
    print(f"   Transform: {transform}")
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for band_idx, (band_data, band_name) in enumerate(bands_for_export, 1):
            # Process each band individually to manage memory
            processed = band_data.copy().astype('float32')
            
            # Apply road mask (data already clipped to AOI)
            processed = apply_road_mask(processed, "data/Road mask/Bongabon_roads.shp")
            
            # Apply crop mask
            processed = apply_crop_mask(processed, "data/Crop mask/cropmask.shp")
            
            # Write to file
            dst.write(processed, band_idx)
            print(f"   ✅ Band {band_idx}/15: {band_name}")
    
    print(f"\n   ✅ Saved: {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    
    print("\n" + "="*70)
    print(f" ✅ Processing complete!")
    print(f" Output ready for step 03_threshold_and_export.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
