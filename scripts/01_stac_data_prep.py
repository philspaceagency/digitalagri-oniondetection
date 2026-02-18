"""
    01_stac_data_prep.py

    Processes:
    - Loads shapefile from /data folder
    - creates stac for Sentinel-1 and Sentinel-2
    - clips created stac within shapefile 
""" 

import argparse
import geopandas as gpd
from shapely.geometry import box
from pystac_client import Client

def load_aoi_shapefile():
    """Load AOI shapefile from data/AOI/bongabon_aoi.shp and return GeoDataFrame and bounds."""
    aoi_path = "data/AOI/bongabon_aoi.shp"
    gdf = gpd.read_file(aoi_path)
    print(f"✅ AOI shapefile loaded: {aoi_path}")
    print(f"Bounds: {gdf.total_bounds}")
    return gdf, gdf.total_bounds

STAC_API = "https://earth-search.aws.element84.com/v1"


def load_province_shapefile(province_name: str):
    print(f"🔍 Loading province boundary for: {province_name}")

    gdf = gpd.read_file(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_PHL.gpkg",
        layer="ADM_ADM_2"
    )

    province_name = province_name.upper().strip()
    gdf["NAME_1"] = gdf["NAME_1"].str.upper().str.strip()
    selected = gdf[gdf["NAME_1"] == province_name]

    if selected.empty:
        raise ValueError(f"❌ Province '{province_name}' not found!")

    print("✅ Province loaded successfully!")
    return selected.to_crs(4326)


def get_tiles(province_gdf, resolution=1.0):
    bounds = province_gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, x + resolution, y + resolution)
            if province_gdf.intersects(tile).any():
                tiles.append(tile)
            y += resolution
        x += resolution

    print(f"✅ Generated {len(tiles)} tiles for sampling")
    return tiles


def search_sentinel2(bounds, start_date, end_date, cloud_cover):
    client = Client.open(STAC_API)
    bbox = list(bounds)

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_cover}}
    )

    items = list(search.items())
    print(f"🟦 Sentinel-2 found: {len(items)} items")
    return items


def search_sentinel1(bounds, start_date, end_date):
    client = Client.open(STAC_API)
    bbox = list(bounds)

    search = client.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "sar:instrument_mode": {"eq": "IW"},
            "sar:product_type": {"eq": "GRD"},
            "sar:polarizations": {"intersects": ["VV", "VH"]}
        }
    )

    items = list(search.items())
    print(f"⬛ Sentinel-1 found: {len(items)} items")
    return items

def main():

    parser = argparse.ArgumentParser(description="STAC download automation (AOI only)")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--cloud", type=int, default=20)
    args = parser.parse_args()

    # Load AOI shapefile and get tiles
    aoi_gdf, aoi_bounds = load_aoi_shapefile()
    tiles = get_tiles(aoi_gdf)

    s2_list = []
    s1_list = []

    for i, tile in enumerate(tiles):
        print(f"\n📌 Tile {i+1}/{len(tiles)}...")
        s2_items = search_sentinel2(tile.bounds, args.start, args.end, args.cloud)
        s1_items = search_sentinel1(tile.bounds, args.start, args.end)
        s2_list.extend(s2_items)
        s1_list.extend(s1_items)

    print("\n🎯 RESULTS SUMMARY")
    print(f"🟦 Sentinel-2 TOTAL: {len(set([i.id for i in s2_list]))}")
    print(f"⬛ Sentinel-1 TOTAL: {len(set([i.id for i in s1_list]))}")

    # Print acquisition dates for Sentinel-2
    print("\nSentinel-2 acquisition dates:")
    for item in s2_list:
        date = getattr(item, 'datetime', None)
        if date:
            print(date)

    # Print acquisition dates for Sentinel-1
    print("\nSentinel-1 acquisition dates:")
    for item in s1_list:
        date = getattr(item, 'datetime', None)
        if date:
            print(date)


if __name__ == "__main__":
    main()