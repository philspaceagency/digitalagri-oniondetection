"""01_stac_data_prep.py

Load AOI from `data/AOI/bongabon_aoi.shp`, search STAC for Sentinel-2 and Sentinel-1
items that intersect the AOI and write a meta JSON usable by
`scripts/02_compute_indices_and_terrain.py`.

Optional `--run` will invoke `02_compute_indices_and_terrain.py` with the
generated meta file.
"""
import argparse
import json
from pathlib import Path


"""
01_stac_data_prep.py

Load AOI from `data/AOI/bongabon_aoi.shp`, search STAC for Sentinel-2 and Sentinel-1
items that intersect the AOI and write a meta JSON usable by
`scripts/02_compute_indices_and_terrain.py`.

Optional `--run` will invoke `02_compute_indices_and_terrain.py` with the
generated meta file.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box, mapping
from pystac_client import Client
import subprocess
import sys


STAC_API = "https://earth-search.aws.element84.com/v1"


def load_aoi_shapefile(path: str = "data/AOI/bongabon_aoi.shp"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {p}")
    gdf = gpd.read_file(str(p)).to_crs(4326)
    print(f"✅ AOI shapefile loaded: {p}")
    print(f"Bounds: {gdf.total_bounds}")
    return gdf


def get_tiles(aoi_gdf, resolution=1.0):
    bounds = aoi_gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, x + resolution, y + resolution)
            if aoi_gdf.intersects(tile).any():
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
        query={"eo:cloud_cover": {"lt": cloud_cover}},
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
            "sar:polarizations": {"intersects": ["VV", "VH"]},
        },
    )
    items = list(search.items())
    print(f"⬛ Sentinel-1 found: {len(items)} items")
    return items


def write_meta(aoi_gdf, s2_items, s1_items, start, end, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "province": "AOI",
        "start": start,
        "end": end,
        "geometry": mapping(aoi_gdf.unary_union),
        "sentinel2_items": [it.to_dict() for it in s2_items],
        "sentinel1_items": [it.to_dict() for it in s1_items],
    }
    out_path = out_dir / f"meta_AOI_{start}_{end}.json"
    with open(out_path, "w") as fh:
        json.dump(meta, fh)
    print(f"✅ Wrote meta: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="STAC download automation (AOI only)")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--cloud", type=int, default=20)
    parser.add_argument("--run", action="store_true", help="Run scripts/02_compute_indices_and_terrain.py after creating meta")
    args = parser.parse_args()

    aoi_gdf = load_aoi_shapefile()
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

    def print_acquisition_dates(items, label):
        print(f"\n{label} acquisition dates:")
        for item in items:
            date = getattr(item, "datetime", None)
            if date:
                print(date)

    print_acquisition_dates(s2_list, "Sentinel-2")
    print_acquisition_dates(s1_list, "Sentinel-1")

    meta_out = write_meta(aoi_gdf, s2_list, s1_list, args.start, args.end, Path("data/raw"))

    if args.run:
        cmd = [sys.executable, "scripts/02_compute_indices_and_terrain.py", "--meta", str(meta_out), "--outdir", "data/processed"]
        print("▶ Running 02_compute_indices_and_terrain.py...")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
