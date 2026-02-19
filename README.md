# Digital Agri Phase 1: Onion Detection Pipeline

This repo contains a modular, end-to-end Python pipeline that:

1. Downloads Sentinel-2 (optical) and Sentinel-1 (VV/VH) tiles via STAC;
2. Computes optical indices (NDVI, NDWI, NDSI, EVI, NDRE, SAVI, GNDVI, ClI), and a B7_B8_B8A average band;
3. Downloads/loads DEM (SRTM) and derives Slope, Roughness, and TPI;
4. Downloads road network from OpenStreetMap and rasterizes a road mask;
5. Applies the threshold rules to produce a final binary GeoTIFF (1 = onion, 0 = not onion);
6. Exports visualization and final TIFF.

See `scripts/` for the Python code and `requirements.txt` for dependencies.

## Prerequisites
Source codes were written using Python 3.9 and use the following libraries and files:
- pystac-client
- stackstac
- rioxarray
- rasterio
- numpy
- pandas
- geopandas
- shapely
- matplotlib
- osmnx
- scipy
- click
- tqdm
- xarray
- affine
- requests

## Methodology

This methodolgy is coming from 


## Scope and Limitations

The scope of the code is limited yet within the bounds of the pilot site which are Bongabon, Nueva Ecija and a few of its neighboring municipalities. Replication of the code in other provinces is not yet available due to lack of ground validation points collected. 

Furthermore, this rapid mapping of onion extents is precisely effective in detecting matured onions within its croping cycle. Onions during vegetative stages will most likely not be detected by the algorithm. 

## About the Author

The creation of this Onion Extent Detection algorithm is created by the Space Data Mobilization and Applications Division of the Philippine Space Agency. 

For inquiries, kindly send an email at sdmad@philsa.gov.ph. 
