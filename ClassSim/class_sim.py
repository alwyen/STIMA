import argparse
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, box
import os
import time

os_sep = os.sep
cwd = list(os.getcwd().split(os_sep))
STIMA_scripts_dir = os_sep.join(cwd[:len(cwd)-1])
STIMA_dir = os_sep.join(cwd[:len(cwd)-3])

def return_areas(geojson_path, boundary_path):
    # extracting new scenarios
    boundary_df = pd.read_csv(boundary_path)
    xmin_list = boundary_df.xmin.tolist()
    ymin_list = boundary_df.ymin.tolist()
    xmax_list = boundary_df.xmax.tolist()
    ymax_list = boundary_df.ymax.tolist()

    # assertions
    assert len(xmin_list) == len(ymin_list)
    assert len(ymin_list) == len(xmax_list)
    assert len(xmax_list) == len(ymax_list)

    print(xmin_list)
    print(ymin_list)
    print(xmax_list)
    print(ymax_list)

    # start_time = time.time()
    # gdf = gpd.read_file(geojson_path)
    # print(f'Time elapsed for loading file (seconds): {time.time() - start_time}')

def main(args):
    geojson_path = os.path.join(STIMA_scripts_dir, 'ClassSim', 'geojson_files', args.geojson_path)
    boundary_path = os.path.join(STIMA_scripts_dir, 'ClassSim', 'geojson_files', args.boundary_path)
    return_areas(geojson_path, boundary_path)
    # print("hello")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gp', '--geojson_path', required=True, type=str, help='geojson file name')
    parser.add_argument('-bp', '--boundary_path', required=True, type=str, help='boundary csv file name')
    args = parser.parse_args()
    main(args)