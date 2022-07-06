import argparse
from numpy import save
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, box
import os
import time
import sys
import random
import glob

os_sep = os.sep
cwd = list(os.getcwd().split(os_sep))
STIMA_scripts_dir = os_sep.join(cwd[:len(cwd)-1])
STIMA_dir = os_sep.join(cwd[:len(cwd)-3])
BRF_Analysis_dir = os.path.join(STIMA_scripts_dir, 'BRF_Analysis')

sys.path.insert(1, BRF_Analysis_dir)
import brf_database_analysis_v2

'''
DESCRIPTION: generates random points in a polygon

INPUT:  number              number of random points for a polygon
        polygon
OUTPUT: list_of_points      list of random points in lat long format(?)
'''
def generate_random_spatialpoints(number, polygon):
    #Source: https://gis.stackexchange.com/questions/207731/generating-random-coordinates-in-multipolygon-in-python
    list_of_points = []
    minx, miny, maxx, maxy = polygon.bounds
    counter = 0
    while counter < number:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        pnt = Point(x,y)
        if polygon.contains(pnt):
            list_of_points.append((x,y))
            counter += 1
    return list_of_points

'''
DESCRIPTION:    creates a new geo dataframe with random points within new polygons with associated BRFs
                saves data frames in new folder

INPUT:  geojson_path    path to geojson file
        boundary_path   path to lat-long box boundaries to narrow down df

OUTPUT: nothing; files are in `folder_name`
'''
def save_new_dfs(folder_name, save_path, geojson_path, boundary_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # extracting new scenarios; make multiple gdfs,
    # put into new files; make option to process again or just use from saved file
    boundary_df = pd.read_csv(boundary_path)
    xmin_list = boundary_df.xmin.tolist()
    ymin_list = boundary_df.ymin.tolist()
    xmax_list = boundary_df.xmax.tolist()
    ymax_list = boundary_df.ymax.tolist()

    # assertions
    assert len(xmin_list) == len(ymin_list)
    assert len(ymin_list) == len(xmax_list)
    assert len(xmax_list) == len(ymax_list)

    start_time = time.time()
    gdf = gpd.read_file(geojson_path)
    # print(gdf.crs)
    print(f'Time elapsed for loading file (seconds): {time.time() - start_time}')
    print()

    for i in range(len(xmin_list)):
        # filtering out 
        boundary_box = box(xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i])
        bounded_gdf = gdf.intersection(boundary_box)
        bounded_gdf = gpd.GeoDataFrame(bounded_gdf[~bounded_gdf.is_empty])
        bounded_gdf = bounded_gdf.rename(columns={0:'geometry'}).set_geometry('geometry')

        # generate save path
        layout_name_path = os.path.join(save_path, folder_name + '_layout_' + str(i))
        bounded_gdf.to_file(layout_name_path)

'''
DESCRIPTION: loads GeoDataFrame from .shp file from save path

INPUT:      

OUTPUT:
'''

def load_gdf(save_path):
    pass

'''
DESCRIPTION: takes in a GeoDataFrame and computes the random points of buffered polygons

INPUT:      gdf     GeoDataFrame

OUTPUT:     gdf     GeoDataFrame with the following columns:
                    geometry | buffer_geometry | random_points
'''

def compute_random_points(gdf):
    gdf = gdf.to_crs(epsg=3857)
    gdf['buffer_geometry'] = gdf.buffer(20)
    gdf['geometry'] = gdf['geometry'].to_crs(crs=4326)
    gdf['buffer_geometry'] = gdf['buffer_geometry'].to_crs(crs=4326)
    gdf['random_points'] = gdf['buffer_geometry'].apply(lambda row: generate_random_spatialpoints(1, row))
    # gdf.plot()
    # plt.show()

    # print(gdf['geometry'])
    # print(gdf['buffer_geometry'])
    # print(gdf['random_points'])

    # gdf.geometry.plot()
    # gdf.buffer_geometry.plot()


    '''
    TODO:   shapely --> make smaller geojson file to load (and make an option for that)
            print out shapes
            compute centroid of polygons of geometry column (check geopandas df column names)
            (Zeal said something about using `gdf.geometry.centroid` with `.apply`)
            create buffer points from polygon (20m; figure that out)
            geometry columns should be:
                building polygon | centroid of building | then buffer around centroid
            then generate random points inside buffer; result should be:
                building polygon | centroid of building | then buffer around centroid | random point
            need to attach brf label to random point, so create another column with that
            (I think that should be it...?)
            figure out how to plot original buildings + buffer + random points to visualize stuff
    '''

def main(args):
    geojson_path = os.path.join(STIMA_scripts_dir, 'Class_Sim', 'geojson_files', args.geojson_path)
    boundary_path = os.path.join(STIMA_scripts_dir, 'Class_Sim', 'geojson_files', args.boundary_path)

    geojson_temp = list(geojson_path.split(os.sep))
    folder_name = geojson_temp[len(geojson_temp) - 1].split('.')[0]
    save_path = os.path.join(os.getcwd(), folder_name)

    print('Skip file generation? [y]/other')
    user_input = input()
    if user_input != 'y':
        save_new_dfs(folder_name, save_path, geojson_path, boundary_path)
    
    print(glob.glob(save_path + '/*/', recursive=True))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gp', '--geojson_path', required=True, type=str, help='geojson file name; Example: Alaska.geojson')
    parser.add_argument('-bp', '--boundary_path', required=True, type=str, help='boundary csv file name; Example: Alaska_boundaries.csv')
    args = parser.parse_args()
    main(args)