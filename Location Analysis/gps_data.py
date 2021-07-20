import PIL.Image
import PIL.ExifTags
import glob
import os
import pandas as pd

'''
TODO:
    use glob to access all the images
        how will I distinguish between images well enough?
        per each location, 3 image sets; folders of image sets?

    create a dataframe of all the data
    export to CSV file
    dataframe order: [image_name | latitude | longitude | height]
'''

def get_GPS_data(image_name):
    image = PIL.Image.open(image_name)

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in image._getexif().items()
        if k in PIL.ExifTags.TAGS
    }

    keys_list = list(exif.keys())
    values_list = list(exif.values())

    GPSInfo_index = keys_list.index('GPSInfo')

    GPSInfo_list = values_list[GPSInfo_index]

    north_south = GPSInfo_list[1]   #south => negative
    east_west = GPSInfo_list[3]     #west => negative

    lat_degree = float(GPSInfo_list[2][0])
    lat_min = float(GPSInfo_list[2][1])
    lat_sec = float(GPSInfo_list[2][2])

    long_degree = float(GPSInfo_list[4][0])
    long_min = float(GPSInfo_list[4][1])
    long_sec = float(GPSInfo_list[4][2])

    orthometric_height = GPSInfo_list[6]

    latitude = lat_degree + lat_min / 60 + lat_sec / 3600
    longitude = long_degree + long_min / 60 + long_sec / 3600

    if north_south == 'S':
        latitude = -latitude

    if east_west == 'W':
        longitude = -longitude
    
    # print(GPSInfo_list[2])
    # print(latitude)

    # print(GPSInfo_list[4])
    # print(longitude)

    # print()

    return latitude, longitude, orthometric_height

def extract_gps_data(image_folder_path):
    name_list = list()
    latitude_list = list()
    longitude_list = list()
    height_list = list()

    os.chdir(image_folder_path)
    image_list = glob.glob('*.jpg')
    for image_name in image_list:
        latitude, longitude, orthometric_height = get_GPS_data(image_name)
        name_list.append(image_name)
        latitude_list.append(latitude)
        longitude_list.append(longitude)
        height_list.append(orthometric_height)

    d = {'Image_Name': name_list, 'Latitude': latitude_list, 'Longitude': longitude_list, 'Orthometric_Height':height_list}
    df = pd.DataFrame(data = d)

    df.to_csv('gps_metadata.csv')

def rename_images(folder_path, base_name):
    os.chdir(folder_path)
    image_list = glob.glob('*.jpg')
    for i in range(len(image_list)):
            # os.rename(image_list[i], base_name + image_list[i])
            print(base_name + image_list[i])

if __name__ == '__main__':
    image_gps_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\image_gps'

    extract_gps_data(image_gps_path)