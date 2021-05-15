from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from GPSPhoto import gpsphoto
import exifread
import numpy as np

img_path_1 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\EXIF_Test'
img1_path = img_path_1 + '//4-3-2021_14_37_20_19_23.jpg'
img2_path = img_path_1 + '//20201122_233546.jpg'

img_path_2 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\Stereo_Test\trash_recycle'
img3_path = img_path_2 + '//left.jpg'
img4_path = img_path_2 + '//right.jpg'

def return_img(img_path):
    return np.array(Image.open(img_path), dtype = 'float')/255.

def get_exif_1(filename):
    exif = Image.open(filename)._getexif()
    print(exif.items())

    if exif is not None:
        for key, value in exif.items():
            print(key)
            name = TAGS.get(key, key)
            exif[name] = exif.pop(key)

        if 'GPSInfo' in exif:
            for key in exif['GPSInfo'].keys():
                name = GPSTAGS.get(key,key)
                exif['GPSInfo'][name] = exif['GPSInfo'].pop(key)
    else:
        print('well shit')

    return exif

def get_exif_2(filename):
    tags = exifread.process_file(open(filename, 'rb'))
    geo = {i:tags[i] for i in tags.keys() if i.startswith('GPS')}
    print(geo)

# exif = get_exif(img2)

if __name__ == '__main__':
    # data = gpsphoto.getGPSData(img3)
    # print(data['Latitude'], data['Longitude'])

    # get_exif_2(img3)

    img3 = return_img(img3_path)
    img4 = return_img(img4_path)
