from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

img_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\EXIF_Test'
img1 = img_path + '//4-3-2021_14_37_20_19_23.jpg'
img2 = img_path + '//20201122_233546.jpg'

def get_exif(filename):
    exif = Image.open(filename)._getexif()
    print(exif)

    if exif is not None:
        for key, value in exif.items():
            name = TAGS.get(key, key)
            exif[name] = exif.pop(key)

        if 'GPSInfo' in exif:
            for key in exif['GPSInfo'].keys():
                name = GPSTAGS.get(key,key)
                exif['GPSInfo'][name] = exif['GPSInfo'].pop(key)
    else:
        print('well shit')

    return exif

exif = get_exif(img2)