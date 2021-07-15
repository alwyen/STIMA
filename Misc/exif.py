
import exifread

img_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\STIMA\Misc\20210712_152846.jpg'

# Open image file for reading (binary mode)
f = open(img_path, 'rb')

# Return Exif tags
tags = exifread.process_file(f)
print(tags)