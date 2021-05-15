import numpy as np
from skimage import measure
from PIL import Image
import cv2
import os

def get_folder_path():
    # image_path = r"C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\bulb_test_3-30-2020_offset"
    # image_path = r"C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\bulb_tests_3-30-2020_direct"
    # image_path = r"C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\bulb_gray_4-1-2020"
    folder_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\ir_outdoor\arlington_streetlight'
    # folder_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\monochrome\monochrome_camera\keb\gain5x'
    return folder_path

def get_image():
    image_path = get_folder_path()
    image_name = input("Image name: ")
    image_path += "\\" + image_name + ".jpg"
    image = cv2.imread(image_path)
    pix_crop = 40
    image = image[pix_crop:image.shape[0]-pix_crop, 0:image.shape[1]]
    return image



def map(value, from_max, from_min, to_max, to_min):
    result = (value-from_min)/(from_max-from_min)*(to_max-to_min)+to_min
    return result

#still need to do ratio between image with higher degree
def normalize(grayscaled, max_orig, min_orig, max_new, min_new):
    #"i" is row, "j" is column
    max = 0
    for i in range(grayscaled.shape[0]):
        for j in range(grayscaled.shape[1]):
            celsius = map(grayscaled[i][j], 255, 0, max_orig, min_orig)
            # if celsius == max_orig:
            #     print(celsius)
            normalized = (celsius-min_orig)/(max_new-min_new)
            grayscaled[i][j] = normalized*(255)
            if grayscaled[i][j] > max:
                max = grayscaled[i][j]
    print(max)
    return grayscaled

def scale(grayscaled, max_lower, max_higher):
    #"i" is row, "j" is column
    for i in range(grayscaled.shape[0]):
        for j in range(grayscaled.shape[1]):
            grayscaled[i][j] = int(grayscaled[i][j]*max_lower/max_higher)
    return grayscaled

def compare_images():

    max_1 = 40.9
    min_1 = 16.5
    max_2 = 40.8
    min_2 = 17.1

    image_1 = get_image()
    grayed_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    grayed_1_copy = np.copy(grayed_1)
    norm_1 = normalize(grayed_1_copy, max_1, min_1, max_2, min_2)
    # scale_1 = scale(grayed_1_copy, max_1, max_2)
    # changed = np.subtract(grayed_1,norm_1)


    image_2 = get_image()
    grayed_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    grayed_2_copy = np.copy(grayed_2)
    norm_2 = normalize(grayed_2_copy, max_2, min_2, max_2, min_2)

    os.chdir(get_folder_path())

    # horiz_grayscaled = np.concatenate((grayed_1, grayed_2), axis = 0)
    # horiz_norm = np.concatenate((image_1, image_2), axis = 0)
    vert_result_1 = np.concatenate((grayed_1, norm_1), axis = 0)
    vert_result_2 = np.concatenate((grayed_2, norm_2), axis = 0)

    # print("Height: " + str(horiz_raw.shape[0]) + "; Width: " + str(horiz_raw.shape[1]))
    # print("Height: " + str(horiz_grayscaled.shape[0]) + "; Width: " + str(horiz_grayscaled.shape[1]))

    cv2.imshow("image_1", vert_result_1)
    cv2.imshow("image_2", vert_result_2)
    # cv2.imshow("changes?", changed)
    cv2.waitKey()

def test_normalization(grayscaled):
    # ratio = max_image_1/max_image_2
    max = 127
    max_coord = [0,0]
    min = 127
    min_coord = [0,0]
    #"i" is row, "j" is column
    for i in range(grayscaled.shape[0]):
        for j in range(grayscaled.shape[1]):
            # grayscaled[i][j] = int(ratio*grayscaled[i][j])
            if grayscaled[i][j] > max:
                max_coord = [i,j]
                max = grayscaled[i][j]
            elif grayscaled[i][j] < min:
                min = grayscaled[i][j]
                max_coord = [i,j]
    print("Max: " + str(max))
    print("Min: " + str(min))
    print("Max Coord: " + str(max_coord[0]) + "," + str(max_coord[1]))
    print("Min Coord: " + str(min_coord[0]) + "," + str(min_coord[1]))

def outline_pixels(grayscaled, min_thresh, max_thresh, blob_locations):

    pix_arr = np.zeros([grayscaled.shape[0], grayscaled.shape[1]], dtype = int)
    outlined = np.copy(grayscaled)
    outlined = cv2.cvtColor(outlined, cv2.COLOR_GRAY2RGB)

    num_rows = 0
    num_cols = 0
    #outline images normally
    if len(blob_locations) == 0:
        num_rows = grayscaled.shape[0]
        num_cols = grayscaled.shape[1]
        blob_locations.append([0, num_rows, 0, num_cols])

    for locations in blob_locations:
        if locations[0] > 10:
            locations[0] -= 10
        if locations[1] < grayscaled.shape[0]-10:
            locations[1] += 10
        if locations[2] > 10:
            locations[2] -= 10
        if locations[3] < grayscaled.shape[1]-10:
            locations[3] += 10
        print(locations)
        for i in range(locations[0], locations[1]): #gets rows
            for j in range(locations[2], locations[3]): #gets columns
                #checking if pixels are within the threshold
                if grayscaled[i][j] >= min_thresh and grayscaled[i][j] <= max_thresh:
                    pix_arr[i][j] = 255
            # loc_arr = np.sort(np.where(pix_arr))
            # blob_locations.append([loc_arr[0][0], loc_arr[0][len(loc_arr[0])-1], loc_arr[1][0], loc_arr[1][len(loc_arr[1])-1]])
            
    for row in range(1, pix_arr.shape[0]-1):
        for col in range(1, pix_arr.shape[1]-1):
    # for row in range(locations[0], locations[1]):
    #     for col in range(locations[2], locations[3]):
            if pix_arr[row][col] == 255:
                # orig_value = outlined[row][col]
                for i in range(row-1, row+2):
                    for j in range(col-1, col+2):
                        if pix_arr[i][j] != 255:
                            outlined[i][j] = [255, 0, 0]
                # outlined[row][col] = orig_value
                outlined[row][col] = [0, 255, 0]
    
    return outlined

#consider whether to use or not
def save_image(image_name, image):
    # directory = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\ir_outdoor\arlington_streetlight\outlined'
    directory = r'C:\Users\alexy\OneDrive\Documents\STIMA\Thesis 2020'
    os.chdir(directory)
    cv2.imwrite(image_name, image)

def show_outlined(grayscaled, blob_locations):
    step = 15

    for x in range(240, 255, step):
        directory = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\ir_outdoor\arlington_streetlight\outlined'
        os.chdir(directory)

        image_name = "pixels " + str(x) + "-" + str(x+step)
        outlined_image = outline_pixels(grayscaled, x, x+step, blob_locations)

        cv2.imshow(image_name, outlined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if len(blob_locations) == 0:
            cv2.imwrite("unfiltered.jpg", outlined_image)
        else:
            cv2.imwrite("filtered.jpg", outlined_image)

def ir_blob_detector(image, min_thresh, max_thresh):
    image_copy = np.copy(image)
    grayscaled = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # for i in range(grayscaled.shape[0]): #gets rows
    #     for j in range(grayscaled.shape[1]): #gets columns
    #         if grayscaled[i][j] >= min_thresh and grayscaled[i][j] <= max_thresh:
    #             grayscaled[i][j] = 255
    #         else:
    #             grayscaled[i][j] = 0
    x = "5"
    # save_image('normal.png', grayscaled)
    blurred = cv2.GaussianBlur(grayscaled, (int(x), int(x)), 0)
    # save_image('blurred.png', blurred)
    #center blurred enough such that center cross no longer in threshold range
    thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]

    #opening
    # thresh = cv2.dilate(thresh, None, iterations=4)
    # save_image('dilation_1.png', thresh)
    # thresh = cv2.erode(thresh, None, iterations=2)
    # save_image('erosion_1.png', thresh)



    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype = "uint8")

    blob_locations = []

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
    
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")

        #if labels == label, labelMask[label] = 255
        labelMask[labels == label] = 255

        image = Image.fromarray(labelMask)
        save_image("blob" + str(label) + ".png", labelMask)

        numPixels = cv2.countNonZero(labelMask)
    
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 50 and numPixels < 500:
            print(numPixels)
            pix_loc = np.where(labelMask)
            print(pix_loc[0][len(pix_loc[0])-1])
            #add comma separated string; format: row_min, row_max, col_min, col_max
            # blob_boundary = str(pix_loc[0][0]) + "," + str(pix_loc[0][len(pix_loc[0])-1]) + "," + str(pix_loc[1][0]) + "," + str(pix_loc[1][len(pix_loc[1])-1])
            blob_boundary = [pix_loc[0][0], pix_loc[0][len(pix_loc[0])-1], pix_loc[1][0], pix_loc[1][len(pix_loc[1])-1]]
            blob_locations.append(blob_boundary)
            # mask = cv2.add(mask, labelMask)

    # image = Image.fromarray(mask)
    # image.show()

    # while(1):
    #     blurred = cv2.GaussianBlur(grayscaled, (int(x), int(x)), 0)
    #     thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]

    #     cv2.imshow("white", thresh)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     x = input()
    #     if x == "N":
    #         exit()
    # print(blob_locations)
    return blob_locations

def draw_circles():
    radius = 100
    line_thickness = 10
    left = get_image()
    right = get_image()
    l_pt_1 = (201, 717) #blue
    l_pt_2 = (2349, 900) #green
    r_pt_1 = (426, 1026) #green
    r_pt_2 = (2004, 846) #red
    cv2.circle(left,l_pt_1,radius,(255,100,20),line_thickness)
    cv2.circle(left,l_pt_2,radius,(100,255,20),line_thickness)
    cv2.circle(right,r_pt_1,radius,(100,255,20),line_thickness)
    cv2.circle(right,r_pt_2,radius,(100,20,255),line_thickness)
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    save_image('left.jpg', left)
    save_image('right.jpg', right)

if __name__ == "__main__":
    image = get_image()
    blob_locations = ir_blob_detector(image, 225, 255)
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_outlined(grayscaled, blob_locations)

    # draw_circles()

    # cv2.imshow("image", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # compare_images()
    # image = get_image()
    # ir_blob_detector(image)