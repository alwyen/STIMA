import cv2
import math
import numpy as np
import mahotas
import imutils
from scipy.spatial import distance as dist
from itertools import combinations

img1_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\Shape Matching Image Tests\gain17.5x_exp16.67ms_left.jpg'
img2_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\Shape Matching Image Tests\gain17.5x_exp16.67ms_right.jpg'

# img1_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\zernike_test\compare_gray_1.jpg'
# img2_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\zernike_test\compare_gray_2.jpg'
# img3_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\zernike_test\compare_gray_test_img.jpg'
# img4_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\zernike_test\compare_gray_3.jpg'
# img5_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\zernike_test\compare_gray_4.jpg'
# img6_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\zernike_test\compare_gray_5.jpg'
# img7_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\zernike_test\compare_gray_6.jpg'
width = 640
height = 360
shape_weight = 30000
threshold = 230
line_thickness = 1

class ZernikeMoments:
    def __init__(self, radius, degree):
        self.radius = radius
        self.degree = degree
    def describe(self, image):
        return mahotas.features.zernike_moments(image, self.radius, degree = self.degree)

def point_dist(pt_1, pt_2):
    return math.sqrt((pt_1[0] - pt_2[0])**2 + (pt_1[1] - pt_2[1])**2)

def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_from_path(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, height))
    return img

def save_image(name, img):
    cv2.imwrite(name, img)

def extract_images_from_video(video_path):
    rate = 0.375 #seconds
    # rate = 1.5 #seconds
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    new_framerate = round(fps*rate)
    # total_frames = round(video.get(cv2.CAP_PROP_FRAME_COUNT))

    success = 1
    count = 0
    frame_list = []

    while success:
        success, frame = video.read()
        if count%new_framerate == 0:
            frame_list.append(frame)
            orb_feature_detection(frame)
            # cv2.imshow('frame#: ' + str(count), frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            count += 1

        count += 1

    return frame_list

def process_image(img):
    img[img < threshold] = 0
    img[img > threshold] = 255
    # show_image(img)
    img = cv2.erode(img, None, iterations = 1)
    # show_image(img)
    img = cv2.dilate(img, None, iterations = 2)
    # show_image(img)
    return img

def filter_shape_only(img):
    img[img > threshold] = 0
    img[img > 20] = 255
    return img

def draw_centers(center_list, img):
    img_copy = np.copy(img)
    for center in center_list:
        cv2.circle(img_copy, center, 2, 0, 1) #radius, color, thickness
    cv2.imshow('blob centers', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_circle(img, center_pt):
    cv2.circle(img, center_pt, 1, 255, 1)

#type: 'previous' or 'current' image analyzed
def centroid(img, img_type):
    center_list = []

    if img_type == 'previous':
        img[0:height, 0:round(width/3)] = 0
    elif img_type == 'current':
        img[0:height, round(width*2/3):width] = 0

    params = cv2.SimpleBlobDetector_Params()

    #area filter for blob
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 500

    # params.filterByCircularity = True
    # params.minCircularity = 0.8

    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)
    centers = detector.detect(img)

    for center in centers:
        center_list.append((round(center.pt[0]), round(center.pt[1]))) #pt[0] --> x, pt[1] --> y

    # draw_centers(center_list, img)

    return center_list

#shape[0] => row, shape[1] => col
def center_of_mass(img_shape):
    sum_row = 0
    sum_col = 0
    sum_row_col = 0
    for row in range(0, img_shape.shape[0]):
        for col in range(0, img_shape.shape[1]):
            sum_row_col += img_shape[row][col]
            sum_row += row*img_shape[row][col]
            sum_col += col*img_shape[row][col]
    c_row = int(round(sum_row/sum_row_col))
    c_col = int(round(sum_col/sum_row_col))
    return (c_row, c_col)

def outline_shape(img_shape):
    outline = np.zeros(img_shape.shape, dtype = 'uint8')
    cnts = cv2.findContours(img_shape.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    cv2.drawContours(outline, [cnts], -1, 255, -1)
    return outline

def show_centers(img, center_list):
    for pt in center_list:
        draw_circle(img, pt)

'''
the returned list should contain lists with shapes drawn from 3 nodes, 4 nodes, ..., max_number_nodes
in each node list (e.g. 3 nodes used to draw shapes), there should be another list of tuples
each tuple contains the shape image [0] and the original image annotated with the shape [1]
'''
def draw_shapes(orig_image_copy, bw_img, center_list, max_number_nodes):
    if max_number_nodes < 4:
        print('Error: less than 4 blobs detected')
    node_category_list = []
    for num_nodes in range(4, max_number_nodes+1):
    # for num_nodes in range(4, 7):
        print('Node Quantity: ' + str(num_nodes))
        combination_list = list(combinations(center_list, num_nodes)) #create tubles of all possible combinations for n number of nodes; e.g. each tuple has 3 points
        shape_from_num_nodes_list = [] #list for a given number of nodes used to draw shapes
        for combination in combination_list: #go through each tuple and draw the shapes; add those shapes to a list
            # shape = np.zeros((height, width, 1), dtype = np.uint8) #height, width, number of channels
            max_dist = -1 #initialization
            min_dist = -1
            min_dist_pair = None
            copy_img = np.copy(orig_image_copy)
            shape = np.copy(bw_img)
            point_combination = list(combinations(combination, 2)) #for each quantity of nodes (e.g. 3 nodes), make pairs (e.g. [a,b], [a,c], [b,c])
            for pair in point_combination:
                cv2.line(shape, pair[0], pair[1], 100, line_thickness) #drawing line for shape image
                cv2.line(copy_img, pair[0], pair[1], 255, line_thickness) #annotating original image with white lines
                distance_bt_points = point_dist(pair[0], pair[1])
                if max_dist == -1:
                    max_dist = distance_bt_points
                elif max_dist < distance_bt_points:
                    max_dist = distance_bt_points
            shape[shape > 100] = 0
            shape[shape == 100] = 255
            shape_from_num_nodes_list.append((shape, copy_img, max_dist)) #for this set of number of nodes, add shapes to master shape list
        node_category_list.append(shape_from_num_nodes_list)
    return node_category_list

# take in two lists for comparisons
#note: each list should be the same length
#[0] is shape only, [1] is shape drawn on image
def shape_comparer(shape_from_nodes_list_1, shape_from_nodes_list_2, degree):
    shape_score = 0
    print('Degree:' + str(degree))
    for x in range(len(shape_from_nodes_list_1)): #iterate through the number of nodes used to draw shapes (e.g. 3, 4, 5, ..., max)
        for shape_list_1 in shape_from_nodes_list_1[x]:
            for shape_list_2 in shape_from_nodes_list_2[x]:
                shape_list_1_moment = ZernikeMoments(round(shape_list_1[2]/2), degree)
                shape_list_2_moment = ZernikeMoments(round(shape_list_2[2]/2), degree)
                
                shape_outline_1 = outline_shape(shape_list_1[0])
                shape_outline_2 = outline_shape(shape_list_2[0])
                area_shape_1 = np.sum(shape_outline_1)/255
                area_shape_2 = np.sum(shape_outline_2)/255
                
                #arbitrary value listed here
                area_score = abs(area_shape_1 - area_shape_2)/shape_weight

                desc1 = shape_list_1_moment.describe(shape_list_1[0]) #describe shape-only image
                desc2 = shape_list_2_moment.describe(shape_list_2[0])
                # desc1 = shape_list_1_moment.describe(shape_outline_1)
                # desc2 = shape_list_2_moment.describe(shape_outline_2)
                euclid_dist = dist.euclidean(desc1, desc2)
                # print(euclid_dist)

                shape_score = area_score + euclid_dist
                if shape_score < 0.1:
                    print('Euclidean Dist: ' + str(euclid_dist))
                    cv2.imshow('img1', shape_list_1[1])
                    cv2.imshow('img2', shape_list_2[1])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # cv2.imshow('img1', shape_list_1[0])
                # cv2.imshow('img2', shape_list_2[0])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

'''
get x and y points of all nodes
every combination of shapes based on 3 nodes, 4, 5, ... , max nodes, but only for the max number of one image
store shapes in a list
    for each shape, make a new numpy array and draw lines on that
    also draw lines on original image; store each shape/original image with shape as a tuple
brute for matching for each shape in a node category (via euclidean distance)
    for each result, show original images matched with shapes; unsure if want to save numbers for statistics
'''

#combine all methods into this one
def geometry_matching(path1, path2):
    img1 = img_from_path(path1)
    img2 = img_from_path(path2)

    img1_copy = np.copy(img1)
    img2_copy = np.copy(img2)

    processed_1 = process_image(img1)
    processed_2 = process_image(img2)

    img1_blob_centers = centroid(processed_1, 'previous')
    img2_blob_centers = centroid(processed_2, 'current')

    # show_centers(img1_copy, img1_blob_centers)
    # show_centers(img2_copy, img2_blob_centers)

    max_number_nodes = min(len(img1_blob_centers), len(img2_blob_centers))

    shapes_from_node_list_1 = draw_shapes(img1_copy, processed_1, img1_blob_centers, max_number_nodes) 
    shapes_from_node_list_2 = draw_shapes(img2_copy, processed_2, img2_blob_centers, max_number_nodes)    

    # for degree in range(9, 13):
    #     shape_comparer(shapes_from_node_list_1, shapes_from_node_list_2, degree)
    #     print('Done with degree = ' + str(degree))
    shape_comparer(shapes_from_node_list_1, shapes_from_node_list_2, 8)

def zernike_moments_processing():
    pass

def euclid_dst_zernike():
    pass

if __name__ == '__main__':
    geometry_matching(img1_path, img2_path)

    # img1 = img_from_path(img1_path)
    # img2 = img_from_path(img2_path)
    # img1 = process_image(img1)
    # img2 = process_image(img2)
    # center_list_1 = centroid(img1)
    # center_list_2 = centroid(img2)
    # for center in center_list_1:
    #     cv2.circle(img1, center, 1, 0, 1)

    # for center in center_list_2:
    #     cv2.circle(img2, center, 1, 0, 1)

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # draw_shapes(center_list_1, 4)
    # extract_keypoint_locations(center_list_1)
    # geometry_matching(img1, img2)