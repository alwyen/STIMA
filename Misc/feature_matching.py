import cv2
import numpy as np
import code
import os
from natsort import natsorted, ns
import matplotlib.pyplot as plt

#using this as a struct
class gain:
    def __init__(self, gain, exp_time, precision_list):
        self.gain = gain
        self.exp_time = exp_time
        self.precision_list = precision_list

#change save path
def save_image(name, img):
    # folder = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\monochrome\keypoint_tests'
    folder = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\monochrome\self_drawn\monochrome_processed_20_matches'
    os.chdir(folder)
    cv2.imwrite(name, img)

def find_points(width, kp1, kp2, matches, num_matches):
    l_pt = kp1[matches[0].queryIdx].pt
    r_pt = kp2[matches[0].trainIdx].pt
    # l_pt = (int(l_pt[0]),int(l_pt[1]))
    # r_pt = (int(r_pt[0])+width,int(r_pt[1]))

    print('find points: left')
    print('row: ' + str(l_pt[1]))
    print('col: ' + str(l_pt[0]))
    print()

    print('find points: right')
    print('row: ' + str(r_pt[1]))
    print('col: ' + str(r_pt[0]+width))
    print()


'''
need to also return a text file with the number of matches for that given set/precision for each set
ipmlement a method that takes in file and graphs shit
'''
#STILL NOT WORKING CORRECTLY
#ALL THE POINTS ON THE RIGHT ARE MESSED UP
def get_true_positives(width, height, img1, img2, kp1, kp2, matches, num_matches, reference_pts, algo_name):
    true_pos_counter = 0
    precision = 0
    row_thresh = 20 #value needs to be changed/eventually need to run statistics
    col_thresh = 100 #value needs to be changed/eventually need to run statistics
    radius = 3
    line_thickness = 1
    matches_drawn = np.concatenate((img1, img2), axis = 1)
    cv2.line(matches_drawn, (width, 0), (width, height), (255,255,255), line_thickness)
    matches_drawn = cv2.cvtColor(matches_drawn,cv2.COLOR_GRAY2RGB)

    '''
    note:
    [0][0] --> left image row
    [0][1] --> left image col
    [1][0] --> right image row
    [1][1] --> right image col
    '''

    #(col, row) ordering
    # print(num_matches)
    for x in range(num_matches):
        l_pt = kp1[matches[x].queryIdx].pt
        r_pt = kp2[matches[x].trainIdx].pt #issue with out of bounds with sky_pond
        l_pt = (int(l_pt[0]),int(l_pt[1]))
        r_pt = (int(r_pt[0])+width,int(r_pt[1]))

        dif_row = (reference_pts[0][0]-l_pt[1]) - (reference_pts[1][0]-r_pt[1])
        dif_col = (reference_pts[0][1]-l_pt[0]) - (reference_pts[1][1]-r_pt[0])

        # if dif_x < error_thresh and dif_x > -error_thresh and dif_y < error_thresh and dif_y > -error_thresh:
        if -row_thresh <= dif_row and dif_row <= row_thresh and -col_thresh <= dif_col and dif_col <= col_thresh:
            true_pos_counter += 1
            cv2.circle(matches_drawn,l_pt,radius,(0,255,0),line_thickness)
            cv2.circle(matches_drawn,r_pt,radius,(0,255,0),line_thickness)
            cv2.line(matches_drawn,l_pt,r_pt,(0,255,0),line_thickness)
        else:
            cv2.circle(matches_drawn,l_pt,radius,(0,0,255),line_thickness)
            cv2.circle(matches_drawn,r_pt,radius,(0,0,255),line_thickness)
            cv2.line(matches_drawn,l_pt,r_pt,(0,0,255),line_thickness)

    # cv2.imshow(algo_name, matches_drawn)
    precision = round(true_pos_counter/num_matches, 2)
    # precision = true_pos_counter/num_matches
    # print('Precision: ' + str(precision))
    numMatches_precision = [num_matches, precision]
    # return precision

    return matches_drawn, numMatches_precision

'''
CHANGE ALL KEYPOINT AND DESCRIPTOR NAMES TO JUST KP1/KP2 AND DES1/DES2
'''

def sift(img1, img2, width, height, num_matches, reference_pts):
    #SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    sift_kp1, sift_des1 = sift.detectAndCompute(img1, None)
    sift_kp2, sift_des2 = sift.detectAndCompute(img2, None)

    # Brute Force Matching (SIFT)
    sift_bf = cv2.BFMatcher(crossCheck=True)
    sift_matches = sift_bf.match(sift_des1, sift_des2)
    sift_matches = sorted(sift_matches, key = lambda x:x.distance)

    if num_matches > len(sift_matches):
        num_matches = len(sift_matches)

    sift_matching_result = cv2.drawMatches(img1, sift_kp1, img2, sift_kp2, sift_matches[:num_matches], None, flags=2)
    # print('Number of Sift Matches: ' + str(len(sift_matches)))

    # find_points(width, sift_kp1, sift_kp2, sift_matches, num_matches)

    # print("Sift")
    match_image, numMatches_precision = get_true_positives(width, height, img1, img2, sift_kp1, sift_kp2, sift_matches, num_matches, reference_pts, "SIFT")

    # save_image("sift_match.jpg", match_image)
    # save_image("sift_actual_match.jpg", sift_matching_result)

    return sift_matching_result, match_image, numMatches_precision

def surf(img1, img2, width, height, num_matches, reference_pts):
    #SURF detector
    surf = cv2.xfeatures2d.SURF_create()
    surf_kp1, surf_des1 = surf.detectAndCompute(img1, None)
    surf_kp2, surf_des2 = surf.detectAndCompute(img2, None)

    # Brute Force Matching (SURF)
    surf_bf = cv2.BFMatcher(crossCheck=True)
    surf_matches = surf_bf.match(surf_des1, surf_des2)
    surf_matches = sorted(surf_matches, key = lambda x:x.distance)

    if num_matches > len(surf_matches):
        num_matches = len(surf_matches)

    surf_matching_result = cv2.drawMatches(img1, surf_kp1, img2, surf_kp2, surf_matches[:num_matches], None, flags=2)
    # print('Number of Surf Matches: ' + str(len(surf_matches)))

    # print("Surf")
    match_image, numMatches_precision = get_true_positives(width, height, img1, img2, surf_kp1, surf_kp2, surf_matches, num_matches, reference_pts, "SURF")

    return surf_matching_result, match_image, numMatches_precision

def orb(img1, img2, width, height, num_matches, reference_pts):
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching (ORB)
    orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb_matches = orb_bf.match(des1, des2)
    orb_matches = sorted(orb_matches, key = lambda x:x.distance)

    if num_matches > len(orb_matches):
        num_matches = len(orb_matches)

    orb_matching_result = cv2.drawMatches(img1, kp1, img2, kp2, orb_matches[:num_matches], None, flags=2)
    # print('Number of Orb Matches: ' + str(len(orb_matches)))

    # print("Orb")
    match_image, numMatches_precision = get_true_positives(width, height, img1, img2, kp1, kp2, orb_matches, num_matches, reference_pts, "ORB")

    return orb_matching_result, match_image, numMatches_precision


def match_features(pic1_path, pic2_path, save_path, image1_name, image2_name, location):

    reference_pts = []
    #REFERENCE POINTS
    arlington_center = [[38,465],[29,862]]
    arlington_house = [[38,465],[29,862]]
    behind_keb = [[128,408],[158,768]]
    keb = [[56,567],[80,745]]
    sky_pond = [[116,311],[100,741]]

    if location == 'arlington_center':
        reference_pts = arlington_center
    elif location == 'arlington_house':
        reference_pts = arlington_house
    elif location == 'behind_keb':
        reference_pts = behind_keb
    elif location == 'keb':
        reference_pts = keb
    elif location == 'sky_pond':
        reference_pts = sky_pond

    # width = 500
    # height = 420

    # width = 640
    # height = 480

    num_matches = 20

    width = 640
    height = 360

    img1 = cv2.resize(cv2.imread(pic1_path, 0), (width, height))
    img2 = cv2.resize(cv2.imread(pic2_path, 0), (width, height))

    #[number of matches, precision]
    sift_matching_result, draw_tf_sift, sift_precision = sift(img1, img2, width, height, num_matches, reference_pts)
    surf_matching_result, draw_tf_surf, surf_precision = surf(img1, img2, width, height, num_matches, reference_pts)
    orb_matching_result, draw_tf_orb, orb_precision = orb(img1, img2, width, height, num_matches, reference_pts)
    precision_list = [sift_precision, surf_precision, orb_precision]

    # Image saving paths
    save_img1 = save_path + '/' + image1_name
    save_img2 = save_path + '/' + image2_name
    save_sift = save_path + '/' + 'SIFT.jpg'
    save_surf = save_path + '/' + 'SURF.jpg'
    save_orb = save_path + '/' + 'ORB.jpg'

    #Drawing line to distinguish two images
    lineThickness = 1
    cv2.line(sift_matching_result, (width, 0), (width, height), (255,255,255), lineThickness)
    cv2.line(surf_matching_result, (width, 0), (width, height), (255,255,255), lineThickness)
    cv2.line(orb_matching_result, (width, 0), (width, height), (255,255,255), lineThickness)

    #Showing images
    # cv2.imshow(image1_name, img1)
    # cv2.imshow(image2_name, img2)
    # # cv2.imshow("SIFT", sift_matching_result)
    # # cv2.imshow("SURF", surf_matching_result)
    # # cv2.imshow("ORB", orb_matching_result)

    # cv2.imshow("SIFT", draw_tf_sift)
    # cv2.imshow("SURF", draw_tf_surf)
    # cv2.imshow("ORB", draw_tf_orb)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # #Saving images
    # cv2.imwrite(save_img1, img1) #left image
    # cv2.imwrite(save_img2, img2) #right image

    # cv2.imwrite(save_sift, sift_matching_result)
    # cv2.imwrite(save_surf, surf_matching_result)
    # cv2.imwrite(save_orb, orb_matching_result)

    #draw_true_false
    # cv2.imwrite(save_sift, draw_tf_sift)
    # cv2.imwrite(save_surf, draw_tf_surf)
    # cv2.imwrite(save_orb, draw_tf_orb)

    #return list of precision
    return precision_list

def write_note(message):
    f = open("NOTE.txt", "w")
    f.write(message)
    f.close()

'''
in each list contains a gain object
each gain object contains two strings: the gain value and the exposure time value
each object contains a precision_list, which contains 3 elements: each element is a pair of the algorithm used for
that specific gain and exposure time
the element is formatted by [number of matches, precision]; precision is from 0-1
'''

def sort_list(gain_precision_list):
    gain_list = []
    exp_time_list = []
    precision_list = []

    for g in gain_precision_list:
        gain_list.append(g.gain)
        exp_time_list.append(g.exp_time)
        precision_list.append(g.precision_list)

    zipped = zip(gain_list, exp_time_list, precision_list)
    return natsorted(zipped) #sorted list

#graphs as a function of exposure time
def graph_fn_exp(gain_precision_list):
    save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Thesis 2020\Graphs'
    location_name = input('Type in a location name for the graphs: ')
    save_path += '\\' + location_name

    #make a new folder for location
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_path += '\\' + 'gain_graphs'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    os.chdir(save_path)

    width = 0.26

    gain_list = []
    exp_time_list = []
    precision_list = []
    
    sub_exp_time_list = [] #for looping through each value in a gain category
    
    sift_precision_list = []
    surf_precision_list = []
    orb_precision_list = []

    sift_num_match_list = []
    surf_num_match_list = []
    orb_num_match_list = []

    sorted_list = sort_list(gain_precision_list)
    for element in sorted_list:
        gain_list.append(element[0])
        exp_time_list.append(element[1].replace('ms', ''))
        precision_list.append(element[2])

    #gain_values contains just the four different values
    gain_values = natsorted(list(set(gain_list)))
    percent = []
    if len(gain_values) == 4:
        percent = ['25%', '50%', '75%', '100%']
    else:
        percent = ['0%', '25%', '50%', '75%', '100%']

    #make another list based on exposure times of gain values
    # for gain in gain_values:
    for i in range(len(gain_values)):
        for j in range(len(gain_list)):
            if gain_values[i] == gain_list[j]:
                sub_exp_time_list.append(exp_time_list[j])

                sift_num_match_list.append(precision_list[j][0][0])
                surf_num_match_list.append(precision_list[j][1][0])
                orb_num_match_list.append(precision_list[j][2][0])

                sift_precision_list.append(precision_list[j][0][1])
                surf_precision_list.append(precision_list[j][1][1])
                orb_precision_list.append(precision_list[j][2][1])

        #subplots(row, cols)
        # fig, ax = plt.subplots(2, 1)
        fig, ax = plt.subplots(1)
        fig.tight_layout(pad = 4.0)
        # fig.suptitle(location_name, fontsize = 18)
        fig.subplots_adjust(top = 0.85)

        # fig, precision_ax = plt.subplots()
        x = np.arange(len(sub_exp_time_list))
        ax.bar(x-width, sift_precision_list, width, label = 'Sift')
        ax.bar(x, surf_precision_list, width, label = 'Surf')
        ax.bar(x+width, orb_precision_list, width, label = 'Orb')
        ax.set_xticks(x)
        ax.set_xticklabels(sub_exp_time_list)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height*0.5])
        ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        ax.set_title("Precision Graph for " + percent[i] + " Gain (" + gain_values[i].replace("gain", "") + "/30x)")
        ax.set(xlabel = 'Exposure Time (ms)', ylabel = 'Precision')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0,1.1, step = 0.25))

        # x = np.arange(len(sub_exp_time_list))
        # ax[0].bar(x-width, sift_precision_list, width, label = 'Sift')
        # ax[0].bar(x, surf_precision_list, width, label = 'Surf')
        # ax[0].bar(x+width, orb_precision_list, width, label = 'Orb')
        # ax[0].set_xticks(x)
        # ax[0].set_xticklabels(sub_exp_time_list)
        # box = ax[0].get_position()
        # ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height*0.5])
        # ax[0].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        # ax[0].set_title("Precision Graph for " + percent[i] + " Gain (" + gain_values[i].replace("gain", "") + "/30x)")
        # ax[0].set(xlabel = 'Exposure Time (ms)', ylabel = 'Precision')
        # ax[0].set_ylim(0, 1.1)
        # ax[0].set_yticks(np.arange(0,1.1, step = 0.25))

        # ax[1].bar(x-width, sift_num_match_list, width, label = 'Sift')
        # ax[1].bar(x, surf_num_match_list, width, label = 'Surf')
        # ax[1].bar(x+width, orb_num_match_list, width, label = 'Orb')
        # ax[1].set_xticks(x)
        # ax[1].set_xticklabels(sub_exp_time_list)
        # box = ax[1].get_position()
        # ax[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # ax[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        # ax[1].set_title("Number of Matches")
        # ax[1].set(xlabel = 'Exposure Time (ms)', ylabel = 'Number of Matches')
        # ax[1].set_ylim(0, 21)
        # ax[1].set_yticks(np.arange(0,21, step = 5))


        # plt.show()
        plt.savefig(location_name + '_' + gain_values[i] + '.png', bbox_inches = 'tight', pad_inches = 0)
        # plt.savefig(location_name + '_' + gain_values[i] + '.png')
        
        #reset lists
        sub_exp_time_list = []
        sift_precision_list = []
        surf_precision_list = []
        orb_precision_list = []
        sift_num_match_list = []
        surf_num_match_list = []
        orb_num_match_list = []

#graphs as a function of gain
def graph_fn_gain(gain_precision_list):
    save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Thesis 2020\Graphs'
    location_name = input('Type in a location name for the graphs: ')
    save_path += '\\' + location_name

    #make a new folder for location
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_path += '\\' + 'exp_graphs'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    os.chdir(save_path)

    width = 0.26

    gain_list = []
    exp_time_list = []
    precision_list = []
    
    sub_gain_list = [] #for looping through each value in a gain category
    
    sift_precision_list = []
    surf_precision_list = []
    orb_precision_list = []

    sift_num_match_list = []
    surf_num_match_list = []
    orb_num_match_list = []

    sorted_list = sort_list(gain_precision_list)
    for element in sorted_list:
        gain_list.append(element[0])
        exp_time_list.append(element[1])
        precision_list.append(element[2])

    #gain_values contains just the four different values
    exp_values = natsorted(list(set(exp_time_list)))
    gain_values = natsorted(list(set(gain_list)))
    print(exp_values)
    percent = []
    if len(gain_values) == 4:
        percent = ['25%', '50%', '75%', '100%']
    else:
        percent = ['0%', '25%', '50%', '75%', '100%']

    shutter_speed = ['1/120', '1/60', '1/30', '1/15']

    for i in range(len(exp_values)):
        for j in range(len(exp_time_list)):
            if exp_values[i] == exp_time_list[j]:
                index = gain_values.index(gain_list[j])
                sub_gain_list.append(percent[index])

                sift_num_match_list.append(precision_list[j][0][0])
                surf_num_match_list.append(precision_list[j][1][0])
                orb_num_match_list.append(precision_list[j][2][0])

                sift_precision_list.append(precision_list[j][0][1])
                surf_precision_list.append(precision_list[j][1][1])
                orb_precision_list.append(precision_list[j][2][1])
        
        #make x-axis percent list instead**

        #subplots(row, cols)
        fig, ax = plt.subplots(2, 1)
        # fig, ax = plt.subplots(1)
        fig.tight_layout(pad = 4.0)
        # fig.suptitle(location_name, fontsize = 18)
        fig.subplots_adjust(top = 0.85)

        # fig, precision_ax = plt.subplots()
        # x = np.arange(len(sub_gain_list))
        # ax.bar(x-width, sift_precision_list, width, label = 'Sift')
        # ax.bar(x, surf_precision_list, width, label = 'Surf')
        # ax.bar(x+width, orb_precision_list, width, label = 'Orb')
        # ax.set_xticks(x)
        # ax.set_xticklabels(sub_gain_list)
        # box = ax.get_position()
        # # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # ax.set_position([box.x0, box.y0, box.width * 0.5, box.height*0.5])
        # ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        # ax.set_title("Precision Graph for " + shutter_speed[i] + " Shutter Speed")
        # ax.set(xlabel = 'Gain', ylabel = 'Precision')
        # ax.set_ylim(0, 1.1)
        # ax.set_yticks(np.arange(0,1.1, step = 0.25))

        x = np.arange(len(sub_gain_list))
        ax[0].bar(x-width, sift_precision_list, width, label = 'Sift')
        ax[0].bar(x, surf_precision_list, width, label = 'Surf')
        ax[0].bar(x+width, orb_precision_list, width, label = 'Orb')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(sub_gain_list)
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax[0].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        ax[0].set_title("Precision Graph for " + shutter_speed[i] + " Shutter Speed")
        ax[0].set(xlabel = 'Gain', ylabel = 'Precision')
        ax[0].set_ylim(0, 1.1)
        ax[0].set_yticks(np.arange(0,1.1, step = 0.25))

        ax[1].bar(x-width, sift_num_match_list, width, label = 'Sift')
        ax[1].bar(x, surf_num_match_list, width, label = 'Surf')
        ax[1].bar(x+width, orb_num_match_list, width, label = 'Orb')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(sub_gain_list)
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        ax[1].set_title("Number of Matches")
        ax[1].set(xlabel = 'Gain', ylabel = 'Number of Matches')
        ax[1].set_ylim(0, 21)
        ax[1].set_yticks(np.arange(0,21, step = 5))


        plt.savefig(location_name + '_' + exp_values[i] + '.png', bbox_inches = 'tight', pad_inches = 0)
        # plt.savefig(location_name + '_' + exp_values[i] + '.png')
        
        for indices in range(len(sub_gain_list)):
            print("Gain: " + str(sub_gain_list[indices]))
            print("Exposure Time: " + str(exp_values[i]))
            print("SIFT Precision: " + str(sift_precision_list[indices]))
            print("SURF Precision: " + str(surf_precision_list[indices]))
            print("ORB Precision: " + str(orb_precision_list[indices]))
            print()

        #reset lists
        sub_gain_list = []
        sift_precision_list = []
        surf_precision_list = []
        orb_precision_list = []
        sift_num_match_list = []
        surf_num_match_list = []
        orb_num_match_list = []


def heatmap(precision_gain_list):
    save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\Thesis 2020\Graphs'
    location_name = input('Type in a location name for the graphs: ')
    save_path += '\\' + location_name

    #make a new folder for location
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_path += '\\' + 'heatplots'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    os.chdir(save_path)

    gain_list = []
    exp_time_list = []
    precision_list = []
    
    sub_exp_time_list = [] #for looping through each value in a gain category
    
    sift_precision_list = []
    surf_precision_list = []
    orb_precision_list = []

    sift_heatmap = []
    surf_heatmap = []
    orb_heatmap = []

    sorted_list = sort_list(precision_gain_list)
    for element in sorted_list:
        gain_list.append(element[0])
        exp_time_list.append(element[1].replace('ms', ''))
        precision_list.append(element[2])

    #gain_values contains just the four different values
    gain_values = natsorted(list(set(gain_list)))
    percent = []
    if len(gain_values) == 4:
        percent = ['25%', '50%', '75%', '100%']
    else:
        percent = ['0%', '25%', '50%', '75%', '100%']

    shutter_speed = ['1/120', '1/60', '1/30', '1/15']

    for i in range(len(gain_values)):
        for j in range(len(gain_list)):
            if gain_values[i] == gain_list[j]:
                sub_exp_time_list.append(exp_time_list[j])

                sift_precision_list.append(precision_list[j][0][1])
                surf_precision_list.append(precision_list[j][1][1])
                orb_precision_list.append(precision_list[j][2][1])
        
        sift_heatmap.append(sift_precision_list)
        surf_heatmap.append(surf_precision_list)
        orb_heatmap.append(orb_precision_list)

        sift_precision_list = []
        surf_precision_list = []
        orb_precision_list = []
    
    # SIFT
    fig, ax = plt.subplots()
    sift_heatplot = ax.imshow(sift_heatmap, cmap = "Greens")
    ax.set_xticks(np.arange(len(shutter_speed)))
    ax.set_yticks(np.arange(len(percent)))
    ax.set_xticklabels(shutter_speed)
    ax.set_yticklabels(percent)
    ax.set(xlabel = 'Shutter Speed (s)', ylabel = "Gain")
    ax.set_title("SIFT Precision Heat Map")

    for i in range(len(sift_heatmap)):
        for j in range(len(sift_heatmap[i])):
            text = ax.text(j, i, sift_heatmap[i][j],
                        ha="center", va="center", color="black")

    fig.tight_layout()
    colorbar = fig.colorbar(sift_heatplot)
    plt.savefig('sift_precision_heatmap.png')

    # SURF
    fig, ax = plt.subplots()
    surf_heatplot = ax.imshow(surf_heatmap, cmap = "Greens")
    ax.set_xticks(np.arange(len(shutter_speed)))
    ax.set_yticks(np.arange(len(percent)))
    ax.set_xticklabels(shutter_speed)
    ax.set_yticklabels(percent)
    ax.set(xlabel = 'Shutter Speed (s)', ylabel = "Gain")
    ax.set_title("SURF Precision Heat Map")

    for i in range(len(surf_heatmap)):
        for j in range(len(surf_heatmap[i])):
            text = ax.text(j, i, surf_heatmap[i][j],
                        ha="center", va="center", color="black")

    fig.tight_layout()
    colorbar = fig.colorbar(surf_heatplot)
    plt.savefig('surf_precision_heatmap.png')

    # ORB
    fig, ax = plt.subplots()
    orb_heatplot = ax.imshow(orb_heatmap, cmap = "Greens")
    ax.set_xticks(np.arange(len(shutter_speed)))
    ax.set_yticks(np.arange(len(percent)))
    ax.set_xticklabels(shutter_speed)
    ax.set_yticklabels(percent)
    ax.set(xlabel = 'Shutter Speed (s)', ylabel = "Gain")
    ax.set_title("ORB Precision Heat Map")

    for i in range(len(orb_heatmap)):
        for j in range(len(orb_heatmap[i])):
            text = ax.text(j, i, orb_heatmap[i][j],
                        ha="center", va="center", color="black")

    fig.tight_layout()
    colorbar = fig.colorbar(orb_heatplot)
    plt.savefig('orb_precision_heatmap.png')

    # sift_heatplot = ax.imshow(sift_heatmap, cmap = "Greens")
    # ax.set_xticks(np.arange(len(shutter_speed)))
    # ax.set_yticks(np.arange(len(percent)))
    # ax.set_xticklabels(shutter_speed)
    # ax.set_yticklabels(percent)
    # ax.set_title("SIFT Heatmap")

    # for i in range(len(percent)):
    #     for j in range(len(shutter_speed)):
    #         text = ax.text(j, i, sift_heatmap[i, j],
    #                     ha="center", va="center", color="w")

    # sift_heatplot = ax.imshow(sift_heatmap, cmap = "Greens")
    # ax.set_xticks(np.arange(len(shutter_speed)))
    # ax.set_yticks(np.arange(len(percent)))
    # ax.set_xticklabels(shutter_speed)
    # ax.set_yticklabels(percent)
    # ax.set_title("SIFT Heatmap")

    # for i in range(len(percent)):
    #     for j in range(len(shutter_speed)):
    #         text = ax.text(j, i, sift_heatmap[i, j],
    #                     ha="center", va="center", color="w")


def orb_feature_detection(frame):
    width = 640
    height = 360

    # image_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\monochrome\monochrome_camera\arlington_center\gain2.79x'
    # image = input("Type in image name: ")
    # image += '.jpg'
    # image = image_path + '\\' + image

    # img = cv2.resize(cv2.imread(image, 0), (width, height))
    img = cv2.resize(frame, (width, height))
    img_copy = np.copy(img)

    x = "5"
    blurred = cv2.GaussianBlur(img_copy, (int(x), int(x)), 0)

    # grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    erode = cv2.erode(blurred, None, iterations = 2)
    dilate = cv2.dilate(erode, None, iterations = 3)

    orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(erode, None)
    kp2, des2 = orb.detectAndCompute(dilate, None)
    kp3, des3 = orb.detectAndCompute(img, None)

    # keypoint_erode = cv2.drawKeypoints(erode, kp1, None)
    # keypoint_dilate = cv2.drawKeypoints(dilate, kp2, None)
    # keypoint_img = cv2.drawKeypoints(img, kp3, None)
    # keypoint_img = cv2.drawKeypoints(img, kp1, None)
    keypoint_img = cv2.drawKeypoints(img, kp2, None)

    # cv2.imshow("keypoints erode", keypoint_erode)
    # cv2.imshow("keypoints dilate", keypoint_dilate)
    cv2.imshow("keypoints img", cv2.rotate(keypoint_img, cv2.ROTATE_90_CLOCKWISE))
    # cv2.imshow("blurred", blurred)
    # cv2.imshow("erode", erode)
    # cv2.imshow("dilate", dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
Instructions: need to change folder path to run analysis
Change save path for number of matches
'''
def analyze_image_dataset():
    gain_precision_list = []

    location = input("Enter location: ")
    folder = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\monochrome\monochrome_camera'
    folder += "\\" + location

    folder_date = input("Enter today's date: ")

    #CHANGE SAVE PATH
    save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\monochrome\self_drawn\monochrome_processed_20_matches'
    # save_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\monochrome\monochrome_processed_20_matches'
    save_path += "\\" + folder_date
    #create folder if the date does not exist
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)

    option = input("Type: auto or drawn?\n")
    #add this folder

    pic_array = ['8.33ms', '16.67ms', '33.33ms', '66.67ms']

    for root, dirs, files in os.walk(folder, topdown = True):
        location = root.split('\\')
        location = location[len(location)-1]
        save_path += '\\' + location
        if os.path.isdir(save_path):
            answer = input('Analysis for location exists: continue? Y/N\n')
            if answer == 'N':
                print('Exiting.')
                exit()
        else:
            os.mkdir(save_path)
        #"folders" is the gain folder name
        for gain_category in dirs:
            gain_folder = save_path + '\\' + gain_category
            if not os.path.isdir(gain_folder):
                os.mkdir(gain_folder)
            for i in pic_array:
                base_name = gain_category + '_exp' + i
                gain_exp_folder = gain_folder + '\\' + base_name
                if not os.path.isdir(gain_exp_folder):
                    os.mkdir(gain_exp_folder)
                os.chdir(gain_exp_folder)
                left_image = base_name + '_left.jpg'
                right_image = base_name + '_right.jpg'
                left_picture_path = folder + '\\' + gain_category + '\\' + left_image
                right_picture_path = folder + '\\' + gain_category + '\\' + right_image
                precision_list = match_features(left_picture_path, right_picture_path, gain_exp_folder, left_image, right_image, location)
                # print(precision_list)
                g = gain(gain_category, i, precision_list)
                gain_precision_list.append(g)
        break

    return gain_precision_list

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


if __name__ == '__main__':
    # video_path = r'C:\Users\alexy\Downloads\Side view of night driving.mp4'
    video_path = r'C:\Users\alexy\Videos\20200601_231652.mp4'
    # gain_precision_list = analyze_image_dataset()
    # graph_fn_gain(gain_precision_list)

    # #do graphs here
    # graph_fn_exp(gain_precision_list)
    # heatmap(gain_precision_list)

    # orb_feature_detection()
    frame_list = extract_images_from_video(video_path)
    # for frame in frame_list:
    #     orb_feature_detection(frame)
    # img1_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\Shape Matching Image Tests\gain17.5x_exp16.67ms_left.jpg'
    # img2_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\Shape Matching Image Tests\gain17.5x_exp16.67ms_left.jpg'
    # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)