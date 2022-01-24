import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from scipy.signal import convolve
from scipy.ndimage import maximum_filter
import math

image_paths_0 = r'C:\Users\alexy\OneDrive\Documents\STIMA\Images\Rectification_Test'

def grayscale(I):
    r = I[:,:,0]
    g = I[:,:,1]
    b = I[:,:,2]
        
    return (0.21*r + 0.72*g + 0.07*b)

def ImageGradient(I):

    # inputs: 
    # I is the input image (may be mxn for Grayscale or mxnx3 for RGB)
    #
    # outputs:
    # Ix is the derivative of the magnitude of the image w.r.t. x
    # Iy is the derivative of the magnitude of the image w.r.t. y
    
    m, n = I.shape[:2] #why do we need this?
    
    if len(I.shape) == 3:
        grayscaled = grayscale(I)
    else:
        grayscaled = I
    
    Ix = np.zeros(grayscaled.shape)
    Iy = np.zeros(grayscaled.shape)
    
    k_h = np.array([[1/12, -2/3, 0, 2/3, -1/12]])
    k_v = np.array([[1/12], [-2/3], [0], [2/3], [-1/12]])
    
    pad_width = len(k_h[0])//2

    #try doing valid and then do zero padding
    Ix_unpadded = convolve(grayscaled, k_h, mode = 'valid')
    Iy_unpadded = convolve(grayscaled, k_v, mode = 'valid')
    
    Ix[:,pad_width:Ix.shape[1]-pad_width] = Ix_unpadded
    Iy[pad_width:Iy.shape[0]-pad_width,:] = Iy_unpadded
    
    return Ix, Iy

def MinorEigenvalueImage(Ix, Iy, w):
    # Calculate the minor eigenvalue image J
    #
    # inputs:
    # Ix is the derivative of the magnitude of the image w.r.t. x
    # Iy is the derivative of the magnitude of the image w.r.t. y
    # w is the size of the window used to compute the gradient matrix N
    #
    # outputs:
    # J0 is the mxn minor eigenvalue image of N before thresholding

    assert Ix.shape == Iy.shape
    
    m, n = Ix.shape[:2]
    J0 = np.zeros((m,n))

    #Calculate your minor eigenvalue image J0.
    
    Ix_2 = Ix**2
    Iy_2 = Iy**2
    Ix_Iy = Ix*Iy
    
#     for i in range(w//2, m-w//2):
#         for j in range(w//2, n-w//2):
    #getting corners on edges, so reducing computation on the edges
    for i in range(w, m-w):
        for j in range(w, n-w):
            row_start = i-w//2
            row_end = i+w//2
            col_start = j-w//2
            col_end = j+w//2
                        
            Ix_2_window = Ix_2[row_start:(row_end+1),col_start:(col_end+1)]
            Iy_2_window = Iy_2[row_start:(row_end+1),col_start:(col_end+1)]
            Ix_Iy_window = Ix_Iy[row_start:(row_end+1),col_start:(col_end+1)]
            
            sum_Ix_squared_window = np.sum(Ix_2_window)
            sum_Iy_squared_window = np.sum(Iy_2_window)
            sum_Ix_Iy_window = np.sum(Ix_Iy_window)
            
            C = np.array([[sum_Ix_squared_window, sum_Ix_Iy_window], [sum_Ix_Iy_window, sum_Iy_squared_window]])
            
            trace_C = C[0][0] + C[1][1]
            det_C = C[0][0]*C[1][1] - C[0][1]*C[1][0]
            
            x = trace_C**2 - 4*det_C
            eig_min = (trace_C - math.sqrt(max(0,x)))/2
            
            J0[i][j] = eig_min
            
#             if j < 20 and eig_min > 0.2:
#                 print(eig_min)
    
    return J0

def NMS(J, w_nms):
    # Apply nonmaximum supression to J using window w_nms
    #
    # inputs: 
    # J is the minor eigenvalue image input image after thresholding
    # w_nms is the size of the local nonmaximum suppression window
    # 
    # outputs:
    # J2 is the mxn resulting image after applying nonmaximum suppression
    # 
    
    J2 = J.copy()
    J_copy = np.pad(J.copy(), (w_nms//2, w_nms//2))
    J2 = np.pad(J2, (w_nms//2, w_nms//2))
    m, n = J2.shape[:2]
    
    for i in range(w_nms//2, m-w_nms//2):
        for j in range(w_nms//2, n-w_nms//2):
            row_start = i-w_nms//2
            row_end = i+w_nms//2
            col_start = j-w_nms//2
            col_end = j+w_nms//2
            
            window = J2[row_start:(row_end+1),col_start:(col_end+1)]
            window_maximums = maximum_filter(window, w_nms)
            window_max = window_maximums[w_nms//2][w_nms//2]

            num_maxes = len(np.where(window == window_max)[0])
            
            if J_copy[i][j] < window_max:
                J2[i][j] = 0

    return J2[w_nms//2:(m-w_nms//2+1),w_nms//2:(n-w_nms//2-1)]

def ForstnerCornerDetector(Ix, Iy, w, t, w_nms):
    # Calculate the minor eigenvalue image J
    # Threshold J
    # Run non-maxima suppression on the thresholded J
    # Gather the coordinates of the nonzero pixels in J 
    # Then compute the sub pixel location of each point using the Forstner operator
    #
    # inputs:
    # Ix is the derivative of the magnitude of the image w.r.t. x
    # Iy is the derivative of the magnitude of the image w.r.t. y
    # w is the size of the window used to compute the gradient matrix N
    # t is the minor eigenvalue threshold
    # w_nms is the size of the local nonmaximum suppression window
    #
    # outputs:
    # C is the number of corners detected in each image
    # pts is the 2xC array of coordinates of subpixel accurate corners
    #     found using the Forstner corner detector
    # J0 is the mxn minor eigenvalue image of N before thresholding
    # J1 is the mxn minor eigenvalue image of N after thresholding
    # J2 is the mxn minor eigenvalue image of N after thresholding and NMS

#     m, n = Ix.shape[:2]
#     J0 = np.zeros((m,n))
#     J1 = np.zeros((m,n))

    Ix_2 = Ix**2
    Iy_2 = Iy**2
    Ix_Iy = Ix*Iy

    #Calculate your minor eigenvalue image J0 and its thresholded version J1.
    J0 = MinorEigenvalueImage(Ix, Iy, w)
    J1 = np.copy(J0)
    J1[J1 < t] = 0
    #Run non-maxima suppression on your thresholded minor eigenvalue image.
    J2 = NMS(J1, w_nms)
    m,n = J2.shape[:2]
    
    #Detect corners.
    #crop border and pad to remove points along the border:
    num_pixels_cropped = 10
    J2 = J2[num_pixels_cropped:m-num_pixels_cropped, num_pixels_cropped:n-num_pixels_cropped]
    J2 = np.pad(J2, (num_pixels_cropped))
    pts_c1, pts_c2 = np.where(J2 > 0)
    
    assert len(pts_c1) == len(pts_c2)
    
    corners_x = np.array([])
    corners_y = np.array([])
    
    for i in range(len(pts_c1)):
        row_start = pts_c1[i]-w//2
        row_end = pts_c1[i]+w//2
        col_start = pts_c2[i]-w//2
        col_end = pts_c2[i]+w//2
        
        Ix_2_window = Ix_2[row_start:(row_end+1),col_start:(col_end+1)]
        Iy_2_window = Iy_2[row_start:(row_end+1),col_start:(col_end+1)]
        Ix_Iy_window = Ix_Iy[row_start:(row_end+1),col_start:(col_end+1)]
        
        sum_Ix_squared_window = np.sum(Ix_2_window)
        sum_Iy_squared_window = np.sum(Iy_2_window)
        sum_Ix_Iy_window = np.sum(Ix_Iy_window)
        
        C = np.array([[sum_Ix_squared_window, sum_Ix_Iy_window], [sum_Ix_Iy_window, sum_Iy_squared_window]])
        C_inv = np.linalg.inv(C)
        #pts_c1 --> y
        #pts_c2 --> x
        sol = np.array([[np.sum(pts_c2[i]*Ix_2_window + pts_c1[i]*Ix_Iy_window)], [np.sum(pts_c2[i]*Ix_Iy_window + pts_c1[i]*Iy_2_window)]])
#         new_corner = np.matmul(np.linalg.inv(C),sol)
        new_corner_x = C_inv[0][0]*sol[0][0] + C_inv[0][1]*sol[1][0]
        new_corner_y = C_inv[1][0]*sol[0][0] + C_inv[1][1]*sol[1][0]
        
#         corners_x = np.hstack((corners_x, new_corner[0]))
#         corners_y = np.hstack((corners_y, new_corner[1]))
        corners_x = np.hstack((corners_x, new_corner_x))
        corners_y = np.hstack((corners_y, new_corner_y))
#         print(new_corner_x)
#         print(new_corner_y)
#         print()
    
    pts = np.vstack((corners_x,corners_y))
#     print(pts)
    C = len(pts[0])
    
    return C, pts, J0, J1, J2

# feature detection
def RunFeatureDetection(I, w, t, w_nms):
    Ix, Iy = ImageGradient(I)
    C, pts, J0, J1, J2 = ForstnerCornerDetector(Ix, Iy, w, t, w_nms)
    return C, pts, J0, J1, J2



def corner_detection(image_path):
    left_img = cv2.imread(image_path + '\\left_0.jpg')
    right_img = cv2.imread(image_path + '\\right_0.jpg')

    left_gray = cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)

    left_gray = np.float32(left_gray)
    right_gray = np.float32(right_gray)

    left_dst = cv2.cornerHarris(left_gray,2,3,0.04)
    right_dst = cv2.cornerHarris(right_gray,2,3,0.04)

    print(left_dst.shape)
    print(right_dst.shape)

    #result is dilated for marking the corners, not important
    # left_dst = cv2.dilate(left_dst,None)
    # right_dst = cv2.dilate(right_dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    left_img[left_dst>0.01*left_dst.max()]=[0,0,255]
    right_img[right_dst>0.01*right_dst.max()]=[0,0,255]

    cv2.imshow('dst',left_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    cv2.imshow('dst',right_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # corner_detection(image_paths_0)

    # input images
    I1 = np.array(Image.open(image_paths_0 + '\\left_0.jpg'), dtype='float')/255.
    I2 = np.array(Image.open(image_paths_0 + '\\right_0.jpg'), dtype='float')/255.

    # parameters to tune
    w = 25 #smaller window reduces window visibility
    t = 0.1
    w_nms = 15 #larger window reduces close corners together

    tic = time.time()
    # run feature detection algorithm on input images
    C1, pts1, J1_0, J1_1, J1_2 = RunFeatureDetection(I1, w, t, w_nms)
    print('Left Done')
    print(f'{time.time() - tic} Seconds')
    # C2, pts2, J2_0, J2_1, J2_2 = RunFeatureDetection(I2, w, t, w_nms)
    # toc = time.time() - tic

    print('took %f secs'%toc)

    # display results
    plt.figure(figsize=(14,24))

    # show pre-thresholded minor eigenvalue images
    plt.subplot(3,2,1)
    plt.imshow(J1_0, cmap='gray')
    plt.title('pre-thresholded minor eigenvalue image')
    plt.subplot(3,2,2)
    plt.imshow(J2_0, cmap='gray')
    plt.title('pre-thresholded minor eigenvalue image')
    plt.show()

    # # show thresholded minor eigenvalue images
    # plt.subplot(3,2,3)
    # plt.imshow(J1_1, cmap='gray')
    # plt.title('thresholded minor eigenvalue image')
    # plt.subplot(3,2,4)
    # plt.imshow(J2_1, cmap='gray')
    # plt.title('thresholded minor eigenvalue image')
    # plt.show()

    # show corners on original images
    ax = plt.subplot(3,2,5)
    plt.imshow(I1)
    for i in range(C1): # draw rectangles of size w around corners
        x,y = pts1[:,i]
        ax.add_patch(patches.Rectangle((x-w/2,y-w/2),w,w, fill=False))
    # plt.plot(pts1[0,:], pts1[1,:], '.b') # display subpixel corners
    plt.title('found %d corners'%C1)
    plt.show()

    # ax = plt.subplot(3,2,6)
    # plt.imshow(I2)
    # for i in range(C2):
    #     x,y = pts2[:,i]
    #     ax.add_patch(patches.Rectangle((x-w/2,y-w/2),w,w, fill=False))
    # # plt.plot(pts2[0,:], pts2[1,:], '.b')
    # plt.title('found %d corners'%C2)
    # plt.show()