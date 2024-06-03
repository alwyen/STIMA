import cv2
import os

height = 960
width = 2560

def get_max_num_images(home_path):
    img_list = os.listdir(home_path)
    count_list = list()
    for img in img_list:
        img_name = img.split('.')[0]
        img_count = int(img_name.split('_')[-1])
        count_list.append(img_count)

    return max(count_list)


HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1], 'stereopi-setup', 'calibration')
def main():
    num_images = get_max_num_images(HOME)
    for i in range(1,num_images+1):
        if i%10 == 0:
            print(f'Image {i}/{num_images} done.')
            
        img = cv2.imread(HOME + "/scene_{}x{}_{}.png".format(width, height, i))

        img1 = img[:, :int(img.shape[1]/2), :]
        img2 = img[:, int(img.shape[1]/2):, :]

        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
        
        if not os.path.exists('calibration_image_pairs'):
            os.mkdir('calibration_image_pairs')
        if not os.path.exists('calibration_image_pairs/left_camera'):
            os.mkdir('calibration_image_pairs/left_camera')
        if not os.path.exists('calibration_image_pairs/right_camera'):
            os.mkdir('calibration_image_pairs/right_camera')

        cv2.imwrite("calibration_image_pairs/left_camera/pair_Left{}.png".format(i), img1)
        cv2.imwrite("calibration_image_pairs/right_camera/pair_Right{}.png".format(i), img2)


if __name__ == "__main__":
    main()
