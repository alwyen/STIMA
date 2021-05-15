import cv2
from matplotlib import pyplot as plt

def average_pixel(image, left, right):

    total_left = 0
    for i in range(left[0]-2, left[0]+2):
        for j in range(left[1]-2, left[1]+2):
            total_left += image[i][j]

    total_right = 0
    for i in range(right[0]-2, right[0]+2):
        for j in range(right[1]-2, right[1]+2):
            total_right += image[i][j]

    avg_left = total_left/25
    avg_right = total_right/25

    return [avg_left, avg_right]

#normalized graph based on celsius
def graph_celsius():
    folder_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\bulb_varyac'

    voltage = [25, 35, 45, 55, 65, 75, 85, 95]
    celcius = [40, 46, 50.5, 55, 60.5, 64, 71, 74]
    left_pixels = [[126,80], [122,75], [121,75], [126,74], [124,71], [126,72], [126, 73], [125, 73]]
    right_pixels = [[188,156], [185,151], [182,153], [181,154], [184,152], [186,153], [182,153], [181,156]]

    left_bulb = []
    right_bulb = []

    for i in range(len(voltage)):
        image_path = folder_path + "\\" + str(voltage[i]) + "V.jpg"
        image = cv2.imread(image_path)
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixel_avgs = average_pixel(grayed, left_pixels[i], right_pixels[i])
        left_bulb.append(pixel_avgs[0])
        right_bulb.append(pixel_avgs[1])

    plt.plot(celcius, left_bulb, label = "60W Incandescent")
    plt.plot(celcius, right_bulb, label = "25W Incandescent")
    plt.ylabel("Grayscaled Pixel Value")
    plt.xlabel("Degrees in Celsius (of 25W Incandescent)")
    plt.title("Pixel Value vs. Degrees in Celsius")
    plt.legend()
    plt.show()

#this method doesn't make sense
def graph_voltage():
    image_path = r'C:\Users\alexy\OneDrive\Documents\STIMA\scripts\yen_outdoor_tests\ir_camera\bulb_varyac'

    voltage = [25, 35, 45, 55, 65, 75, 85, 95]


if __name__ == "__main__":
    graph_celsius()
