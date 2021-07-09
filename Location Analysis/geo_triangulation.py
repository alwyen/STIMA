import numpy as np

def Sinc(x):
    if x == 0:
        return 1
    else:
        return np.sin(x)/x

def skew(w):
    # Returns the skew-symmetrix represenation of a vector
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def Deparameterize(w):
    # Deparameterizes to get rotation matrix
    mag_w = np.linalg.norm(w)
    R = np.eye(3) + Sinc(mag_w)*skew(w) + (1 - np.cos(mag_w))/(mag_w**2)*skew(w)@skew(w)
    return R

if __name__ == '__main__':
    w = np.array([-0.02360, -0.00790, -0.02877])
    t1 = np.array([0, 0, 0]).reshape(-1,1)
    t2 = np.array([-278.75844, 10.10778, 17.38857]).reshape(-1,1)
    R = Deparameterize(w)
    print(R)