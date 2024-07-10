import pandas as pd
import numpy as np
from tabulate import tabulate 
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    v1_unit = unit_vector(v1)
    v2_unit = unit_vector(v2)

    print( v1_unit)
    print( v2_unit)
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))*180/np.pi


def main():

    args = sys.argv
    i = args[1]
    directory = "C:\\Users\\Anthony\\Documents\\Experiments_LCG\\static_experiments\\experiment_7_2\\exp"

    error_file = "\\distance_error.csv"
    local_pts_est = "\\local_cartesian_exp"
    local_pts_GT = "\\local_cartesian_GT_exp"
    local_cal_pts = "\\local_cartesian_cal"

    # Analyzing error data  
    error_file = directory + str(i) + error_file
    err_data = pd.read_csv(error_file)

    columns = err_data.columns.values.tolist()
    dataSP = err_data[columns[0]].tolist()
    dataVal = err_data[columns[1]].tolist()

    # table = zip(dataSP, dataVal)

    # Analyzing Local Cartesian Points 

    ## Analyze depth (GT and Estimated)

    center = np.zeros((1,3))
    # Estimated Data
    local_cartesian_points = directory + str(i) + local_pts_est + str(i) + ".csv"
    lc_data = pd.read_csv(local_cartesian_points)
    # print(lc_data)

    lc_pts = np.array([lc_data['x'].tolist(), lc_data['y'].tolist(), lc_data['z'].tolist()])

    # Grount Truth Data
    local_cartesian_points_GT = directory + str(i) + local_pts_GT + str(i) + ".csv"
    lc_data_GT = pd.read_csv(local_cartesian_points_GT)

    lc_pts_GT = np.array([lc_data_GT['x'].tolist(), lc_data_GT['y'].tolist(), lc_data_GT['z'].tolist()])
    scenePts = lc_data_GT['scenePt'].tolist()
 

    ## Read Local Calibration Points
    local_cartesian_points_cal = directory + str(i) + local_cal_pts + str(i) + ".csv"
    lc_data_cal = pd.read_csv(local_cartesian_points_cal)
    # print(lc_data_cal )

    lc_pts_cal = np.array([lc_data_cal['x'].tolist(), lc_data_cal['y'].tolist(), lc_data_cal['z'].tolist()])

    depth_est = []

    for ind in range(lc_pts.shape[1]):
        dist = np.linalg.norm(center - lc_pts[:, ind])
        depth_est.append(dist)
    

    depth_GT = []
    for ind in range(lc_pts_GT.shape[1]):
        dist = np.linalg.norm(center - lc_pts_GT[:, ind])
        depth_GT.append(dist)

    depth_cal = []
    for ind in range(lc_pts_cal.shape[1]):
        dist = np.linalg.norm(center - lc_pts_cal[:, ind])
        depth_cal.append(dist)

    angle_diff = []
    for ind in range(lc_pts_GT.shape[1]):
        angle_diff.append(angle(lc_pts[:, ind], lc_pts_GT[:, ind]))

    table = zip(dataSP, dataVal, angle_diff, depth_est, depth_GT, depth_cal)
    columns += ['angle', 'depth (est)(m)', 'depth (GT)(m)', 'depth (local)(m)']
    print(tabulate(table, columns))
    

    # 3D data Understanding
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.view_init(azim=90, elev=-1, roll=180)
    # ax.set_xlabel('$X$', fontsize=10)
    # ax.set_ylabel('$Y$', fontsize=10)
    # ax.set_zlabel('$Z$', fontsize=10, rotation = 0)
    # ax.scatter3D(lc_pts_cal[0,:], lc_pts_cal[1,:], lc_pts_cal[2,:], c=lc_pts_cal[2,:], cmap='Greens') 
    # plt.show()

    lc_pts_cal_plot = lc_pts_cal.T
    center_point = np.array([[0,0,0]])
    lc_pts_cal_plot = np.vstack((center_point, lc_pts_cal_plot))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lc_pts_cal_plot)
    colors = np.zeros((lc_pts_cal.shape[1], 3))
    colors[0,:] = np.array([0.0,0.9,0.0])
    colors[1:,:] = np.array([0.0,0.0,0.99])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd2 = o3d.geometry.PointCloud()
    lc_pts_est = lc_pts.T #np.vstack((center_point, lc_pts.T))
    pcd2.points = o3d.utility.Vector3dVector(lc_pts_est)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd3 = o3d.geometry.PointCloud()
    lc_pts_gt = lc_pts_GT.T #np.vstack((center_point, lc_pts_GT.T))
    pcd3.points = o3d.utility.Vector3dVector(lc_pts_gt)

    #o3d.visualization.draw_geometries([pcd])
    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()
    # o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh.create_coordinate_frame(), pcd])
    # o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh.create_coordinate_frame(), pcd2, pcd3])
    #o3d.visualization.draw_geometries([pcd])
    # viewer.add_geometry([pcd])
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([0.5, 0.5, 0.5])
    # viewer.run()
    #tabulate()

    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)

    window.add_child(scene)

    matGT = rendering.MaterialRecord()
    matGT.shader = 'defaultUnlit'
    matGT.point_size = 7.0
    matGT.base_color = np.ndarray(shape=(4,1), buffer=np.array([0.0, 0.0, 1.0, 1.0]), dtype=float)

    matEst = rendering.MaterialRecord()
    matEst.shader = 'defaultUnlit'
    matEst.point_size = 7.0
    matEst.base_color = np.ndarray(shape=(4,1), buffer=np.array([1.0, 0.0, 0.0, 1.0]), dtype=float)
   

    scene.scene.add_geometry("mesh_name", pcd2, matEst)
    scene.scene.add_geometry("mesh_name2", pcd3, matGT)
    scene.scene.add_geometry("mesh_name3", o3d.geometry.TriangleMesh.create_coordinate_frame(), rendering.MaterialRecord())

    bounds = pcd2.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    labels = [[0, 0, 0]]
    #for coordinate in labels:
    print(lc_pts_est.shape[0])
    print(lc_pts_gt.shape[0])
    # scenePts = [len(scenePts)] + scenePts
    # scenePts
    print(scenePts)
    for row in range(lc_pts_est.shape[0]):
        coordinate = lc_pts_est[row,:]
        count = scenePts[row]
        scene.add_3d_label(coordinate, str(count))

    for row in range(lc_pts_gt.shape[0]):
        coordinate = lc_pts_gt[row,:]
        count = scenePts[row]
        scene.add_3d_label(coordinate, str(count))
    gui.Application.instance.run()  # Run until user closes window

main()