from triangulation_projective_geo import *

'''
Create a CSV file of all the param names as well as the increment value; this will require manual thinking and intuition on the value of the increment
go through each row and run through inner loop of increment values (starting from -[increment_val]*5 to [increment_val]*5)

also extracts all values to input into param_quant

Input: number_increments        number of increments +/-; e.g. 5 with increments value == 2 --> -10, -8, -6, ... , 6, 8, 10

TODO: Characterize each parameter by quantizing the amount of error reduced?

'''
def error_quant_analysis(number_increments, csv_file, save_path):
    file_df = pd.read_csv(csv_file)
    param_name_list = file_df.Param_Name.tolist()
    increment_val_list = file_df.Increment_Val.tolist()

    geo_centric_detic_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\gps_data\images_10_08_21\1_350_exp_3200_iso_autofocus_out.csv'
    xleft_coord_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\gps_data\feature_coords\10_08_21\left_x.csv'
    yleft_coord_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\gps_data\feature_coords\10_08_21\left_y.csv'
    xright_coord_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\gps_data\feature_coords\10_08_21\right_x.csv'
    yright_coord_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\gps_data\feature_coords\10_08_21\right_y.csv'
    light_gis_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\gps_data\ground_truth_gis_lights.csv'

    left_img_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\images_8_12_21\Left\left0.jpg'
    right_img_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\images_8_12_21\Right\right0.jpg'

    left_img = open_image(left_img_path)
    right_img = open_image(right_img_path)

    fx_left = 3283.87648 
    fy_left = 3299.03024
    cx_left = 2030.36278
    cy_left = 1550.33756
    
    K1 = np.array([fx_left, 0, cx_left, 0, fy_left, cy_left, 0, 0, 1]).reshape(3,3)

    fx_right = 3269.07563
    fy_right = 3276.23450
    cx_right = 2086.51620
    cy_right = 1587.00571

    K2 = np.array([fx_right, 0, cx_right, 0, fy_right, cy_right, 0, 0, 1]).reshape(3,3)

    #angle of right camera
    omega = np.array([0.00659, -0.01284, -0.02433]).reshape(-1,1)

    # print(Deparameterize_Omega(omega))

    t1 = np.array([0, 0, 0]).reshape(-1,1)
    t12 = np.array([-0.33418658, 0.00541115, -0.00189281]).reshape(-1,1)
    # t2 = np.array([-0.353, 0.00541115, -0.00189281]).reshape(-1,1)


    R1 = np.eye(3)
    R12 = Deparameterize_Omega(omega)

    
    geo_centric_detic_df = pd.read_csv(geo_centric_detic_path)
    name_list = geo_centric_detic_df.Base_Name.tolist()
    yaw_list = geo_centric_detic_df.Yaw_X.tolist()
    pitch_list = geo_centric_detic_df.Pitch_Y.tolist()
    roll_list = geo_centric_detic_df.Roll_Z.tolist()
    geo_x_list = geo_centric_detic_df.Geocentric_X.tolist()
    geo_y_list = geo_centric_detic_df.Geocentric_Y.tolist()
    geo_z_list = geo_centric_detic_df.Geocentric_Z.tolist()
    lat_list = geo_centric_detic_df.Lat.tolist()
    long_list = geo_centric_detic_df.Long.tolist()

    xleft_df = pd.read_csv(xleft_coord_path)
    yleft_df = pd.read_csv(yleft_coord_path)

    xright_df = pd.read_csv(xright_coord_path)
    yright_df = pd.read_csv(yright_coord_path)

    gis_df = pd.read_csv(light_gis_path)

    for index in range(len(param_name_list)):

        print(param_name_list[index])

        column_name_list = list() # name of column (e.g. image0_light1)
        column_list = list() # error list associated with column

        increment_val = increment_val_list[index]
        value_list = np.linspace(-increment_val*number_increments, increment_val*number_increments, number_increments*2+1)

        column_name_list.append('Increment_Values')
        column_list.append(value_list)

        # also need to cycle through lights 1-4
        for i in range(len(name_list)):
            # left = 'left_' + name_list[i]
            # right = 'right_' + name_list[i]

            # cycling through lights 1-4
            for j in range(1,5):

                column_name = name_list[i] + f'light{j}'

                xleft_temp = xleft_df.loc[xleft_df['Base_Name'] == name_list[i]]
                yleft_temp = yleft_df.loc[yleft_df['Base_Name'] == name_list[i]]

                xright_temp = xright_df.loc[xright_df['Base_Name'] == name_list[i]]
                yright_temp = yright_df.loc[yright_df['Base_Name'] == name_list[i]]

                # print(xleft_temp)

                xleft_coord = xleft_temp.iloc[0][j]
                yleft_coord = yleft_temp.iloc[0][j]

                xright_coord = xright_temp.iloc[0][j]
                yright_coord = yright_temp.iloc[0][j]

                if xleft_coord == 0:
                    continue

                # x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, t1, K2, t2, omega

                x1 = np.array([xleft_coord, yleft_coord]).reshape(-1,1)
                x2 = np.array([xright_coord, yright_coord]).reshape(-1,1)

                C_origin = np.array([geo_x_list[i], geo_y_list[i], geo_z_list[i]]).reshape(-1,1)

                latitude_origin = lat_list[i]
                longitude_origin = long_list[i]

                plat_yaw = yaw_list[i]
                plat_pitch = pitch_list[i]
                plat_roll = roll_list[i]

                plat_yaw = plat_yaw + 13

                light_j = gis_df.loc[gis_df['Light_Number'] == j]

                geo_x = light_j.iloc[0]['Geocentric_X']
                geo_y = light_j.iloc[0]['Geocentric_Y']
                geo_z = light_j.iloc[0]['Geocentric_Z']

                light_geo_coord = np.array([geo_x, geo_y, geo_z]).reshape(-1,1)

                estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)

                param_error_list = list()

                for value in np.linspace(-increment_val*number_increments, increment_val*number_increments, number_increments*2+1):

                    if param_name_list[index] == 'fx_left':
                        new_K1 = np.copy(K1)
                        new_K1[0][0] = K1[0][0] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, new_K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)
                    
                    elif param_name_list[index] == 'fy_left':
                        new_K1 = np.copy(K1)
                        new_K1[1][1] = K1[1][1] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, new_K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'cx_left':
                        new_K1 = np.copy(K1)
                        new_K1[0][2] = K1[0][2] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, new_K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'cy_left':
                        new_K1 = np.copy(K1)
                        new_K1[1][2] = K1[1][2] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, new_K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)
                    
                    elif param_name_list[index] == 'fx_right':
                        new_K2 = np.copy(K2)
                        new_K2[0][0] = K2[0][0] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, new_K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'fy_right':
                        new_K2 = np.copy(K2)
                        new_K2[1][1] = K2[1][1] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, new_K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'cx_right':
                        new_K2 = np.copy(K2)
                        new_K2[0][2] = K2[0][2] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, new_K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'cy_right':
                        new_K2 = np.copy(K2)
                        new_K2[1][2] = K2[1][2] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, new_K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    # omega --> R12
                    elif param_name_list[index] == 'omega_x':
                        new_omega = np.copy(omega)
                        new_omega[0] = omega[0] + value
                        new_R12 = Deparameterize_Omega(new_omega)
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, new_R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'omega_y':
                        new_omega = np.copy(omega)
                        new_omega[1] = omega[1] + value
                        new_R12 = Deparameterize_Omega(new_omega)
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, new_R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'omega_z':
                        new_omega = np.copy(omega)
                        new_omega[2] = omega[2] + value
                        new_R12 = Deparameterize_Omega(new_omega)
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, new_R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)
                    
                    elif param_name_list[index] == 't12_x':
                        new_t12 = np.copy(t12)
                        new_t12[0] = t12[0] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, new_t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 't12_y':
                        new_t12 = np.copy(t12)
                        new_t12[1] = t12[1] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, new_t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 't12_z':
                        new_t12 = np.copy(t12)
                        new_t12[2] = t12[2] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, new_t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'img1_x':
                        new_x1 = np.copy(x1)
                        new_x1[0] = x1[0] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, new_x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'img1_y':
                        new_x1 = np.copy(x1)
                        new_x1[1] = x1[1] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, new_x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'img2_x':
                        new_x2 = np.copy(x2)
                        new_x2[0] = x2[0] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, new_x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'img2_y':
                        new_x2 = np.copy(x2)
                        new_x2[1] = x2[1] + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, new_x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'ori_x':
                        new_plat_pitch = plat_pitch + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, new_plat_pitch, plat_roll, K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'ori_y':
                        new_plat_yaw = plat_yaw + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, new_plat_yaw, plat_pitch, plat_roll, K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                    elif param_name_list[index] == 'ori_z':
                        new_plat_roll = plat_roll + value
                        new_estimated_geo_point = geocentric_triangulation2View(left_img, right_img, x1, x2, C_origin, latitude_origin, longitude_origin, plat_yaw, plat_pitch, new_plat_roll, K1, K2, R12, t12)
                        error = np.linalg.norm(light_geo_coord - new_estimated_geo_point)
                        param_error_list.append(error)

                column_name_list.append(column_name)
                column_list.append(param_error_list)

        param_error_df = pd.DataFrame(dict(zip(column_name_list, column_list)))

        param_error_df.to_csv(save_path + f'\{param_name_list[index]}.csv')

def table_error(saved_path, new_save_path):
    # lists
    param_name_list = list()
    avg_increment_value_list = list()
    med_increment_value_list = list()
    orig_avg_error_list = list()
    new_avg_error_list = list()
    percent_error_reduced_avg_list = list()
    orig_med_error_list = list()
    new_med_error_list = list()
    percent_error_reduced_med_list = list()

    file_list = glob.glob(saved_path + '\*.csv')
    for file_path in file_list:
        parsed_path = file_path.split('\\')
        param_name = parsed_path[len(parsed_path)-1].split('.')[0]

        print(param_name)

        param_df = pd.read_csv(file_path)

        rows_list = param_df.values.tolist()

        min_avg_error = 9999
        min_avg_error_index = None

        min_med_error = 9999
        min_med_error_index = None

        for i in range(len(rows_list)):
            increment_value_error_list = rows_list[i][2:]
            avg_error = np.mean(increment_value_error_list)
            med_error = np.mean(increment_value_error_list)

            if avg_error < min_avg_error:
                min_avg_error = avg_error
                min_avg_error_index = i

            if med_error < min_med_error:
                min_med_error = med_error
                min_med_error_index = i

        orig_error_list = rows_list[5][2:]
        new_error_list = rows_list[min_avg_error_index][2:]

        # *index values are hard coded which depends on how the CSV files were ordered*
        orig_avg_error = np.mean(orig_error_list)
        new_avg_error = np.mean(new_error_list)
        orig_med_error = np.median(orig_error_list)
        new_med_error = np.median(new_error_list)

        param_name_list.append(param_name)

        # if orig_avg_error - new_avg_error  < 0.0001:
        #     avg_increment_value_list.append(0)
        # else:
        avg_increment_value_list.append(rows_list[min_avg_error_index][1])

        # if orig_med_error - new_med_error < 0.0001:
        #     med_increment_value_list.append(0)
        # else:
        med_increment_value_list.append(rows_list[min_med_error_index][1])

        orig_avg_error_list.append(orig_avg_error)
        new_avg_error_list.append(new_avg_error)
        percent_error_reduced_avg_list.append((orig_avg_error - new_avg_error) / orig_avg_error * 100)
        orig_med_error_list.append(orig_med_error)
        new_med_error_list.append(new_med_error)
        percent_error_reduced_med_list.append((orig_med_error - new_med_error) / orig_med_error * 100)

    column_names_avg = list(['Param_Name', 'Parameter_Delta_value', 'Original_Average_Error', 'New_Average_Error', 'Percent_Error_Reduced_Avg'])

    column_names_med = list(['Param_Name', 'Parameter_Delta_value', 'Original_Median_Error', 'New_Median_Error', 'Percent_Error_Reduced_Med'])

    column_lists_avg = list()
    column_lists_avg.append(param_name_list)
    column_lists_avg.append(avg_increment_value_list)
    column_lists_avg.append(orig_avg_error_list)
    column_lists_avg.append(new_avg_error_list)
    column_lists_avg.append(percent_error_reduced_avg_list)

    column_lists_med = list()
    column_lists_med.append(param_name_list)
    column_lists_med.append(med_increment_value_list)
    column_lists_med.append(orig_med_error_list)
    column_lists_med.append(new_med_error_list)
    column_lists_med.append(percent_error_reduced_med_list)

    avg_error_reduction_df = pd.DataFrame(dict(zip(column_names_avg, column_lists_avg)))
    avg_error_reduction_df.to_csv(new_save_path + '\\minimize_avg_error_params.csv')

    med_error_reduction_df = pd.DataFrame(dict(zip(column_names_med, column_lists_med)))
    med_error_reduction_df.to_csv(new_save_path + '\\minimize_med_error_params.csv')

    
if __name__ == '__main__':
    number_increments = 5
    quant_params_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\param_error\quant_params.csv'
    save_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\param_error\param_error_results'
    new_save_path = r'C:\Users\alexy\Dropbox\STIMA\scripts\STIMA\Location Analysis\GPS_estimation\param_error'

    # error_quant_analysis(number_increments, quant_params_path, save_path)   
    table_error(save_path, new_save_path)
