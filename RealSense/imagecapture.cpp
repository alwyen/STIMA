#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
//#include <iostream>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include "example.hpp"          // Include short list of convenience functions for rendering


int main() try
{
    //using namespace cv;
    using namespace rs2;

    rs2::pipeline p;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 1280, 720, RS2_FORMAT_Y8, 30); 
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 1280, 720, RS2_FORMAT_Y8, 30); 
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_BGR8, 30); 

    rs2::frameset frames = p.wait_for_frames();

    auto ir_frame_left = frames.get_infrared_frame(1);
    auto ir_frame_right = frames.get_infrared_frame(2);
    auto depth = frames.get_depth_frame();
    auto colored_frame = frames.get_color_frame();


    ////save files
    //cv::Mat dMat_left = cv::Mat(cv::Size(1280, 720), CV_8UC1, (void*)ir_frame_left.get_data());
    //cv::Mat dMat_right = cv::Mat(cv::Size(1280, 720), CV_8UC1, (void*)ir_frame_right.get_data());
    //cv::Mat dMat_colored = cv::Mat(cv::Size(1920, 1080), CV_8UC1, (void*)colored_frame.get_data());


    //std::cout << "Hello" << std::endl;
    //std::cout << dMat_right << std::endl;

    /*cv::imwrite("C:/Users/aquir/Documents/Images/irLeft0.png", dMat_left);
    cv::imwrite("C:/Users/aquir/Documents/Images/irRight0.png", dMat_right); 
    cv::imwrite("C:/Users/aquir/Documents/Images/coloredCV0.png", dMat_colored);*/
    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    //std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    //std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
