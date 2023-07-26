#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "C:\Program Files (x86)\Intel RealSense SDK 2.0\samples\example-imgui.hpp"          // Include short list of convenience functions for rendering


enum class snapshot
{
    to_wait,
    to_capture
};
// Forward definition of UI rendering, implemented below
void render_button(rect location, snapshot* dir);

int main() try
{
    //using namespace cv;
    using namespace rs2;

    // Create and initialize GUI related objects
    window app(2560, 720, "RealSense Align Example"); // Simple window handling
    ImGui_ImplGlfw_Init(app, false); 
    texture l1_image, l2_image;



    rs2::pipeline p;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 1280, 720, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 1280, 720, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    p.start(cfg);
    // rs2::frameset frames = p.wait_for_frames();

    //auto ir_frame_left = frames.get_infrared_frame(1);
    //auto ir_frame_right = frames.get_infrared_frame(2);
    //auto depth = frames.get_depth_frame();
    //auto colored_frame = frames.get_color_frame();


    //New Code

    snapshot snap = snapshot::to_wait;
    while (app)
    {
        rs2::frameset frames = p.wait_for_frames();
         
        auto ir_frame_left = frames.get_infrared_frame(1); 
        auto ir_frame_right = frames.get_infrared_frame(2); 
        auto depth = frames.get_depth_frame(); 
        auto colored_frame = frames.get_color_frame(); 

        l1_image.render(ir_frame_left, { 0, 0, 1280, app.height() });
        l2_image.render(ir_frame_right, { 1280, 0, 1280, app.height() });

        if (snap == snapshot::to_capture) {
            cv::Mat dMat_left = cv::Mat(cv::Size(1280, 720), CV_8UC1, (void*)ir_frame_left.get_data()); 
            cv::Mat dMat_right = cv::Mat(cv::Size(1280, 720), CV_8UC1, (void*)ir_frame_right.get_data()); 
            cv::Mat dMat_colored = cv::Mat(cv::Size(640, 480), CV_8UC1, (void*)colored_frame.get_data()); 

            cv::imwrite("C:/Users/aquir/Documents/Images/irLeft0.png", dMat_left); 
            cv::imwrite("C:/Users/aquir/Documents/Images/irRight0.png", dMat_right);  
            cv::imwrite("C:/Users/aquir/Documents/Images/coloredCV0.png", dMat_colored); 
            break;
        }
        // Render the UI:
        ImGui_ImplGlfw_NewFrame(1); 
        //render_slider({ 15.f, app.height() - 60, app.width() - 30, app.height() }, &alpha, &dir);
        render_button({ 15.f, app.height() - 60, app.width() - 30, app.height() }, &snap); 
        ImGui::Render(); 
    }
    //

    ////save files
    //cv::Mat dMat_left = cv::Mat(cv::Size(1280, 720), CV_8UC1, (void*)ir_frame_left.get_data());
    //cv::Mat dMat_right = cv::Mat(cv::Size(1280, 720), CV_8UC1, (void*)ir_frame_right.get_data());
    //cv::Mat dMat_colored = cv::Mat(cv::Size(640, 480), CV_8UC1, (void*)colored_frame.get_data());


    //std::cout << "Hello" << std::endl;
    ////std::cout << dMat_right << std::endl;

    //cv::imwrite("C:/Users/aquir/Documents/Images/irLeft0.png", dMat_left);
    //cv::imwrite("C:/Users/aquir/Documents/Images/irRight0.png", dMat_right);
    //cv::imwrite("C:/Users/aquir/Documents/Images/coloredCV0.png", dMat_colored);
    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

void render_button(rect location, snapshot* snap)
{
    static const int flags = ImGuiWindowFlags_NoCollapse 
        | ImGuiWindowFlags_NoScrollbar 
        | ImGuiWindowFlags_NoSavedSettings 
        | ImGuiWindowFlags_NoTitleBar 
        | ImGuiWindowFlags_NoResize 
        | ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos({ location.x, location.y }); 
    ImGui::SetNextWindowSize({ location.w, location.h }); 

    ImGui::Begin("To Capture Image", nullptr, flags);
    ImGui::PushItemWidth(-1); 
    // Render direction checkboxes:
    bool to_wait = (*snap == snapshot::to_wait);
    bool to_capture = (*snap == snapshot::to_capture);

    if (ImGui::Checkbox("Wait for Image", &to_wait))
    { 
        *snap = to_wait ? snapshot::to_wait : snapshot::to_capture; 
    }
    ImGui::SameLine();
    ImGui::SetCursorPosX(location.w - 140);
    if (ImGui::Checkbox("Take Image", &to_capture))
    {
        *snap = to_capture ? snapshot::to_capture : snapshot::to_wait;
    }

    ImGui::End(); 
}