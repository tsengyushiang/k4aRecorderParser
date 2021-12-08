
#include <iostream>
#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <cstdio>
#include <string>
#include <cassert>
template< typename... Args >
std::string string_sprintf(const char* format, Args... args) {
    int length = std::snprintf(nullptr, 0, format, args...);
    assert(length >= 0);

    char* buf = new char[length + 1];
    std::snprintf(buf, length + 1, format, args...);

    std::string str(buf);
    delete[] buf;
    return std::move(str);
}

class MKVParser {

    k4a_record_configuration_t config;
    k4a::playback device_handle;
    k4a::capture capture;
    k4a::image transformed_depth_image;
    k4a::calibration calibration;
    k4a::transformation transformation;
    long long recording_length;
    std::string serial;
    std::string filename;
    int w, h;

public:

    MKVParser(std::string filename) {
        try {
            this->filename = filename;
            device_handle = k4a::playback::open(filename.c_str());
            config = device_handle.get_record_configuration();
            recording_length = device_handle.get_recording_length().count();
            calibration = device_handle.get_calibration();
            transformation = k4a::transformation(calibration);
            device_handle.get_tag("K4A_DEVICE_SERIAL_NUMBER", &(serial));
            device_handle.set_color_conversion(K4A_IMAGE_FORMAT_COLOR_BGRA32);
            float fx = calibration.color_camera_calibration.intrinsics.parameters.param.fx;
            float fy = calibration.color_camera_calibration.intrinsics.parameters.param.fy;
            float ppx = calibration.color_camera_calibration.intrinsics.parameters.param.cx;
            float ppy = calibration.color_camera_calibration.intrinsics.parameters.param.cy;
            uint32_t offset = config.start_timestamp_offset_usec;
            std::cout << filename << ",serial number : " << serial<< std::endl;

            transformed_depth_image = k4a::image::create(
                K4A_IMAGE_FORMAT_DEPTH16,
                calibration.color_camera_calibration.resolution_width,
                calibration.color_camera_calibration.resolution_height,
                calibration.color_camera_calibration.resolution_width * (int)sizeof(uint16_t));

            switch (config.color_resolution) {
                case K4A_COLOR_RESOLUTION_OFF:     /**< Color camera will be turned off with this setting */
                    w = 0; h = 0;	break;
                case K4A_COLOR_RESOLUTION_720P:    /**< 1280 * 720  16:9 */
                    w = 1280; h = 720; break;
                case K4A_COLOR_RESOLUTION_1080P:   /**< 1920 * 1080 16:9 */
                    w = 1920; h = 1080; break;
                case K4A_COLOR_RESOLUTION_1440P:   /**< 2560 * 1440 16:9 */
                    w = 2560; h = 1440; break;
                case K4A_COLOR_RESOLUTION_1536P:   /**< 2048 * 1536 4:3  */
                    w = 2048; h = 1536; break;
                case K4A_COLOR_RESOLUTION_2160P:   /**< 3840 * 2160 16:9 */
                    w = 3840; h = 2160; break;
            }
        }
        catch (...) {
            std::cout<<"Cannot open file : "<< filename <<std::endl;
        }        
    };
    ~MKVParser() {
    };
    void getAlignedRGBD(std::string prefix) {
        int index = 0;
        k4a::capture capture = NULL;
		while (device_handle.get_next_capture(&capture))
		{
            index++;
            k4a::image color_image = capture.get_color_image();
            if (color_image) {
                cv::Mat image(cv::Size(w, h), CV_8UC4, (void*)color_image.get_buffer(), cv::Mat::AUTO_STEP);
                cv::imwrite(string_sprintf(prefix.c_str(),index) +".color.png", image);
                if (index == 1 && !cv::utils::fs::exists(string_sprintf(prefix.c_str(), index) + ".color.png")){
                    //check images create success
                    std::cout << std::endl<<"Faild to save frame, folder not exists.";
                    break;
                }                
			}

            k4a::image depth_image = capture.get_depth_image();
			if (depth_image)
			{
				transformation.depth_image_to_color_camera(depth_image, &transformed_depth_image);
                cv::Mat image(cv::Size(w, h), CV_16UC1, (uint16_t*)transformed_depth_image.get_buffer(), cv::Mat::AUTO_STEP);
                cv::imwrite(string_sprintf(prefix.c_str(), index) + ".depth.png", image);
			}

            int barWidth = 30;
            std::cout << filename << "[";
            float progress = (float)color_image.get_device_timestamp().count() / recording_length;
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
		}
        std::cout << std::endl;
    }

};

int main(int argc, char** argv)
{
    if (argc == 3) {
        MKVParser file(argv[1]);
        file.getAlignedRGBD(argv[2]);
    }
    else {
        std::cout << "Params is Required : k4aMKVparser.exe <input.mkv> <saveingPrefix>" << std::endl;
        std::cout << "params example : " << std::endl;
        std::cout << "<input.mkv> : \"D:/nerf-data/20121202/8cam/6/device0.mkv\"" << std::endl;
        std::cout << "<saveingPrefix> : ./output/%08d" << std::endl;
    }

    // example
    //MKVParser file(R"(D:\nerf-data\20121202\8cam\6\device0.mkv)");
    //file.getAlignedRGBD("%08d");
}