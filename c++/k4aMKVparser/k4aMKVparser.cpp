
#include <iostream>
#include <fstream> 
#include <cstdio>
#include <string>
#include <cassert>
// Azurekienct SDK
#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
// OpenCV
#include<opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
// Json file
#include "json/json.hpp"

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
    float* xy_table;

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
            genXYtable();
        }
        catch (...) {
            std::cout<<"Cannot open file : "<< filename <<std::endl;
        }        
    };
    ~MKVParser() {
        device_handle.close();
        free(xy_table);
    };
    std::string getDepthfilename(std::string prefix, int index) {
        return string_sprintf(prefix.c_str(), index, ".depth.png");
    }
    std::string getRGBfilename(std::string prefix, int index) {
        return string_sprintf(prefix.c_str(), index, ".color.png");
    }
    void genXYtable() {
        xy_table = (float*)calloc(w * h * 2, sizeof(float));
        k4a_float2_t p;
        k4a_float3_t ray;
        int valid;
        int height = h;
        int width = w;
        for (int y = 0; y < height; y++)
        {
            p.xy.y = (float)y;
            for (int x = 0; x < width; x++)
            {
                int idx = (height - 1 - y) * width + x;
                p.xy.x = (float)x;

                k4a_calibration_2d_to_3d(
                    &calibration, &p, 1.f, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &ray, &valid);

                if (valid)
                {
                    xy_table[idx * 2] = ray.xyz.x;
                    xy_table[idx * 2 + 1] = ray.xyz.y;
                }
                else
                {
                    xy_table[idx * 2] = 0;
                    xy_table[idx * 2 + 1] = 0;
                }
            }
        }
    }
    void exportData(std::string folder) {        
        if (!cv::utils::fs::isDirectory(folder)) {
            cv::utils::fs::createDirectories(folder);
            std::vector<nlohmann::json> frames = getAlignedRGBD(folder, "%08d%s");
            getConfig(folder, frames);
        }
        else {
            //check images create success
            std::cout << std::endl << "Faild to export data, folder already exists.";
        }
    }
    void getConfig(std::string folder, std::vector<nlohmann::json> frames) {
        std::string config = cv::utils::fs::join(folder, "config.json");
        std:: vector<float> xy_table(xy_table, xy_table + w * h * 2);
        nlohmann::json out;
        nlohmann::json intrinsic;

        out["mapping_2d_to_3d_table"] = xy_table;
        out["frames"] = frames;
        out["width"] = w;
        out["height"] = h;

        intrinsic["cx"] = calibration.color_camera_calibration.intrinsics.parameters.param.cx;
        intrinsic["cy"] = calibration.color_camera_calibration.intrinsics.parameters.param.cy;
        intrinsic["fx"] = calibration.color_camera_calibration.intrinsics.parameters.param.fx;
        intrinsic["fy"] = calibration.color_camera_calibration.intrinsics.parameters.param.fy;
        intrinsic["codx"] = calibration.color_camera_calibration.intrinsics.parameters.param.codx;
        intrinsic["cody"] = calibration.color_camera_calibration.intrinsics.parameters.param.cody;
        intrinsic["k1"] = calibration.color_camera_calibration.intrinsics.parameters.param.k1;
        intrinsic["k2"] = calibration.color_camera_calibration.intrinsics.parameters.param.k2;
        intrinsic["k3"] = calibration.color_camera_calibration.intrinsics.parameters.param.k3;
        intrinsic["k4"] = calibration.color_camera_calibration.intrinsics.parameters.param.k4;
        intrinsic["k5"] = calibration.color_camera_calibration.intrinsics.parameters.param.k5;
        intrinsic["k6"] = calibration.color_camera_calibration.intrinsics.parameters.param.k6;
        intrinsic["metric_radius"] = calibration.color_camera_calibration.intrinsics.parameters.param.metric_radius;
        intrinsic["p1"] = calibration.color_camera_calibration.intrinsics.parameters.param.p1;
        intrinsic["p2"] = calibration.color_camera_calibration.intrinsics.parameters.param.p2;
        out["intrinsic"] = intrinsic;

        std::ofstream o(config);
        o << std::setw(4) << out << std::endl;
        o.close();
    }
    std::vector<nlohmann::json> getAlignedRGBD(std::string folder,std::string prefix) {
        int index = 0;
        k4a::capture capture = NULL;
        std::vector<nlohmann::json> frames;

		while (true)
		{   
            try {
                if (device_handle.get_next_capture(&capture)==false) {
                    break;
                }
            }
            catch (...) {
                continue;
            }

            nlohmann::json info;

            index++;
            k4a::image color_image = capture.get_color_image();
            if (color_image) {
                cv::Mat image(cv::Size(w, h), CV_8UC4, (void*)color_image.get_buffer(), cv::Mat::AUTO_STEP);
                std::string filename = getRGBfilename(prefix, index);
                info["color"] = filename;
                std::string path = cv::utils::fs::join(folder, filename);
                cv::imwrite(path, image);
			}

            k4a::image depth_image = capture.get_depth_image();
			if (depth_image)
			{
				transformation.depth_image_to_color_camera(depth_image, &transformed_depth_image);
                cv::Mat image(cv::Size(w, h), CV_16UC1, (uint16_t*)transformed_depth_image.get_buffer(), cv::Mat::AUTO_STEP);
                std::string filename = getDepthfilename(prefix, index);
                info["depth"] = filename;
                std::string path = cv::utils::fs::join(folder, filename);
                cv::imwrite(path, image);
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
            frames.push_back(info);
		}
        std::cout << std::endl;
        return frames;
    }

};

int main(int argc, char** argv)
{
    if (argc == 3) {
        MKVParser file(argv[1]);
        file.exportData(argv[2]);
    }
    else {
        std::cout << "Params is Required : k4aMKVparser.exe <input.mkv> <folder>" << std::endl;
        std::cout << "params example : " << std::endl;
        std::cout << "<input.mkv> : \"D:/nerf-data/20121202/8cam/6/device0.mkv\"" << std::endl;
        std::cout << "<folder> : ./output" << std::endl;
    }

    // example
    //MKVParser file(R"(D:\nerf-data\20121202\8cam\6\device0.mkv)");
    //file.getAlignedRGBD("%08d");
}