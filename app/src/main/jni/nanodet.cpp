// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"
#include "iostream"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

static const std::vector<cv::Scalar> colors={cv::Scalar(139,0,0),cv::Scalar(255,0,0),cv::Scalar(255,69,0),cv::Scalar(255,127,0),cv::Scalar(255,165,0),cv::Scalar(152,251,152),cv::Scalar(124,252,0),cv::Scalar(102,205,170),cv::Scalar(0,255,255),cv::Scalar(0,191,255),cv::Scalar(30,144,255),cv::Scalar(0,0,255),cv::Scalar(132,112,255),cv::Scalar(123,104,238),cv::Scalar(106,90,205)};
static const std::vector<std::vector<int>> PART_LINE={{0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {17, 11}, {17, 12},{11, 13}, {12, 14}, {13, 15}, {14, 16}};
const int OUTPUT_H=64;
const int OUTPUT_W=48;
const float CONF_THRESH=0.4;
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    nanodet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodet.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodet.opt.use_vulkan_compute = use_gpu;
#endif

    nanodet.opt.num_threads = ncnn::get_big_cpu_count();
    nanodet.opt.blob_allocator = &blob_pool_allocator;
    nanodet.opt.workspace_allocator = &workspace_pool_allocator;

//    char parampath[256];
//    char modelpath[256];
//    sprintf(parampath, "nanodet-%s.param", modeltype);
//    sprintf(modelpath, "nanodet-%s.bin", modeltype);
//
//    nanodet.load_param(parampath);
//    nanodet.load_model(modelpath);
//
//    target_size = _target_size;
//    mean_vals[0] = _mean_vals[0];
//    mean_vals[1] = _mean_vals[1];
//    mean_vals[2] = _mean_vals[2];
//    norm_vals[0] = _norm_vals[0];
//    norm_vals[1] = _norm_vals[1];
//    norm_vals[2] = _norm_vals[2];

    nanodet.load_param("hrnet32_256x192-opt-fp16.param");
    nanodet.load_model("hrnet32_256x192-opt-fp16.bin");

    target_size = 256;
    mean_vals[0] = 123.675f;
    mean_vals[1] = 116.28f;
    mean_vals[2] = 103.53f;
    norm_vals[0] = 1.f / 58.395f;
    norm_vals[1] = 1.f / 57.12f;
    norm_vals[2] = 1.f / 57.375f;

    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    nanodet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodet.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodet.opt.use_vulkan_compute = use_gpu;
#endif

    nanodet.opt.num_threads = ncnn::get_big_cpu_count();
    nanodet.opt.blob_allocator = &blob_pool_allocator;
    nanodet.opt.workspace_allocator = &workspace_pool_allocator;

//    char parampath[256];
//    char modelpath[256];
//    sprintf(parampath, "nanodet-%s.param", modeltype);
//    sprintf(modelpath, "nanodet-%s.bin", modeltype);

//    nanodet.load_param(mgr, parampath);
//    nanodet.load_model(mgr, modelpath);

    nanodet.load_param(mgr, "hrnet32_256x192-opt-fp16.param");
    nanodet.load_model(mgr, "hrnet32_256x192-opt-fp16.bin");

//    target_size = _target_size;
//    mean_vals[0] = _mean_vals[0];
//    mean_vals[1] = _mean_vals[1];
//    mean_vals[2] = _mean_vals[2];
//    norm_vals[0] = _norm_vals[0];
//    norm_vals[1] = _norm_vals[1];
//    norm_vals[2] = _norm_vals[2];

    target_size = 256;
    mean_vals[0] = 123.675f;
    mean_vals[1] = 116.28f;
    mean_vals[2] = 103.53f;
    norm_vals[0] = 1.f / 58.395f;
    norm_vals[1] = 1.f / 57.12f;
    norm_vals[2] = 1.f / 57.375f;

    return 0;
}
void get_realpoint(cv::Rect rect, std::vector<cv::Point> &hint_poses) {
    int l, t;
    float r_w = OUTPUT_W / (rect.width * 1.0);
    float r_h = OUTPUT_H / (rect.height * 1.0);


    if (r_h > r_w) {
        for (int i = 0; i < hint_poses.size(); ++i) {
            l = hint_poses[i].x;
            t = hint_poses[i].y - (OUTPUT_H - r_w * rect.height) / 2;
            hint_poses[i].x = l / r_w;
            hint_poses[i].y = t / r_w;
        }
    }
    else {
        for (int i = 0; i < hint_poses.size(); ++i) {
            l = hint_poses[i].x  - (OUTPUT_W - r_h * rect.width) / 2;
            t = hint_poses[i].y ;
            hint_poses[i].x = l / r_h;
            hint_poses[i].y = t / r_h;
        }
    }
}
int NanoDet::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if ((float)w/(float)h >= 192.f/256.f)
    {
        scale = (float)192 / w;
        w = 192;
        h = h * scale;
    }
    else
    {
        scale = (float)256 / h;
        h = 256;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    int wpad = 192-w;
    int hpad = 256-h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = nanodet.create_extractor();
//    ex.set_light_mode(true);
//    ex.set_num_threads(4);

    ex.input("input.1", in_pad);


    ncnn::Mat point_pred;
    ex.extract("2873", point_pred);

    std::vector<cv::Point> hint_poses;
    std::vector<float> points_conf;
    for (int part_i=0; part_i<point_pred.c; part_i++)
    {
        float* ptr = point_pred.channel(part_i);
        int maxpos = std::max_element(ptr, ptr+OUTPUT_H*OUTPUT_W)-ptr;

        points_conf.push_back(ptr[maxpos]);
        hint_poses.push_back(cv::Point(maxpos%OUTPUT_W,maxpos/OUTPUT_W));
    }
    points_conf.push_back((points_conf[5]+points_conf[6])/2.0);
    hint_poses.push_back(cv::Point((hint_poses[5].x+hint_poses[6].x)/2,(hint_poses[5].y+hint_poses[6].y)/2));
    cv::Rect rect(0,0,width, height);
    get_realpoint(rect, hint_poses);


    for (int part_i = 0; part_i < 17+1; part_i++) {
        if (points_conf[part_i]>CONF_THRESH){
            cv::circle(rgb, hint_poses[part_i],4, cv::Scalar(255,255,255),-1,0);
        }
    }
    for (int line_i = 0; line_i < PART_LINE.size(); line_i++) {
        if (points_conf[PART_LINE[line_i][0]]>CONF_THRESH && points_conf[PART_LINE[line_i][1]]>CONF_THRESH){
            cv::line(rgb, hint_poses[PART_LINE[line_i][0]], hint_poses[PART_LINE[line_i][1]], colors[line_i] , 2, 0);
        }
    }

    return 0;
}

int NanoDet::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{


    return 0;
}
