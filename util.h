//
// Created by Starry Night on 2021/8/4.
//

#ifndef POINT_CLOUD__UTIL_H
#define POINT_CLOUD__UTIL_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iostream>
#include <json/json.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;
using namespace Eigen;

// Type redefinitions
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

#define THRESH_DIST 2000 // Threshold for eliminating distant points

// Camera params
const double cam_factor = 1000;
const double cam_fx = 727.998979;
const double cam_fy = 712.603153;
const double cam_cx = 644.078869;
const double cam_cy = 416.644178;

class Config {
  public:
    String rgb_dir;
    String depth_dir;
    String mask_dir;
    String mask_file;
    String output_dir;
    bool valid = false;
};

// Detection object class containing its geometric information
class Object {
  public:
    String img_id;     // Mask image file index of object
    String id;         // Index of object
    String label;      // Object label
    Vector3f dir;      // Orientation
    Vector4f centroid; // 3D centroid
    vector<int> bbox;  // x and y coords of left-upper corner, width and height
    double conf;       // Confidence value

    Object() { bbox.resize(4); }
};

// Load configurations
void load_config(Config &config);

// Read object list and prep for tasks
void get_obj_list(String mask_file, vector<Object> &list);

// Get hisogram of a single-channel image
void get_histogram(Mat src, vector<float> &hist);

// Operate erode on mask image
void erode_mask(Mat mask, Mat &dst, int ksize);

// Save point cloud into disk (in .ply format)
void save_point_cloud(PointCloud::Ptr cloud, string filename = "point_cloud");

// Add x, y and z axes to point cloud for reference
void add_axes(PointCloud::Ptr cloud);

// Get point cloud from a pair of RGB and D images
void get_point_cloud(Mat rgb, Mat depth, Mat mask, PointCloud::Ptr cloud,
                     bool addAxes = false);

// Remove outliers in point cloud
void pcl_remove_outlier(PointCloud::Ptr cloud_in, PointCloud::Ptr cloud_out,
                        int nr_k, double stddev_mult);

// Visualize point cloud
void get_cloud_visual(PointCloud::Ptr cloud);

// Write result to JSON file
void write_result(String output_dir, vector<Object> obj_list);

#endif // POINT_CLOUD__UTIL_H