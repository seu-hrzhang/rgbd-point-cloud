//
// Created by Starry Night on 2021/8/4.
//

#include "util.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void load_config(Config &config) {
    ifstream file("../config/path.json", ios::binary);
    if (!file.is_open()) {
        cout << "Error: could not load configuration file." << endl;
        return;
    }

    Json::Reader reader;
    Json::Value root;
    if (reader.parse(file, root)) {
        config.rgb_dir = root["rgb_dir"].asString();
        config.depth_dir = root["depth_dir"].asString();
        config.mask_dir = root["mask_dir"].asString();
        config.mask_file = root["mask_file"].asString();
        config.output_dir = root["output_dir"].asString();
    }
    config.valid = !(config.rgb_dir.empty() || config.depth_dir.empty() ||
                     config.mask_dir.empty() || config.mask_file.empty() ||
                     config.output_dir.empty());
}

void get_obj_list(String mask_file, vector<Object> &obj_list) {
    obj_list.clear();
    ifstream file(mask_file, ios::in);
    Object obj;
    int x, y, w, h;
    while (file >> obj.img_id >> obj.id >> obj.label >> obj.bbox[0] >>
           obj.bbox[1] >> obj.bbox[2] >> obj.bbox[3] >> obj.conf)
        obj_list.push_back(obj);
}

void get_histogram(Mat src, vector<float> &hist) {
    // Initialize hisogram vector
    if (src.type() == CV_8UC1)
        hist.resize(256);
    else if (src.type() == CV_16UC1)
        hist.resize(65536);
    else {
        cout << "Input image in wrong format." << endl;
        return;
    }

    if (src.type() == CV_8UC1) {
        for (int i = 0; i < src.rows; ++i)
            for (int j = 0; j < src.cols; ++j)
                hist[src.at<ushort>(i, j)]++;
    } else {
        for (int i = 0; i < src.rows; ++i)
            for (int j = 0; j < src.cols; ++j)
                hist[src.at<uchar>(i, j)]++;
    }
}

void erode_mask(Mat mask, Mat &dst, int ksize) {
    Mat element = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
    erode(mask, dst, element);
}

void save_point_cloud(PointCloud::Ptr cloud, string filename) {
    try {
        pcl::io::savePLYFile("../src/" + filename + ".ply", *cloud);
        cout << "PLY filed saved to \'../src/" + filename + ".ply\'" << endl;
    } catch (pcl::IOException &e) {
        cout << e.what() << endl;
    }
}

void add_axes(PointCloud::Ptr cloud) {
    for (int i = 0; i < 100; ++i) {
        PointT p_x((double)i / 100, 0, 0);
        PointT p_y(0, (double)i / 100, 0);
        PointT p_z(0, 0, (double)i / 100);
        p_x.r = 255;
        p_x.g = 0;
        p_x.b = 0;
        p_y.r = 0;
        p_y.g = 255;
        p_y.b = 0;
        p_z.r = 0;
        p_z.g = 0;
        p_z.b = 255;
        cloud->points.push_back(p_x);
        cloud->points.push_back(p_y);
        cloud->points.push_back(p_z);
    }
}

void get_point_cloud(Mat rgb, Mat depth, Mat mask, PointCloud::Ptr cloud,
                     bool addAxes) {
    cout << "Generating point cloud..." << endl;
    cout << "Total points: " << rgb.rows * rgb.cols << endl;
    for (int i = 0; i < depth.rows; ++i) {
        for (int j = 0; j < depth.cols; ++j) {
            if (mask.at<uchar>(i, j) == 0)
                continue;
            ushort pxl = depth.at<ushort>(i, j);
            // if (pxl != 0 && abs(pxl - 65536) > 1000) {
            if (pxl != 0 && pxl < THRESH_DIST) {
                // debug
                // cout << "pxl = " << pxl << endl;
                PointT p;
                p.z = double(pxl) / cam_factor;
                p.x = (j - cam_cx) * p.z / cam_fx;
                p.y = (i - cam_cy) * p.z / cam_fy;
                p.b = rgb.at<Vec3b>(i, j)[0];
                p.g = rgb.at<Vec3b>(i, j)[1];
                p.r = rgb.at<Vec3b>(i, j)[2];

                cloud->points.push_back(p);
                // debug
                // cout << cloud->points.size() << endl;
            }
        }
    }
    cout << "Valid points: " << cloud->size() << endl;
    if (cloud->size() == 0) {
        cout << "Point cloud empty. Skipping..." << endl << endl;
        return;
    }
    if (addAxes)
        add_axes(cloud);
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
}

void pcl_remove_outlier(PointCloud::Ptr cloud_in, PointCloud::Ptr cloud_out,
                        int nr_k, double stddev_mult) {
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud_in);
    sor.setMeanK(nr_k);
    sor.setStddevMulThresh(stddev_mult);
    sor.filter(*cloud_out);
    cout << "Valid points (removed outliers): " << cloud_out->size() << endl;
}

void get_cloud_visual(PointCloud::Ptr cloud) {
    PointT max_pt, min_pt;
    Vector3f centroid_org;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    Vector3f shape;
    shape = max_pt.getVector3fMap() - min_pt.getVector3fMap();
    float scale = (shape(0) + shape(1) + shape(2)) / 3;

    pcl::visualization::PCLVisualizer viewer;
    // pcl::visualization::PointCloudColorHandlerCustom<PointT>
    // color_handler(cloud);
    // viewer.addPointCloud(cloud, color_handler, "cloud");
    viewer.addPointCloud(cloud, "cloud");

    viewer.addCoordinateSystem(0.5f * scale);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void write_result(String output_dir, vector<Object> obj_list) {
    Json::Value root;

    root["code"] = Json::Value(0);
    root["msg"] = Json::Value("ok");

    Json::Value result;
    Json::Value multi_obj;
    String proc_id;

    Json::Value id;
    Json::Value props;
    Json::Value dir;
    Json::Value pos;
    Json::Value bbox;
    Json::Value conf;

    for (int i = 0; i < obj_list.size(); ++i) {
        if (obj_list[i].img_id != proc_id) {
            multi_obj.clear();
        }
        proc_id = obj_list[i].img_id;

        id = Json::Value(obj_list[i].label);
        props.clear();
        dir.clear();
        pos.clear();
        bbox.clear();
        conf.clear();

        dir.append(obj_list[i].dir[0]);
        dir.append(obj_list[i].dir[1]);
        dir.append(obj_list[i].dir[2]);
        pos.append(obj_list[i].centroid[0]);
        pos.append(obj_list[i].centroid[1]);
        pos.append(obj_list[i].centroid[2]);
        bbox.append(obj_list[i].bbox[0]);
        bbox.append(obj_list[i].bbox[1]);
        bbox.append(obj_list[i].bbox[2]);
        bbox.append(obj_list[i].bbox[3]);
        conf.append(obj_list[i].conf);

        props["Orientation"].append(dir);
        props["Position"].append(pos);
        props["bbox"].append(bbox);
        props["confidence"].append(conf);
        props["ids"].append(0);

        multi_obj["ID"] = Json::Value(id);
        multi_obj["Properties"] = Json::Value(props);

        result["MultiObj"].append(multi_obj);

        root["result"] = Json::Value(result);

        if ((i < obj_list.size()) && (obj_list[i + 1].img_id != proc_id)) {
            Json::StyledWriter writer;
            ofstream file;
            file.open(output_dir + obj_list[i].img_id + ".json", ios::app);
            file << writer.write(root);
            file.close();
            result["MultiObj"].clear();
        }
    }
}