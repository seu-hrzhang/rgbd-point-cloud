#include "bbox.h"
#include "util.h"

using namespace std;
using namespace cv;

// Run point cloud construction, bbox detection and visualization on a single
// image
void proc_single_image(Config config, Object obj) {
    Mat rgb = imread(config.rgb_dir + obj.img_id + ".jpg", IMREAD_UNCHANGED);
    Mat depth =
        imread(config.depth_dir + obj.img_id + ".png", IMREAD_UNCHANGED);
    Mat mask = imread(config.mask_dir + obj.img_id + "-" + obj.id + ".jpg",
                      IMREAD_UNCHANGED);
    if (rgb.empty() || depth.empty() || mask.empty()) {
        cout << "Error: file not found." << endl;
        return;
    }

    PointCloud::Ptr cloud(new PointCloud), cloud_filt(new PointCloud);

    get_point_cloud(rgb, depth, mask, cloud, false);
    if (cloud->size() == 0)
        return;

    pcl_remove_outlier(cloud, cloud_filt, 500, 1);

    get_bbox(cloud_filt, obj, true);

    cloud->points.clear();
    cloud_filt->points.clear();
    cout << endl;
}

int main() {
    Config config;
    load_config(config);

    if (!config.valid) {
        cout << "Error: invalid configuration file." << endl;
        return -1;
    }

    vector<Object> obj_list;
    get_obj_list(config.mask_file, obj_list);

    int size = obj_list.size();
    PointCloud::Ptr cloud(new PointCloud), cloud_filt(new PointCloud);

    for (int i = 0; i < size; ++i) {
        cout << "Processing image " << obj_list[i].img_id << ".jpg" << endl;
        Mat rgb = imread(config.rgb_dir + obj_list[i].img_id + ".jpg",
                         IMREAD_UNCHANGED);
        Mat depth = imread(config.depth_dir + obj_list[i].img_id + ".png",
                           IMREAD_UNCHANGED);
        Mat mask = imread(config.mask_dir + obj_list[i].img_id + "-" +
                              obj_list[i].id + ".jpg",
                          IMREAD_UNCHANGED);
        if (rgb.empty() || depth.empty() || mask.empty()) {
            cout << "Error: file not found." << endl;
            return -1;
        }
        // erode_mask(mask, mask, 10);

        // PointCloud::Ptr cloud(new PointCloud), cloud_filt(new PointCloud);

        get_point_cloud(rgb, depth, mask, cloud, false);
        if (cloud->size() == 0) {
            obj_list.erase(obj_list.begin() + i--);
            size--;
            continue;
        }
        pcl_remove_outlier(cloud, cloud_filt, 500, 1);

        // save_point_cloud(cloud);
        // save_point_cloud(cloud_filt, "point_cloud_filt");

        // get_cloud_visual(cloud_filt);

        get_bbox(cloud_filt, obj_list[i], false);

        cloud->points.clear();
        cloud_filt->points.clear();
        cout << endl;
    }
    write_result(config.output_dir, obj_list);

    return 0;
}