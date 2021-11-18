//
// Created by Starry Night on 2021/8/4.
//

#ifndef POINT_CLOUD__BBOX_H
#define POINT_CLOUD__BBOX_H

#include "util.h"

using namespace std;
using namespace Eigen;

// Get spatial params (mass centroid, eigen vectors & eigen values) of input
// point cloud
void get_cloud_params(PointCloud::Ptr cloud, Vector4f &pcaCentroid,
                      Matrix3f &eigenVecs, Vector3f &eigenVals);

// Calculate transform matrix from initial position to origin
void calc_transform_matrix(Vector4f pcaCentroid, Matrix3f eigenVecs,
                           Matrix4f &tm);

// Operate bounding box acquisition and visualization
void get_bbox(PointCloud::Ptr cloud, Object &obj, bool getVisual = true);

#endif // POINT_CLOUD__BBOX_H