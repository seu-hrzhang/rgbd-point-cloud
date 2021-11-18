//
// Created by Starry Night on 2021/8/4.
//

#include "bbox.h"

using namespace std;
using namespace Eigen;

void get_cloud_params(PointCloud::Ptr cloud, Vector4f &centroid,
                      Matrix3f &eigenVecs, Vector3f &eigenVals) {
    pcl::compute3DCentroid(*cloud, centroid);
    Eigen::Matrix3f cov;
    pcl::computeCovarianceMatrixNormalized(*cloud, centroid, cov);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(
        cov, Eigen::ComputeEigenvectors);
    eigenVecs = eigen_solver.eigenvectors();
    eigenVals = eigen_solver.eigenvalues();

    // Ensure orthogonality across 3 directions
    eigenVecs.col(2) = eigenVecs.col(0).cross(eigenVecs.col(1));
    eigenVecs.col(0) = eigenVecs.col(1).cross(eigenVecs.col(2));
    eigenVecs.col(1) = eigenVecs.col(2).cross(eigenVecs.col(0));
}

void calc_transform_matrix(Vector4f centroid, Matrix3f eigenVecs,
                           Matrix4f &tm) {
    tm = Matrix4f::Identity();
    tm.block<3, 3>(0, 0) = eigenVecs.transpose();
    tm.block<3, 1>(0, 3) =
        -1.0f * (eigenVecs.transpose()) * (centroid.head<3>());
}

void get_bbox(PointCloud::Ptr cloud, Object &obj, bool getVisual) {
    // Get centroid, eigen vectors and eigen values using PCA
    Vector4f centroid;
    Matrix3f eigenVecs;
    Vector3f eigenVals;
    get_cloud_params(cloud, centroid, eigenVecs, eigenVals);

    // debug
    // cout << "PCA Centroid:" << endl << centroid << endl;
    // cout << "Eigen Vectors:" << endl << eigenVecs << endl;
    // cout << "Eigen Values:" << endl << eigenVals << endl;

    // Transform & inv-transform matrices
    Matrix4f tm = Matrix4f::Identity(), tm_inv = Matrix4f::Identity();
    calc_transform_matrix(centroid, eigenVecs, tm);
    tm_inv = tm.inverse();

    // Transform input point cloud to origin
    pcl::PointCloud<PointT>::Ptr cloud_org(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cloud, *cloud_org, tm);

    // Max and min point of transformed point cloud
    PointT max_pt_org, min_pt_org;
    Vector3f bbox_ctr_org;
    pcl::getMinMax3D(*cloud_org, min_pt_org, max_pt_org);

    // bbox_ctr = (min + max) / 2
    bbox_ctr_org =
        0.5f * (min_pt_org.getVector3fMap() + max_pt_org.getVector3fMap());

    Eigen::Affine3f aff_inv(tm_inv); // Inv-affine transformation
    Vector3f bbox_ctr;
    pcl::transformPoint(bbox_ctr_org, bbox_ctr, aff_inv);

    // debug
    // cout << "Bbox Center: " << endl << bbox_ctr_org << endl;

    /* Shape of transformed point cloud (width, height & depth)
     which is the same as initial cloud */
    Vector3f shape;
    shape = max_pt_org.getVector3fMap() - min_pt_org.getVector3fMap();
    // Average scale of point cloud
    float scale = (shape(0) + shape(1) + shape(2)) / 3;

    const Quaternionf bboxQ_org(Quaternionf::Identity());
    const Vector3f bboxT_org(bbox_ctr_org);

    const Quaternionf bboxQ(tm_inv.block<3, 3>(0, 0));
    const Vector3f bboxT(bbox_ctr);

    // Coordinate origin point (centroid of initial cloud, aka anchor)
    PointT anchor;
    anchor.x = centroid(0);
    anchor.y = centroid(1);
    anchor.z = centroid(2);
    // End points of x, y and z axis
    PointT arrow_x, arrow_y, arrow_z;
    // Get arrow shape of x axis
    arrow_x.x = scale * eigenVecs(0, 0) + anchor.x;
    arrow_x.y = scale * eigenVecs(1, 0) + anchor.y;
    arrow_x.z = scale * eigenVecs(2, 0) + anchor.z;
    // Get arrow shape of y axis
    arrow_y.x = scale * eigenVecs(0, 1) + anchor.x;
    arrow_y.y = scale * eigenVecs(1, 1) + anchor.y;
    arrow_y.z = scale * eigenVecs(2, 1) + anchor.z;
    // Get arrow shape of z axis
    arrow_z.x = scale * eigenVecs(0, 2) + anchor.x;
    arrow_z.y = scale * eigenVecs(1, 2) + anchor.y;
    arrow_z.z = scale * eigenVecs(2, 2) + anchor.z;

    /* Write params to object */
    // Write centroid
    obj.centroid = centroid;
    // Find main orientation
    Mat cvVals;
    eigen2cv(eigenVals, cvVals);
    double maxVal;
    Point maxLoc;
    minMaxLoc(cvVals, NULL, &maxVal, NULL, &maxLoc);
    // Write orientation
    obj.dir = eigenVecs.col(maxLoc.y);

    // Visualization
    if (getVisual) {
        pcl::visualization::PCLVisualizer viewer;
        viewer.addPointCloud(cloud, "cloud");
        viewer.addCube(bboxT, bboxQ, shape(0), shape(1), shape(2), "bbox");
        viewer.setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "bbox");
        viewer.setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "bbox");

        viewer.addArrow(arrow_x, anchor, 1.0, 0.0, 0.0, false, "arrow_x");
        viewer.addArrow(arrow_y, anchor, 0.0, 1.0, 0.0, false, "arrow_y");
        viewer.addArrow(arrow_z, anchor, 0.0, 0.0, 1.0, false, "arrow_z");

        viewer.addCoordinateSystem(0.5f * scale);
        viewer.setBackgroundColor(1.0, 1.0, 1.0);
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    }
}