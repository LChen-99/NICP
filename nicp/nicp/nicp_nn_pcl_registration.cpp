/*
 * @Author: LuoChen 1425523063@qq.com
 * @Date: 2023-02-21 10:52:16
 * @LastEditors: LuoChen 1425523063@qq.com
 * @LastEditTime: 2023-02-23 14:42:25
 * @FilePath: /nicp/nicp/nicp/nicp_nn_pcl_registration.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
#include <pcl/common/transforms.h>
#include <opencv2/highgui/highgui.hpp>
#include <thread>
#include <chrono>
#include <nicp/imageutils.h>
#include <nicp/pinholepointprojector.h>
#include <nicp/depthimageconverterintegralimage.h>
#include <nicp/statscalculatorintegralimage.h>
#include <nicp/alignernn.h>
#include <nicp/pclcloudconverter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <nicp/aligner2d.h>
int main(int argc, char **argv) {
  // Print usage
  if(argc < 3) {
    std::cout << "USAGE: ";
    std::cout << "nicp_registration referencecloudname currentcloudname" << std::endl;
    std::cout << "   referencecloudname [string] \t-->\t input reference pcd file" << std::endl;
    std::cout << "   currentcloudname [string] \t-->\t input pcd file to register" << std::endl;
    
    return 0;
  }
  
 
  nicp::PointInformationMatrixCalculator2d pointInformationMatrixCalculator;
  nicp::NormalInformationMatrixCalculator2d normalInformationMatrixCalculator;



  pointInformationMatrixCalculator.setCurvatureThreshold(0.02f);

  normalInformationMatrixCalculator.setCurvatureThreshold(0.02f);

  nicp::CloudConverter converter(
            &pointInformationMatrixCalculator,
            &normalInformationMatrixCalculator);

  nicp::CorrespondenceFinderNN correspondenceFinder;
  correspondenceFinder.setInlierDistanceThreshold(0.5f);
  correspondenceFinder.setInlierNormalAngularThreshold(0.9f);
  correspondenceFinder.setFlatCurvatureThreshold(0.02f);

  // Create Linearizer and Aligner
  nicp::Linearizer2d linearizer;
  nicp::AlignerNN2d aligner;

  linearizer.setInlierMaxChi2(9e3);
  linearizer.setRobustKernel(true);
  linearizer.setZScaling(false);
  linearizer.setAligner(&aligner);

  aligner.setOuterIterations(10);
  aligner.setLambda(1e3);
 
  aligner.setCorrespondenceFinder(&correspondenceFinder);
  aligner.setLinearizer(&linearizer);

  // Get clouds
  Eigen::Isometry2f initialGuess = Eigen::Isometry2f::Identity();
  nicp::Cloud referenceCloud, currentCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr referencePclCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr currentPclCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *referencePclCloud) == -1)
  {
      PCL_ERROR ("Couldn't read file  \n");
      return (-1);
  }
  
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[2], *currentPclCloud) == -1)
  {
      PCL_ERROR ("Couldn't read file  \n");
      return (-1);
  }

  std::chrono::steady_clock::time_point start, end;
  start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point start_t, end_t;
  start_t = std::chrono::steady_clock::now();
  
  converter.compute(referenceCloud, referencePclCloud);
  converter.compute(currentCloud, currentPclCloud);
  end_t = std::chrono::steady_clock::now();
  cout << " converter.compute " << std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count() << "us" << endl;

  
  
  aligner.setDebug(true);
  
  aligner.setReferenceCloud(&referenceCloud);
  aligner.setCurrentCloud(&currentCloud);
  aligner.setInitialGuess(initialGuess);
  aligner.align();
  end = std::chrono::steady_clock::now();
  cout << "cost " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << endl;
  Eigen::Isometry2f T = aligner.T();

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

	transform_2.translation() << T.matrix()(0, 2),  T.matrix()(1, 2), 0.0;	

	transform_2.rotate(Eigen::AngleAxisf(atan2(T.matrix()(1, 0), T.matrix()(0, 0)), Eigen::Vector3f::UnitZ()));	
	pcl::transformPointCloud(*currentPclCloud, *outputCloud, transform_2);	


  std::cout << std::endl << "Transformation: " << std::endl << T.matrix().inverse() << std::endl;
  pcl::visualization::PCLVisualizer::Ptr
  viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer_final->setBackgroundColor (0, 0, 0);

  // Coloring and visualizing target cloud (red).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  referencePclCloud_color (referencePclCloud, 0, 255, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (referencePclCloud, referencePclCloud_color, "referencePclCloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "referencePclCloud");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
 outputCloud_color (outputCloud,255, 0, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (outputCloud, outputCloud_color, "outputCloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "outputCloud");
  // Coloring and visualizing transformed input cloud (green).
  // string name = "line";
  // // cout << "currentPclCloud.size()" << currentPclCloud->size() << endl;
  // for(int i = 0; i < referencePclCloud->size(); i++){
  //   viewer_final->addLine<pcl::PointXYZ>(referencePclCloud->points[i], 
  //   pcl::PointXYZ(referencePclCloud->points[i].x + referenceCloud.normals().at(i).x(), 
  //   referencePclCloud->points[i].y + referenceCloud.normals().at(i).y(), 
  //   referencePclCloud->points[i].z + referenceCloud.normals().at(i).z()), name + to_string(i));
  // }
  
  // Starting visualizer
  viewer_final->addCoordinateSystem (1.0, "global");
  viewer_final->initCameraParameters ();

  // Wait until visualizer window is closed.
  while (!viewer_final->wasStopped ())
  {
    viewer_final->spinOnce (100);
    
    // this_thread::sleep_for(chrono_literals::100ms);
  }


  return 0;
}
