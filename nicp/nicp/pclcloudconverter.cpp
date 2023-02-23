/*
 * @Author: LuoChen 1425523063@qq.com
 * @Date: 2023-02-16 20:04:34
 * @LastEditors: LuoChen 1425523063@qq.com
 * @LastEditTime: 2023-02-23 14:22:41
 * @FilePath: /nicp/nicp/nicp/pclcloudconverter.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "pclcloudconverter.h"

namespace nicp{
  CloudConverter::CloudConverter(
			
			PointInformationMatrixCalculator2d *pointInformationMatrixCalculator,
			NormalInformationMatrixCalculator2d *normalInformationMatrixCalculator,
      float radius, int minPoints) : _radius(radius), _minPoints(minPoints) {
        _curvatureThreshold = 0.9;
        
        _pointInformationMatrixCalculator = pointInformationMatrixCalculator;
        _normalInformationMatrixCalculator = normalInformationMatrixCalculator;
      }


  void CloudConverter::compute(Cloud &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr T_cloud){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    
    kdtree.setInputCloud(T_cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    cloud.resize( T_cloud->points.size());

    for(size_t i = 0; i < cloud.size(); i++){
    
      cloud.points().at(i) = Point(Eigen::Vector3f(T_cloud->points.at(i).x, T_cloud->points.at(i).y, 0));
    }
    
    NormalVector& normals = cloud.normals();

    StatsVector& statsVector = cloud.stats();
    
    if(statsVector.size() != T_cloud->points.size()) statsVector.resize(T_cloud->points.size());
    if(normals.size() != T_cloud->points.size()) normals.resize(T_cloud->points.size());
    Normal dummyNormal = Normal::Zero();
    std::fill(normals.begin(), normals.end(), dummyNormal);
    std::fill(statsVector.begin(), statsVector.end(), Stats());

    for(size_t index = 0; index < T_cloud->points.size(); ++index){
      Eigen::Vector3f mean = Eigen::Vector3f::Zero();
      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      int nums = kdtree.radiusSearch (T_cloud->at(index), _radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
      // cout << "nums" << nums << endl;
      if ( nums > _minPoints )
      {
        for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){
          mean[0] += (*T_cloud)[pointIdxRadiusSearch[i]].x;
          mean[1] += (*T_cloud)[pointIdxRadiusSearch[i]].y;
        }
        mean /= static_cast<float>(pointIdxRadiusSearch.size());
        for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){
          
          auto near_point = Eigen::Vector3f((*T_cloud)[pointIdxRadiusSearch[i]].x, (*T_cloud)[pointIdxRadiusSearch[i]].y, 0.0f);
          cov += (near_point - mean) * (near_point - mean).transpose();
        }

        //协方差cov分解
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver;
	      eigenSolver.computeDirect(cov, Eigen::ComputeEigenvectors);
        
        Stats &stats = statsVector[index];
        stats.setZero();
        stats.setEigenVectors(eigenSolver.eigenvectors());
        stats.setMean(mean);
        Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
        if(eigenValues(1) < 0.0f) 
        { 
          eigenValues(1) = 0.0f; 
        }	  
        stats.setEigenValues(eigenValues);
        stats.setN(pointIdxRadiusSearch.size ());
        stats.setCurvature(eigenValues(1) / (eigenValues(1) + eigenValues(2)));
        normals.at(index) = stats.block<4, 1>(0, 1);
        if(stats.curvature() < _curvatureThreshold) {
          if(normals.at(index).dot(cloud.points()[index]) > 0) { normals.at(index) = -normals.at(index); }
        } 
        else { 
          normals.at(index).setZero(); 
        }      	
        
      }
      
    }
    
    _pointInformationMatrixCalculator->compute(cloud.pointInformationMatrix(), cloud.stats(), cloud.normals());
    _normalInformationMatrixCalculator->compute(cloud.normalInformationMatrix(), cloud.stats(), cloud.normals());

  }
}
