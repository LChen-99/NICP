/*
 * @Author: LuoChen 1425523063@qq.com
 * @Date: 2023-02-02 09:56:53
 * @LastEditors: LuoChen 1425523063@qq.com
 * @LastEditTime: 2023-02-21 18:55:40
 * @FilePath: /nicp/nicp/nicp/informationmatrixcalculator.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "informationmatrixcalculator.h"

#include <omp.h>
#include <iostream>
using namespace Eigen;

namespace nicp {

  void PointInformationMatrixCalculator::compute(InformationMatrixVector &informationMatrix,
						 const StatsVector &statsVector,
						 const NormalVector &imageNormals) {
    assert(statsVector.size() > 0 && "PointInformationMatrixCalculator: statsVector has zero size");
    assert(imageNormals.size() > 0 && "PointInformationMatrixCalculator: imageNormals has zero size");

    informationMatrix.resize(statsVector.size());
    
#pragma omp parallel for
    for(size_t i = 0; i < statsVector.size(); i++) {
      const Stats &stats = statsVector[i];
      InformationMatrix U = Matrix4f::Zero();
      U.block<3, 3>(0, 0) = stats.eigenVectors(); 
      
      if(imageNormals[i].squaredNorm() > 0) {
        if(stats.curvature() < _curvatureThreshold)
          informationMatrix[i] = U * _flatInformationMatrix * U.transpose();
        else {
          informationMatrix[i] = U * _nonFlatInformationMatrix * U.transpose();
        }
      } 
      else informationMatrix[i] = InformationMatrix();
      // std::cout << informationMatrix[i] << std::endl;
      // std::cout << "----" << std::endl;
    }
  }
  void PointInformationMatrixCalculator2d::compute(InformationMatrixVector &informationMatrix,
						 const StatsVector &statsVector,
						 const NormalVector &imageNormals) {
    assert(statsVector.size() > 0 && "PointInformationMatrixCalculator: statsVector has zero size");
    assert(imageNormals.size() > 0 && "PointInformationMatrixCalculator: imageNormals has zero size");

    informationMatrix.resize(statsVector.size());
    
#pragma omp parallel for
    for(size_t i = 0; i < statsVector.size(); i++) {
      const Stats &stats = statsVector[i];
      InformationMatrix U = Matrix4f::Zero();
      U.block<3, 3>(0, 0) = stats.eigenVectors(); 
      
      if(imageNormals[i].squaredNorm() > 0) {
        if(stats.curvature() < _curvatureThreshold)
          informationMatrix[i] = U * _flatInformationMatrix * U.transpose();
        else {
          informationMatrix[i] = U * _nonFlatInformationMatrix * U.transpose();
        }
      } 
      else informationMatrix[i] = InformationMatrix();
      // std::cout << informationMatrix[i] << std::endl;
      // std::cout << "----" << std::endl;
    }
  }
  void NormalInformationMatrixCalculator::compute(InformationMatrixVector &informationMatrix,
						  const StatsVector &statsVector,
						  const NormalVector &imageNormals) {
    assert(statsVector.size() > 0 && "PointInformationMatrixCalculator: statsVector has zero size");
    assert(imageNormals.size() > 0 && "PointInformationMatrixCalculator: imageNormals has zero size");

    informationMatrix.resize(statsVector.size());

#pragma omp parallel for
    for(size_t i = 0; i < statsVector.size(); i++) {
      const Stats &stats = statsVector[i];
      InformationMatrix U = Matrix4f::Zero();
      U.block<3, 3>(0, 0) = stats.eigenVectors(); 
      if(imageNormals[i].squaredNorm()>0) {
        if(stats.curvature() < _curvatureThreshold)
          informationMatrix[i] = U * _flatInformationMatrix * U.transpose();
        else {	  
          informationMatrix[i] = U * _nonFlatInformationMatrix * U.transpose();
        }
      } 
      else informationMatrix[i] = InformationMatrix();
    }
  }

  
  void NormalInformationMatrixCalculator2d::compute(InformationMatrixVector &informationMatrix,
						  const StatsVector &statsVector,
						  const NormalVector &imageNormals) {
    assert(statsVector.size() > 0 && "PointInformationMatrixCalculator: statsVector has zero size");
    assert(imageNormals.size() > 0 && "PointInformationMatrixCalculator: imageNormals has zero size");

    informationMatrix.resize(statsVector.size());

#pragma omp parallel for
    for(size_t i = 0; i < statsVector.size(); i++) {
      const Stats &stats = statsVector[i];
      InformationMatrix U = Matrix4f::Zero();
      U.block<3, 3>(0, 0) = stats.eigenVectors(); 
      if(imageNormals[i].squaredNorm()>0) {
        if(stats.curvature() < _curvatureThreshold)
          informationMatrix[i] = U * _flatInformationMatrix * U.transpose();
        else {	  
          informationMatrix[i] = U * _nonFlatInformationMatrix * U.transpose();
        }
      } 
      else informationMatrix[i] = InformationMatrix();
    }
  }

}
