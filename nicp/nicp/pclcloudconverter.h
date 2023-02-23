/*
 * @Author: LuoChen 1425523063@qq.com
 * @Date: 2023-02-16 19:41:16
 * @LastEditors: LuoChen 1425523063@qq.com
 * @LastEditTime: 2023-02-23 17:38:54
 * @FilePath: /nicp/nicp/nicp/pclclouconverter.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef PCLCLOUDCONVERT_H
#define PCLCLOUDCONVERT_H
#include <pcl/io/pcd_io.h>
#include "statscalculator.h"
#include "informationmatrixcalculator.h"
#include "cloud.h"
#include "homogeneousvector4f.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/SVD>  
#include <Eigen/Dense>  
namespace nicp{


	class CloudConverter{
	public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		CloudConverter(
			
			PointInformationMatrixCalculator2d *_pointInformationMatrixCalculator = 0,
			NormalInformationMatrixCalculator2d *_normalInformationMatrixCalculator = 0,
      float _radius = 0.5,
      int _minPoints = 30);
		
    virtual ~CloudConverter() {}

		inline PointInformationMatrixCalculator2d* pointInformationMatrixCalculator() { return _pointInformationMatrixCalculator; }
    
    /**
     *  Method that set the PointInformationMatrixCalculator used by the DepthImageConverter to the one given in input.
     *  @param pointInformationMatrixCalculator_ is a pointer to the PointInformationMatrixCalculator used to update the DepthImageConverter's PointInformationMatrixCalculator. 
     *  @see pointInformationMatrixCalculator()
     */
    inline void setPointInformationMatrixCalculator(PointInformationMatrixCalculator2d *pointInformationMatrixCalculator_) { _pointInformationMatrixCalculator = pointInformationMatrixCalculator_; }

    /**
     *  Method that returns a pointer to the NormalInformationMatrixCalculator used by the DepthImageConverter.
     *  @return a pointer to the DepthImageConverter's NormalInformationMatrixCalculator.
     *  @see setNormalInformationMatrixCalculator()
     */
    inline NormalInformationMatrixCalculator2d* normalInformationMatrixCalculator() { return _normalInformationMatrixCalculator; }

    /**
     *  Method that set the NormalInformationMatrixCalculator used by the DepthImageConverter to the one given in input.
     *  @param normalInformationMatrixCalculator_ is a pointer to the NormalInformationMatrixCalculator used to update the DepthImageConverter's NormalInformationMatrixCalculator. 
     *  @see normalInformationMatrixCalculator()
     */
    inline void setNormalInformationMatrixCalculator(NormalInformationMatrixCalculator2d *normalInformationMatrixCalculator_) { _normalInformationMatrixCalculator = normalInformationMatrixCalculator_; }

		
		void compute(Cloud &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr T_cloud);
		
	private:
	
		PointInformationMatrixCalculator2d *_pointInformationMatrixCalculator; /**< Pointer to the PointInformationMatrixCalculator used by the DepthImageConverter to compute the information matrix of the points. */
    NormalInformationMatrixCalculator2d *_normalInformationMatrixCalculator; /**< Pointer to the NormalInformationMatrixCalculator used by the DepthImageConverter to compute the information matrix of the normals. */
    float _radius;
    int _minPoints;
    float _curvatureThreshold;
	};


}

#endif