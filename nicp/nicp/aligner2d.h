/*
 * @Author: LuoChen 1425523063@qq.com
 * @Date: 2023-02-16 17:28:50
 * @LastEditors: LuoChen 1425523063@qq.com
 * @LastEditTime: 2023-02-23 14:27:20
 * @FilePath: /nicp/nicp/nicp/aligner2d.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include "aligner.h"
#include "correspondencefindernn.h"
#include <chrono>
namespace nicp {

  /** \class Aligner aligner.h "aligner.h"
   *  \brief Class for point cloud alignment.
   *
   *  This class allows to compute the homogeneous transformation that brings
   *  a point cloud to superpose an other reference point cloud. Data association
   *  is computed via nearest neighbor.
   */
class AlignerNN2d : public Aligner {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     *  Empty constructor.
     *  This constructor creates a Projective Aligner with default values for all its attributes.
     *  All the pointers to objects implementing an algorithm have to be set since
     *  this constructor sets them to zero.
     */
    AlignerNN2d() : Aligner() {
      _T.matrix() = Matrix3f::Identity();
      _innerIterations = 20;
    }

    /**
     *  This constructor creates an Aligner with default values for all its attributes.
     *  All internal algorithms are set using the input values.
     *  @param projector_ is a pointer to the point projector that will be used by the Aligner.
     *  @param linearizer_ is a pointer to the linearizer that will be used by the Aligner.
     *  @param correspondenceFinder_ is a pointer to the point correspondence finder that will be used by the Aligner.
     */
    AlignerNN2d(PointProjector* projector_, Linearizer* linearizer_, CorrespondenceFinderNN* correspondenceFinder_) : Aligner(projector_, linearizer_, correspondenceFinder_) {}
    inline const Eigen::Isometry2f& T() const { return _T; }
    /**
     *  Destructor.
     */
    virtual ~AlignerNN2d() {}
    void setInitialGuess(Eigen::Isometry2f T){
      _initialGuess = T;
    };
    inline void setLinearizer(Linearizer2d* linearizer_) {
      _linearizer = linearizer_;
      if( _linearizer)
	      _linearizer->setAligner(this);
    }
    /**
     *  This method computes the final transformation that brings the cloud to align to superpose the reference
     *  cloud.
     */
    virtual void align();

    Eigen::Isometry3f To3d(Eigen::Isometry2f T_2d){
      Eigen::Isometry3f transform_3d;
      Eigen::Matrix4f T = Eigen::Matrix4f::Zero();
      T.block<2, 2>(0, 0) = T_2d.matrix().block<2, 2>(0, 0);
      T.block<2, 1>(0, 3) = T_2d.matrix().block<2, 1>(0, 2);
      transform_3d.matrix() = T;
      return transform_3d;
    }
    Eigen::Isometry2f _T;
    Eigen::Isometry2f _initialGuess;
    Linearizer2d *_linearizer;
  };

}
