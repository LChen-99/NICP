/*
 * @Author: LuoChen 1425523063@qq.com
 * @Date: 2023-02-02 09:56:53
 * @LastEditors: LuoChen 1425523063@qq.com
 * @LastEditTime: 2023-02-23 11:26:30
 * @FilePath: /nicp/nicp/nicp/alignernn.h
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
class AlignerNN : public Aligner {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     *  Empty constructor.
     *  This constructor creates a Projective Aligner with default values for all its attributes.
     *  All the pointers to objects implementing an algorithm have to be set since
     *  this constructor sets them to zero.
     */
    AlignerNN() : Aligner() {}

    /**
     *  This constructor creates an Aligner with default values for all its attributes.
     *  All internal algorithms are set using the input values.
     *  @param projector_ is a pointer to the point projector that will be used by the Aligner.
     *  @param linearizer_ is a pointer to the linearizer that will be used by the Aligner.
     *  @param correspondenceFinder_ is a pointer to the point correspondence finder that will be used by the Aligner.
     */
    AlignerNN(PointProjector* projector_, Linearizer* linearizer_, CorrespondenceFinderNN* correspondenceFinder_) : Aligner(projector_, linearizer_, correspondenceFinder_) {}

    /**
     *  Destructor.
     */
    virtual ~AlignerNN() {}

    /**
     *  This method computes the final transformation that brings the cloud to align to superpose the reference
     *  cloud.
     */
    virtual void align();

  };

}
