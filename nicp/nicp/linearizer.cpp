#include "aligner.h"

#include <omp.h>

using namespace std;

namespace nicp {

  Linearizer::Linearizer() {
    _aligner = 0;
    _H.setZero();
    _b.setZero();
    _inlierMaxChi2 = 9e3;
    _robustKernel = true;
    _demotedToGeneralizedICP = false;
    _zScaling = false;
    _scale = 1.0f;
    _T.matrix() = Eigen::Matrix4f::Identity();
  }

  void Linearizer::update() {
    assert(_aligner && "Aligner: missing _aligner");

    // Variables initialization.
    _b = Vector6f::Zero();
    _H = Matrix6f::Zero();
    const InformationMatrixVector &pointOmegas = _aligner->currentCloud()->pointInformationMatrix();
    const InformationMatrixVector &normalOmegas = _aligner->currentCloud()->normalInformationMatrix();
    
    // Allocate the variables for the sum reduction;
    int numThreads = omp_get_max_threads();
    Matrix4f _Htt[numThreads], _Htr[numThreads], _Hrr[numThreads];
    Vector4f _bt[numThreads], _br[numThreads];
    int _inliers[numThreads];
    float _errors[numThreads];
    
    int iterationsPerThread = _aligner->correspondenceFinder()->numCorrespondences() / numThreads;
#pragma omp parallel
    {
      int threadId = omp_get_thread_num();
      int imin = iterationsPerThread * threadId;
      int imax = imin + iterationsPerThread;
      if(imax > _aligner->correspondenceFinder()->numCorrespondences())
	      imax = _aligner->correspondenceFinder()->numCorrespondences();

      Eigen::Matrix4f Htt;
      Eigen::Matrix4f Htr;
      Eigen::Matrix4f Hrr;
      Eigen::Vector4f bt;
      Eigen::Vector4f br;
      int inliers;
      float error;
      Htt = Matrix4f::Zero(); 
      Htr = Matrix4f::Zero(); 
      Hrr = Matrix4f::Zero();
      bt = Vector4f::Zero();
      br = Vector4f::Zero();
      error = 0;
      inliers = 0;
      for(int i = imin; i < imax; i++) {	
        const Correspondence &correspondence = _aligner->correspondenceFinder()->correspondences()[i];
        const Point referencePoint = _T * _aligner->referenceCloud()->points()[correspondence.referenceIndex];
        const Normal referenceNormal = _T * _aligner->referenceCloud()->normals()[correspondence.referenceIndex];
        const Point &currentPoint = _aligner->currentCloud()->points()[correspondence.currentIndex];
        const Normal &currentNormal = _aligner->currentCloud()->normals()[correspondence.currentIndex];
        InformationMatrix omegaP = pointOmegas[correspondence.currentIndex];	
        InformationMatrix omegaN = _scale * normalOmegas[correspondence.currentIndex];
        if(_zScaling) {
          omegaP *= 1.0f / fabs(currentPoint.z());
          omegaN *= 1.0f / fabs(currentPoint.z());
        }
        //如果用GICP
        if(_demotedToGeneralizedICP) { omegaN.setZero(); }
            
        const Vector4f pointError = referencePoint - currentPoint;
        const Vector4f normalError = referenceNormal - currentNormal;
        const Vector4f ep = omegaP * pointError;
        const Vector4f en = omegaN * normalError;

        float localError = pointError.dot(ep) + normalError.dot(en);

        float kscale = 1;
        if(localError > _inlierMaxChi2) {
          if (_robustKernel) {
            kscale = _inlierMaxChi2 / localError;
          } 
          else {
            continue;
          }
        }
        inliers++;
        error += kscale * localError;	
        //skew中*2了。。。
        Matrix4f Sp = skew(referencePoint);
        Matrix4f Sn = skew(referenceNormal);
        Htt.noalias() += omegaP * kscale;
        Htr.noalias() += omegaP * Sp * kscale;
        Hrr.noalias() += (Sp.transpose() * omegaP * Sp + Sn.transpose() * omegaN * Sn) * kscale;
        bt.noalias() += kscale * ep;
        br.noalias() += kscale * (Sp.transpose() * ep + Sn.transpose() * en);
      }

      _Htt[threadId] = Htt;
      _Htr[threadId] = Htr;
      _Hrr[threadId] = Hrr;
      _bt[threadId] = bt;
      _br[threadId] = br;
      _errors[threadId] = error;
      _inliers[threadId] = inliers;
    }

    // Now do the reduce
    Eigen::Matrix4f Htt = Eigen::Matrix4f::Zero();
    Eigen::Matrix4f Htr = Eigen::Matrix4f::Zero();
    Eigen::Matrix4f Hrr = Eigen::Matrix4f::Zero();
    Eigen::Vector4f bt = Eigen::Vector4f::Zero();
    Eigen::Vector4f br = Eigen::Vector4f::Zero();
    this->_inliers = 0;
    this->_error = 0;
    for(int t = 0; t < numThreads; t++) {
      Htt += _Htt[t];
      Htr += _Htr[t];
      Hrr += _Hrr[t];
      bt += _bt[t];
      br += _br[t];
      this->_inliers += _inliers[t];
      this->_error += _errors[t];
    }
    _H.block<3, 3>(0, 0) = Htt.block<3, 3>(0, 0);
    _H.block<3, 3>(0, 3) = Htr.block<3, 3>(0, 0);
    _H.block<3, 3>(3, 3) = Hrr.block<3, 3>(0, 0);
    _H.block<3, 3>(3, 0) = _H.block<3, 3>(0, 3).transpose();
    _b.block<3, 1>(0, 0) = bt.block<3, 1>(0, 0);
    _b.block<3, 1>(3, 0) = br.block<3, 1>(0, 0);
    // _b(2, 0) = 0.0f;
    // _b(4, 0) = 0.0f;
    // _b(3, 0) = 0.0f;
    // cout << "_H" << _H << endl;
    // cout << "_b" << _b << endl;
  }


  Linearizer2d::Linearizer2d(){
    _aligner = 0;
    _H.setZero();
    _b.setZero();
    _inlierMaxChi2 = 9e3;
    _robustKernel = true;
    _demotedToGeneralizedICP = false;
    _zScaling = false;
    _scale = 1.0f;
    _T.matrix() = Eigen::Matrix3f::Identity();
  }

  void Linearizer2d::update() {
    assert(_aligner && "Aligner: missing _aligner");
   
    // Variables initialization.
    _b = Vector3f::Zero();
    _H = Matrix3f::Zero();
    const InformationMatrixVector &pointOmegas = _aligner->currentCloud()->pointInformationMatrix();
    const InformationMatrixVector &normalOmegas = _aligner->currentCloud()->normalInformationMatrix();
    int inliers;
    float error;
    error = 0;
    inliers = 0;
    Eigen::Matrix4f T = Eigen::Matrix4f::Zero();
    T.block<2, 2>(0, 0) = _T.matrix().block<2, 2>(0, 0);
    T.block<2, 1>(0, 3) = _T.matrix().block<2, 1>(0, 2);
    
    T(2, 2) = 1.0f;
    T(3, 3) = 1.0f;
    
    Matrix3f H = Matrix3f::Zero();
    
    Vector3f b = Vector3f::Zero();
    
    
    for(int i = 0; i < _aligner->correspondenceFinder()->numCorrespondences(); i++) {	
      const Correspondence &correspondence = _aligner->correspondenceFinder()->correspondences()[i];
      const Point referencePoint = T * _aligner->referenceCloud()->points()[correspondence.referenceIndex];
      const Normal referenceNormal = T * _aligner->referenceCloud()->normals()[correspondence.referenceIndex];
      const Point &currentPoint = _aligner->currentCloud()->points()[correspondence.currentIndex];
      const Normal &currentNormal = _aligner->currentCloud()->normals()[correspondence.currentIndex];
      // InformationMatrix omegaP = pointOmegas[correspondence.currentIndex];	
      // InformationMatrix omegaN = _scale * normalOmegas[correspondence.currentIndex];
      
      Matrix4f omega = Matrix4f::Zero();
      omega.block<2, 2>(0, 0) = pointOmegas[correspondence.currentIndex].block<2, 2>(0, 0);
      omega.block<2, 2>(2, 2) = _scale * normalOmegas[correspondence.currentIndex].block<2, 2>(0, 0);
      if(_zScaling) {
        omega.block<2, 2>(0, 0) *= 1.0f / fabs(currentPoint.z());
        omega.block<2, 2>(2, 2) *= 1.0f / fabs(currentPoint.z());
      }
      // //如果用GICP
      // if(_demotedToGeneralizedICP) { omegaN.setZero(); }
          
      Vector4f Error;
      Error.block<2, 1>(0, 0) = referencePoint.block<2, 1>(0, 0) - currentPoint.block<2, 1>(0, 0);
      Error.block<2, 1>(2, 0) = referenceNormal.block<2, 1>(0, 0) - currentNormal.block<2, 1>(0, 0);
      Matrix<float, 4, 3> J;
      float x = referencePoint(0, 0);
      float y = referencePoint(1, 0);
      float sin = _T.matrix()(1, 0);
      float cos = _T.matrix()(0, 0);

      float x_n = currentNormal(0, 0);
      float y_n = currentNormal(1, 0);
      J << 1, 0, -x*sin-y*cos, 0, 1, x*cos-y*sin, 0 , 0, -x_n*sin-y_n*cos, 0, 0, x_n*cos-y_n*sin;
      
      
      float localError = Error.dot(Error);

      float kscale = 1;
      if(localError > _inlierMaxChi2) {
        if (_robustKernel) {
          kscale = _inlierMaxChi2 / localError;
        } 
        else {
          continue;
        }
      }

      H += kscale * J.transpose() * omega * J;
      b += kscale * J.transpose() * omega * Error;
      inliers++;
      error += kscale * localError;	

    }
    _H = H;
    _b = b;
    
  }
  
}
