#include "aligner2d.h"

#include <sys/time.h>
#include <omp.h>

#include <opencv2/highgui/highgui.hpp>
#include "nicp/imageutils.h"

#include "unscented.h"
#include "bm_se3.h"

#include <cstdio>
#include "sphericalpointprojector.h"
#include "pinholepointprojector.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;

namespace nicp {

  void AlignerNN2d::align() {

    // assert(_projector && "AlignerProjective: missing _projector");
    assert(_linearizer && "AlignerProjective: missing _linearizer");
    assert(_correspondenceFinder && "AlignerProjective: missing _correspondenceFinder");
    assert(_referenceCloud && "AlignerProjective: missing _referenceCloud");
    assert(_currentCloud && "AlignerProjective: missing _currentCloud");
  
    CorrespondenceFinderNN* cfnn = dynamic_cast<CorrespondenceFinderNN*>(_correspondenceFinder);
    assert(cfnn && "AlignerProjective: _correspondenceFinder is not of type CorrespondnceFinderNN");

    struct timeval tvStart, tvEnd;
    gettimeofday(&tvStart, 0);
    // std::chrono::steady_clock::time_point start, end;
    // start = std::chrono::steady_clock::now();
    cfnn->init(*_referenceCloud, *_currentCloud);
    _T = _initialGuess;
    
    for(int i = 0; i < _outerIterations; i++) {
      /************************************************************************
       *                         Correspondence Computation                   *
       ************************************************************************/
      // Compute the indices of the current scene from the point of view of the sensor
      _T.matrix().row(2) << 0.0f, 0.0f, 1.0f;
        
      // Correspondences computation.
      // std::chrono::steady_clock::time_point start_t, end_t;
      // start_t = std::chrono::steady_clock::now();
      
      cfnn->compute(*_referenceCloud, *_currentCloud, To3d(_T.inverse()));
      // end_t = std::chrono::steady_clock::now();
      // cout << "cfnn cost " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << "ms" << endl;
  
      /************************************************************************
       *                            Alignment                                 *
       ************************************************************************/
      Eigen::Isometry2f invT = _T.inverse();

      for(int k = 0; k < _innerIterations; k++) {
        
        invT.matrix().block<1, 3>(2, 0) << 0.0f, 0.0f, 1.0f;
        Matrix3f H;
        Vector3f b;

        _linearizer->setT(invT);
        
        _linearizer->update();
 
        H = _linearizer->H();
        b = _linearizer->b();
        H += Matrix3f::Identity() * _lambda;
        

        Vector3f dx = H.ldlt().solve(-b);
        Eigen::Isometry2f dT;
        dT.matrix() << cos(dx(2)), -sin(dx(2)), dx(0), sin(dx(2)), cos(dx(2)), dx(1), 0, 0, 1;
        invT = dT * invT;
        if(dx.norm() < 1e-6) break;
      }

      _T = invT.inverse();
    
      _T.matrix().block<1, 3>(2, 0) << 0.0f, 0.0f, 1.0f;
      
    }
   
    // end = std::chrono::steady_clock::now();
    gettimeofday(&tvEnd, 0);
    double tStart = tvStart.tv_sec * 1000.0f + tvStart.tv_usec * 0.001f;
    double tEnd = tvEnd.tv_sec * 1000.0f + tvEnd.tv_usec * 0.001f;
    _totalTime = tEnd - tStart;
    _error = _linearizer->error();
    _inliers = _linearizer->inliers();

    // if(_debug){
    //   cout << "cost " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;
    // }
    // _computeStatistics(_mean, _omega, _translationalEigenRatio, _rotationalEigenRatio);
  //   if (_rotationalEigenRatio > _rotationalMinEigenRatio ||
	// _translationalEigenRatio > _translationalMinEigenRatio) {
  //     _solutionValid = false;
  //     if (_debug) {
	// cout << endl;
	// cout << "************** WARNING SOLUTION MIGHT BE INVALID (eigenratio failure) **************" << endl;
	// cout << "tr: " << _translationalEigenRatio << " rr: " << _rotationalEigenRatio << endl;
	// cout << "************************************************************************************" << endl;
  //     }
  //   }
  //   else {
  //     _solutionValid = true;
  //     if (_debug) {
	// cout << "************** I FOUND SOLUTION VALID SOLUTION   (eigenratio ok) *******************" << endl;
	// cout << "tr: " << _translationalEigenRatio << " rr: " << _rotationalEigenRatio << endl;
	// cout << "************************************************************************************" << endl;
  //     }
  //   }
  //   if (_debug) {
  //     cout << "Solution statistics in (t, mq): " << endl;
  //     cout << "mean: " << _mean.transpose() << endl;
  //     cout << "Omega: " << endl;
  //     cout << _omega << endl;
  //   }
  }

}
