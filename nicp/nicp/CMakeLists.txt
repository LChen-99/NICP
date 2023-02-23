ADD_LIBRARY(nicp SHARED
  aligner2d.h aligner2d.cpp
  definitions.h
  imageutils.cpp imageutils.h
  aligner.cpp aligner.h
  alignernn.cpp alignernn.h
  alignerprojective.cpp alignerprojective.h
  correspondencefinder.cpp correspondencefinder.h
  correspondencefindernn.cpp correspondencefindernn.h
  correspondencefinderprojective.cpp correspondencefinderprojective.h
  depthimageconverter.cpp depthimageconverter.h
  depthimageconverterintegralimage.cpp depthimageconverterintegralimage.h
  cloud.cpp cloud.h
  gaussian.h
  gaussian3.cpp gaussian3.h
  homogeneousvector4f.h
  informationmatrix.h
  informationmatrixcalculator.h informationmatrixcalculator.cpp
  linearizer.cpp linearizer.h
  merger.cpp merger.h
  pinholepointprojector.cpp pinholepointprojector.h
  pointaccumulator.h
  pointintegralimage.cpp pointintegralimage.h
  pointprojector.cpp pointprojector.h
  se3_prior.cpp se3_prior.h
  parallelcylindricalpointprojector.cpp parallelcylindricalpointprojector.h
  stats.h
  statscalculator.cpp statscalculator.h
  statscalculatorintegralimage.cpp statscalculatorintegralimage.h
  sphericalpointprojector.cpp sphericalpointprojector.h
  transformable_vector.h
  voxelcalculator.cpp voxelcalculator.h
  pclcloudconverter.h pclcloudconverter.cpp
)

SET_TARGET_PROPERTIES(nicp PROPERTIES OUTPUT_NAME ${LIB_PREFIX})
TARGET_LINK_LIBRARIES(nicp ${OpenCV_LIBS} ${FLANN_LIBRARY} ${PCL_LIBRARIES})

ADD_EXECUTABLE(nicp_simple_aligner nicp_simple_aligner.cpp )
SET_TARGET_PROPERTIES(nicp_simple_aligner PROPERTIES OUTPUT_NAME nicp_simple_aligner)
TARGET_LINK_LIBRARIES(nicp_simple_aligner nicp ${OpenCV_LIBS} ${PCL_LIBRARIES})

ADD_EXECUTABLE(nicp_aligner nicp_aligner.cpp )
SET_TARGET_PROPERTIES(nicp_aligner PROPERTIES OUTPUT_NAME nicp_aligner)
TARGET_LINK_LIBRARIES(nicp_aligner nicp ${OpenCV_LIBS} ${PCL_LIBRARIES})

ADD_EXECUTABLE(nicp_aligner_2d nicp_nn_pcl_registration.cpp )
# SET_TARGET_PROPERTIES(nicp_aligner_2d PROPERTIES OUTPUT_NAME nicp_aligner)
TARGET_LINK_LIBRARIES(nicp_aligner_2d nicp ${OpenCV_LIBS} ${PCL_LIBRARIES})

ADD_EXECUTABLE(nicp_nn_depth_image_registration nicp_nn_depth_image_registration.cpp)
TARGET_LINK_LIBRARIES(nicp_nn_depth_image_registration
  nicp
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  
)

ADD_EXECUTABLE(nicp_depth_image_registration nicp_depth_image_registration.cpp)
TARGET_LINK_LIBRARIES(nicp_depth_image_registration
  nicp
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
ADD_EXECUTABLE(hello hello.cpp
  
)
TARGET_LINK_LIBRARIES(hello
  nicp
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
