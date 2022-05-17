

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was OpenFaceConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

find_dependency(dlib 18.18)
find_dependency(OpenCV 3.3)
find_dependency(Boost)
find_dependency(BLAS)
find_dependency(TBB)

include("${CMAKE_CURRENT_LIST_DIR}/OpenFaceTargets.cmake")

set_and_check(OpenFace_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/../../../src/config/openface/lib/local/GazeAnalyser/include;../../../src/config/openface/lib/local/FaceAnalyser/include;../../../src/config/openface/lib/local/LandmarkDetector/include;../../../src/config/openface/lib/local/Utilities/include;
                            ${dlib_INCLUDE_DIRS};
                            ${Boost_INCLUDE_DIRS};
                            ${Boost_INCLUDE_DIRS}/boost;")
set(OpenFace_LIBRARIES OpenFace::GazeAnalyser;OpenFace::FaceAnalyser;OpenFace::LandmarkDetector;OpenFace::Utilities)
