# Install script for directory: /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/lib/local/LandmarkDetector/libLandmarkDetector.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenFace" TYPE FILE FILES
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/CCNF_patch_expert.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/CEN_patch_expert.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/CNN_utils.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/FaceDetectorMTCNN.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/LandmarkCoreIncludes.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/LandmarkDetectionValidator.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/LandmarkDetectorFunc.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/LandmarkDetectorModel.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/LandmarkDetectorParameters.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/LandmarkDetectorUtils.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/Patch_experts.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/PAW.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/PDM.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/SVR_patch_expert.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/LandmarkDetector/include/stdafx.h"
    )
endif()

