# Install script for directory: /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/lib/local/Utilities/libUtilities.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenFace" TYPE FILE FILES
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/ImageCapture.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/RecorderCSV.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/RecorderHOG.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/RecorderOpenFace.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/RecorderOpenFaceParameters.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/SequenceCapture.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/stdafx_ut.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/VisualizationUtils.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/Visualizer.h"
    "/home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/lib/local/Utilities/include/ConcurrentQueue.h"
    )
endif()

