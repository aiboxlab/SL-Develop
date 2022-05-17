# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/config/openface

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build

# Include any dependencies generated for this target.
include exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/depend.make

# Include the progress variables for this target.
include exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/progress.make

# Include the compile flags for this target's objects.
include exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/flags.make

exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.o: exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/flags.make
exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.o: /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/exe/FaceLandmarkVid/FaceLandmarkVid.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.o"
	cd /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/exe/FaceLandmarkVid && /usr/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.o -c /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/exe/FaceLandmarkVid/FaceLandmarkVid.cpp

exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.i"
	cd /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/exe/FaceLandmarkVid && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/exe/FaceLandmarkVid/FaceLandmarkVid.cpp > CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.i

exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.s"
	cd /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/exe/FaceLandmarkVid && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/exe/FaceLandmarkVid/FaceLandmarkVid.cpp -o CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.s

# Object files for target FaceLandmarkVid
FaceLandmarkVid_OBJECTS = \
"CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.o"

# External object files for target FaceLandmarkVid
FaceLandmarkVid_EXTERNAL_OBJECTS =

bin/FaceLandmarkVid: exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/FaceLandmarkVid.cpp.o
bin/FaceLandmarkVid: exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/build.make
bin/FaceLandmarkVid: lib/local/LandmarkDetector/libLandmarkDetector.a
bin/FaceLandmarkVid: lib/local/FaceAnalyser/libFaceAnalyser.a
bin/FaceLandmarkVid: lib/local/GazeAnalyser/libGazeAnalyser.a
bin/FaceLandmarkVid: lib/local/Utilities/libUtilities.a
bin/FaceLandmarkVid: /usr/local/lib/libopencv_objdetect.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_calib3d.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_features2d.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_highgui.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_flann.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_videoio.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_imgcodecs.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_imgproc.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libopencv_core.so.4.1.0
bin/FaceLandmarkVid: /usr/local/lib/libdlib.a
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libnsl.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libSM.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libICE.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libX11.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libXext.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libpng.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libz.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libjpeg.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libopenblas.so
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
bin/FaceLandmarkVid: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
bin/FaceLandmarkVid: exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/FaceLandmarkVid"
	cd /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/exe/FaceLandmarkVid && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FaceLandmarkVid.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/build: bin/FaceLandmarkVid

.PHONY : exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/build

exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/clean:
	cd /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/exe/FaceLandmarkVid && $(CMAKE_COMMAND) -P CMakeFiles/FaceLandmarkVid.dir/cmake_clean.cmake
.PHONY : exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/clean

exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/depend:
	cd /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/config/openface /home/marianna/Documentos/gitlab/libras/FacialActionLibras/src/features/openface/exe/FaceLandmarkVid /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/exe/FaceLandmarkVid /home/marianna/Documentos/gitlab/libras/FacialActionLibras/models/openface/build/exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : exe/FaceLandmarkVid/CMakeFiles/FaceLandmarkVid.dir/depend
