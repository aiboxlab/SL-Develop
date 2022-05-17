# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(LandmarkVid2Way LandmarkVid2Way.cpp)

target_link_libraries(LandmarkVid2Way LandmarkDetector)
target_link_libraries(LandmarkVid2Way FaceAnalyser)
target_link_libraries(LandmarkVid2Way GazeAnalyser)
target_link_libraries(LandmarkVid2Way Utilities)

install (TARGETS LandmarkVid2Way DESTINATION bin)
