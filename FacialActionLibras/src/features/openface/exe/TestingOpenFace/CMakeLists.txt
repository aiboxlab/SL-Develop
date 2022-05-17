# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(TestingOpenFace TestingOpenFace.cpp)

target_link_libraries(TestingOpenFace LandmarkDetector)
target_link_libraries(TestingOpenFace FaceAnalyser)
target_link_libraries(TestingOpenFace GazeAnalyser)
target_link_libraries(TestingOpenFace Utilities)

install (TARGETS TestingOpenFace DESTINATION bin)