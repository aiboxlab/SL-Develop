# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(ActionUnitVid2Way ActionUnitVid2Way.cpp)

target_link_libraries(ActionUnitVid2Way LandmarkDetector)
target_link_libraries(ActionUnitVid2Way FaceAnalyser)
target_link_libraries(ActionUnitVid2Way GazeAnalyser)
target_link_libraries(ActionUnitVid2Way Utilities)

install (TARGETS ActionUnitVid2Way DESTINATION bin)
