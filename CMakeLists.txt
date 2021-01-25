cmake_minimum_required(VERSION 3.15)

project(opencv-cpp-project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(.)	

add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DWIN32_LEAN_AND_MEAN)
add_definitions(-DVC_EXTRALEAN)
add_definitions(-DNOMINMAX)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-D_WIN32_WINNT=0x0A00)
add_definitions(-D_ENABLE_ATOMIC_ALIGNMENT_FIX)
  
set(OpenCV_DIR "opencv")
find_package(OpenCV REQUIRED)

#########################################################################
file(GLOB TARGET_HEADERS "*.h")

add_executable(opencv-cpp-project
  ${TARGET_HEADERS}
  main.cc
  ioFunctions.cc
  onGridOperations.cc
  getWrongPoints.cc	
  matchingCostFunctions.cc
  estimationSGMmethod.cc
  smallSupportFunctions.cc
  usedStructnFunctions.cc
  Ladimo_Stereo_Matcher.cc
  preGridOperations.cc
  postProcessingFunctions.cc
)

target_include_directories(opencv-cpp-project PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(opencv-cpp-project
  ${OpenCV_LIBRARIES}
  )
#########################################################################

#########################################################################
# Section for testing functions
file(GLOB TARGET_HEADERS "*.h")

add_executable(derivative-tester
  ${TARGET_HEADERS}
  derivative_tester_main.cc
  ioFunctions.cc
  onGridOperations.cc
  onGridOperations_tester.cc
  getWrongPoints.cc	
  matchingCostFunctions.cc
  smallSupportFunctions.cc
  testFunctions.cc
  usedStructnFunctions.cc
  estimationSGMmethod.cc
)

target_include_directories(derivative-tester PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(derivative-tester
  ${OpenCV_LIBRARIES}
  )
#########################################################################

#########################################################################
# Section for testing functions
file(GLOB TARGET_HEADERS "*.h")

add_executable(post-processing-tester
  ${TARGET_HEADERS}
  post_processing_test_main.cc
  ioFunctions.cc
  onGridOperations.cc
  onGridOperations_tester.cc
  getWrongPoints.cc	
  matchingCostFunctions.cc
  smallSupportFunctions.cc
  testFunctions.cc
  usedStructnFunctions.cc
  estimationSGMmethod.cc
)

target_include_directories(post-processing-tester PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(post-processing-tester
  ${OpenCV_LIBRARIES}
  )
#########################################################################

#########################################################################
# Section for testing functions
file(GLOB TARGET_HEADERS "*.h")

add_executable(tester
  ${TARGET_HEADERS}
  tester_main.cc
  ioFunctions.cc
  smallSupportFunctions.cc
  usedStructnFunctions.cc
  preWorkingLadimoImages.cc
  getWrongPoints.cc	
  matchingCostFunctions.cc
  preGridOperations.cc
  onGridOperations.cc
  postProcessingFunctions.cc
  realLadimoGridEstimations.cc
  testFunctions.cc
)

target_include_directories(tester PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(tester
  ${OpenCV_LIBRARIES}
  )
#########################################################################

#########################################################################
# Section for testing opencv-based built-in method
file(GLOB TARGET_HEADERS "opencv-builtin-method/*.h")

set(SOURCE_FILES
  opencv-builtin-method/main_opencv_builtin_method.cc
)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/opencv-builtin-method
)
add_executable(opencv-builtin-sgbm-method
  ${TARGET_HEADERS}
  ${SOURCE_FILES}
)
target_include_directories(opencv-builtin-sgbm-method PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )
target_link_libraries(opencv-builtin-sgbm-method
  ${OpenCV_LIBRARIES}
  )
#########################################################################

#########################################################################
# Section for simulated grid method
# file(GLOB TARGET_HEADERS "*.h")

# add_executable(simulated-grid-method
  # ${TARGET_HEADERS}
  # main_simulated_grid_method.cc
  # ioFunctions.cc
  # onGridOperations.cc
  # getWrongPoints.cc	
  # matchingCostFunctions.cc
  # estimationSGMmethod.cc
  # smallSupportFunctions.cc
  # usedStructnFunctions.cc
  # Ladimo_Stereo_Matcher.cc
  # preGridOperations.cc
  # postProcessingFunctions.cc
  # preWorkingLadimoImages.cc
# )

# target_include_directories(simulated-grid-method PRIVATE
  # ${OpenCV_INCLUDE_DIRS}
  # )

# target_link_libraries(simulated-grid-method
  # ${OpenCV_LIBRARIES}
  # )
#########################################################################

#########################################################################
# Section for sgm-based method
file(GLOB TARGET_HEADERS "*.h")

add_executable(sgm-based-method
  ${TARGET_HEADERS}
  main_sgm_based_method.cc
  ioFunctions.cc
  onGridOperations.cc
  getWrongPoints.cc	
  matchingCostFunctions.cc
  estimationSGMmethod.cc
  smallSupportFunctions.cc
  usedStructnFunctions.cc
  preGridOperations.cc
  postProcessingFunctions.cc
  preWorkingLadimoImages.cc
  realLadimoGridEstimations.cc
  testFunctions.cc
)

target_include_directories(sgm-based-method PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(sgm-based-method
  ${OpenCV_LIBRARIES}
  )
#########################################################################

#########################################################################
# Section for simulated grid method
file(GLOB TARGET_HEADERS 
  "simulated-grid-method/*.h"
  "common-files/*.h"
)
set(COMMON_FILES
  common-files/io_functions.cc
  common-files/common_structures.cc
  common-files/image_processing_function.cc
)
set(SOURCE_FILES
  simulated-grid-method/main_simulated_grid_method.cc
  simulated-grid-method/estimation_functions_simulated_grid.cc
  simulated-grid-method/post_processing_simulated_grid.cc
  simulated-grid-method/pre_processing_simulated_grid.cc
  simulated-grid-method/support_functions_simulated_grid.cc
  simulated-grid-method/support_structures_simulated_grid.cc
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/simulated-grid-method
  ${CMAKE_CURRENT_SOURCE_DIR}/common-files
)

add_executable(simulated-grid-method
  ${TARGET_HEADERS}
  ${SOURCE_FILES}
  ${COMMON_FILES}
)

target_include_directories(simulated-grid-method PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(simulated-grid-method
  ${OpenCV_LIBRARIES}
  )
#########################################################################

#########################################################################
# Section for LaDiMo grid method
file(GLOB TARGET_HEADERS 
  "ladimo-grid-method/*.h"
  "common-files/*.h"
)

set(COMMON_FILES
  common-files/io_functions.cc
  common-files/common_structures.cc
  common-files/image_processing_function.cc
)
set(SOURCE_FILES
  ladimo-grid-method/main_ladimo_grid_method.cc
  ladimo-grid-method/estimation_functions_ladimo_grid_method.cc
  ladimo-grid-method/support_functions_ladimo_grid_method.cc
  ladimo-grid-method/support_structures_ladimo_grid_method.cc
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/ladimo-grid-method
  ${CMAKE_CURRENT_SOURCE_DIR}/common-files
)
add_executable(ladimo-grid-method
  ${TARGET_HEADERS}
  ${SOURCE_FILES}
  ${COMMON_FILES}
)

target_include_directories(ladimo-grid-method PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )
target_link_libraries(ladimo-grid-method
  ${OpenCV_LIBRARIES}
  )
#########################################################################
get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

add_custom_command(TARGET opencv-cpp-project POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different $<$<CONFIG:Debug>:${__dll_dbg}> $<$<NOT:$<CONFIG:Debug>>:${__dll_release}> $<TARGET_FILE_DIR:opencv-cpp-project>)
