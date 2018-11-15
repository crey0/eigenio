CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

project(eigenio)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  )

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN "*.h")

if(${BUILD_TEST})
  include(CTest)
  enable_testing()

  set(TEST_SOURCES
    eigenio.cpp
    )

  set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)

  add_library(CatchMain OBJECT ${TEST_DIR}/catch_main.cpp)

  foreach(file_name ${TEST_SOURCES})
    string(REPLACE ".cpp" "" name "${TEST_SOURCES}")
    string(REPLACE "/" "+" name "${name}")
    set(name "test+${name}")
    
    add_executable(${name} ${TEST_DIR}/${file_name} $<TARGET_OBJECTS:CatchMain>)
    target_include_directories(${name} PRIVATE ${TEST_DIR})
    
    add_test(NAME ${name} COMMAND ${name})
  endforeach()

endif()
