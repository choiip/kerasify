cmake_minimum_required(VERSION 3.9)

# common options

enable_language(CXX)

# set(CMAKE_C_COMPILER   "/usr/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -W -Wall -Wextra -Wpedantic \
      -Waggressive-loop-optimizations \
      -Wcast-align -Wcast-qual \
      -Wdouble-promotion -Wduplicated-branches -Wduplicated-cond \
      -Wfloat-equal -Wformat=2 -Wformat-signedness -Wframe-larger-than=32768 \
      -Wlogical-op \
      -Wnull-dereference \
      -Wodr -Wold-style-cast \
      -Wshadow=local -Wshift-overflow=2 -Wstrict-aliasing=2 -Wsuggest-final-methods -Wsuggest-final-types -Wsync-nand \
      -Wtrampolines \
      -Wuseless-cast -Wno-unused-but-set-parameter \
      -Wwrite-strings"
      CACHE INTERNAL ""
  )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -Weverything \
      -Wno-class-varargs -Wno-padded \
      -Wc++17-compat \
      -Wno-c++98-compat -Wno-c++98-compat-pedantic"
      # -Wno-c++14-extensions
      # -Wno-c++17-extensions
      # -Wno-switch-enum -Wno-unused-macros
      CACHE INTERNAL ""
  )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og"
      CACHE INTERNAL ""
  )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -Oz \
      -fno-exceptions -fno-rtti \
      -fno-stack-protector -fomit-frame-pointer \
      -ffunction-sections -fdata-sections \
      -fno-math-errno -fno-unroll-loops -fmerge-all-constants -fno-ident"
      CACHE INTERNAL ""
  )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
      -O3 -ffast-math -funroll-loops \
      -fno-exceptions -fno-rtti"
      CACHE INTERNAL ""
  )
endif()

# project configuration

project(kerasify)

include_directories(SYSTEM "/usr/include/c++/7")
include_directories(kerasify PRIVATE "./include")

add_executable(kerasify "")
target_sources(kerasify PRIVATE
    src/layer/activation.cpp
    src/layer/conv2d.cpp
    src/layer/dense.cpp
    src/layer/elu.cpp
    src/layer/embedding.cpp
    src/layer/flatten.cpp
    src/layer/lstm.cpp
    src/layer/maxpooling2d.cpp
    src/model.cpp
    src/utility.cpp
)

# find_package(OpenCV REQUIRED)
# find_package(Threads REQUIRED)
# target_link_libraries(kerasify PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

