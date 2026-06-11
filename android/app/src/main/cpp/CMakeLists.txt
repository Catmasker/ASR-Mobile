cmake_minimum_required(VERSION 3.22.1)
project(asr_mobile)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/whisper.cpp)

add_library(asr_mobile SHARED whisper_jni.cpp)

target_include_directories(asr_mobile PRIVATE
    third_party/whisper.cpp/include
    third_party/whisper.cpp/ggml/include
)

target_link_libraries(asr_mobile whisper android log)
