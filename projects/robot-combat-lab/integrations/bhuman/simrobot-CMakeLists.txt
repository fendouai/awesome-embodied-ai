cmake_minimum_required(VERSION 3.16)
if(CMAKE_HOST_UNIX)
  set(CMAKE_CXX_COMPILER clang++)
endif()

project(SimRobot LANGUAGES CXX)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL arm64 OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL aarch64)
  set(X86 false)
  set(ARM true)
  set(ARCH "Arm64")
else()
  set(X86 true)
  set(ARM false)
  set(ARCH "")
endif()

set(WINDOWS false)
set(MACOS false)
set(LINUX false)
if(APPLE)
  set(MACOS true)
  set(PLATFORM "macOS")
  if(NOT ARM)
    message(FATAL_ERROR "macOS/x86 is not supported")
  endif()
  set(ARCH "")
  add_definitions(-DMACOS)
elseif(WIN32)
  set(WINDOWS true)
  set(PLATFORM "Windows")
  add_definitions(-DWINDOWS)
else()
  set(LINUX true)
  set(PLATFORM "Linux")
  add_definitions(-DLINUX)
endif()

set(CMAKE_CONFIGURATION_TYPES Debug Develop Release CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SIMROBOT_PREFIX "${PROJECT_SOURCE_DIR}/../..")
set(OUTPUT_PREFIX "${PROJECT_SOURCE_DIR}/../..")

if(MACOS)
  set(CMAKE_SUPPRESS_REGENERATION ON)
  add_definitions(
      -DFIX_MACOS_EDIT_MENU
      -DFIX_MACOS_PINCH_SCALE_RELATIVE_BUG
      -DFIX_MACOS_TOOLBAR_WIDGET_NOT_CLOSING)
  set(CMAKE_XCODE_ATTRIBUTE_ALWAYS_SEARCH_USER_PATHS "NO")
  set(CMAKE_XCODE_ATTRIBUTE_USE_HEADERMAP "NO")
  set(CMAKE_SKIP_RPATH TRUE)
  find_library(APP_KIT_FRAMEWORK AppKit)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER CMake)
set_property(GLOBAL PROPERTY AUTOGEN_SOURCE_GROUP ".Generated Files")
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT SimRobot)

foreach(config DEBUG DEVELOP RELEASE)
  set(CMAKE_CXX_FLAGS_${config} "")
  foreach(type EXE MODULE SHARED STATIC)
    set(CMAKE_${type}_LINKER_FLAGS_${config} "")
  endforeach()
endforeach()

if(MSVC)
  add_compile_options(/W3 /GF /GS- /MP /EHsc /permissive- /Zc:preprocessor /Zc:__cplusplus /wd5105 /w34100 /w34189)
  add_definitions(-DNOMINMAX -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)

  execute_process(COMMAND bash -c "cat /proc/cpuinfo" OUTPUT_VARIABLE _cpuinfo)
  string(FIND "${_cpuinfo}" "avx2" _has_avx2)
  string(FIND "${_cpuinfo}" "avx" _has_avx)
  if(NOT ${_has_avx2} EQUAL -1)
    add_compile_options(/arch:AVX2)
    set(AVX true)
  elseif(NOT ${_has_avx} EQUAL -1)
    add_compile_options(/arch:AVX)
    set(AVX true)
  else()
    set(AVX false)
  endif()
else()
  add_compile_options(-pipe
      -Wall
      -Wextra
      -Wconversion
      -Wno-unknown-warning-option
      -Wno-sign-conversion
      -Wno-implicit-int-float-conversion
      -Wunreachable-code
      -Wsign-compare)
  if(X86)
    add_compile_options(-march=native)
  endif()
  if(NOT MACOS)
    add_link_options(-fuse-ld=mold)
  endif()
endif()

set(_is_msvc "$<CXX_COMPILER_ID:MSVC>")
set(_is_clang "$<NOT:$<CXX_COMPILER_ID:MSVC>>")
set(_not_mac "$<NOT:$<PLATFORM_ID:Darwin>>")

add_library(Flags::DebugInDevelop INTERFACE IMPORTED)
target_compile_definitions(Flags::DebugInDevelop INTERFACE
    $<$<CONFIG:Release>:NDEBUG>)
target_compile_options(Flags::DebugInDevelop INTERFACE
    $<${_is_msvc}:$<$<CONFIG:Debug>:/ZI>>
    $<${_is_msvc}:$<$<CONFIG:Develop>:/Zi>>
    $<${_is_msvc}:$<$<NOT:$<CONFIG:Release>>:/Od>>
    $<${_is_msvc}:$<$<CONFIG:Develop>:/Ob2>>
    $<${_is_msvc}:$<$<CONFIG:Develop>:/Oi>>
    $<${_is_msvc}:$<$<CONFIG:Develop>:/Ot>>
    $<${_is_msvc}:$<$<CONFIG:Develop>:/Oy>>
    $<${_is_msvc}:$<$<CONFIG:Release>:/Ox>>
    $<${_is_msvc}:$<$<NOT:$<CONFIG:Debug>>:/GT>>
    $<${_is_clang}:$<$<NOT:$<CONFIG:Release>>:-g>>
    $<${_is_clang}:$<$<CONFIG:Release>:-O3>>
    $<${_is_clang}:$<$<CONFIG:Release>:-fomit-frame-pointer>>
    $<${_is_clang}:$<$<CONFIG:Release>:-Wno-unused-parameter>>
    $<${_is_clang}:$<$<CONFIG:Release>:-Wno-unused-variable>>
    $<${_is_clang}:$<$<CONFIG:Release>:-Wno-unused-lambda-capture>>)
target_link_options(Flags::DebugInDevelop INTERFACE
    $<${_is_msvc}:$<$<NOT:$<CONFIG:Release>>:/DEBUG>>
    $<${_is_msvc}:$<$<NOT:$<CONFIG:Debug>>:/INCREMENTAL:NO>>
    $<${_is_msvc}:$<$<CONFIG:Develop>:/SAFESEH:NO>>
    $<$<AND:${_not_mac},${_is_clang}>:$<$<CONFIG:Release>:-s>>)

add_library(Flags::Default INTERFACE IMPORTED)
target_compile_definitions(Flags::Default INTERFACE
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>)
target_compile_options(Flags::Default INTERFACE
    $<${_is_msvc}:$<$<CONFIG:Debug>:/ZI>>
    $<${_is_msvc}:$<$<CONFIG:Debug>:/Od>>
    $<${_is_msvc}:$<$<NOT:$<CONFIG:Debug>>:/Ox>>
    $<${_is_msvc}:$<$<NOT:$<CONFIG:Debug>>:/GT>>
    $<${_is_clang}:$<$<CONFIG:Debug>:-g>>
    $<${_is_clang}:$<$<NOT:$<CONFIG:Debug>>:-O3>>
    $<${_is_clang}:$<$<NOT:$<CONFIG:Debug>>:-fomit-frame-pointer>>
    $<${_is_clang}:$<$<NOT:$<CONFIG:Debug>>:-Wno-unused-parameter>>
    $<${_is_clang}:$<$<NOT:$<CONFIG:Debug>>:-Wno-unused-variable>>
    $<${_is_clang}:$<$<NOT:$<CONFIG:Debug>>:-Wno-unused-lambda-capture>>)
target_link_options(Flags::Default INTERFACE
    $<${_is_msvc}:$<$<CONFIG:Debug>:/DEBUG>>
    $<${_is_msvc}:$<$<NOT:$<CONFIG:Debug>>:/INCREMENTAL:NO>>
    $<$<AND:${_not_mac},${_is_clang}>:$<$<CONFIG:Release>:-s>>)

set(SIMROBOT_CONTROLLERS Factory SimpleVehicle Soccer)
if(MACOS)
  set(SIMROBOT_LIBRARY_DIR "${OUTPUT_PREFIX}/Build/${PLATFORM}/SimRobot/$<CONFIG>/SimRobot.app/Contents/lib")
else()
  set(SIMROBOT_LIBRARY_DIR "${OUTPUT_PREFIX}/Build/${PLATFORM}/SimRobot/$<CONFIG>")
endif()

include("../CMake/Box2D.cmake")
include("../CMake/Eigen.cmake")
include("../CMake/MuJoCo.cmake")
include("../CMake/Qt6.cmake")
include("../CMake/SimRobot.cmake")
include("../CMake/SimRobotCommon.cmake")
include("../CMake/SimRobotCore3.cmake")
include("../CMake/SimRobotCore2D.cmake")
include("../CMake/SimRobotEditor.cmake")
include("../CMake/SimpleVehicle.cmake")
include("../CMake/Factory.cmake")
include("../CMake/Soccer.cmake")

source_group(".PCH" REGULAR_EXPRESSION ".*[ch]xx$")
source_group(".Visualizers" REGULAR_EXPRESSION ".*natvis$")
source_group(".CMake Rules" REGULAR_EXPRESSION ".*rule$")
