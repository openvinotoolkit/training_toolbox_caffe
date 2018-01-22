if (NOT __PROTOBUF_INCLUDED) # guard against multiple includes
  set(__PROTOBUF_INCLUDED TRUE)

  # use the system-wide gflags if present
  find_package(ProtoBuf)
  if (PROTOBUF_FOUND)
    set(PROTOBUF_EXTERNAL FALSE)
  else()
    # build directory
    set(protobuf_PREFIX ${CMAKE_BINARY_DIR}/external/protobuf-prefix)
    # install directory
    set(protobuf_INSTALL ${CMAKE_BINARY_DIR}/external/protobuf-install)

    # we build protobuf statically, but want to link it into the caffe shared library
    # this requires position-independent code
    if (UNIX)
        set(GFLAGS_EXTRA_COMPILER_FLAGS "-fPIC")
    endif()

    set(PROTOBUF_VERSION_MAJOR 3)
    set(PROTOBUF_VERSION_MINOR 5)
    set(PROTOBUF_VERSION_PATCH 0)
    set(PROTOBUF_VERSION ${PROTOBUF_VERSION_MAJOR}.${PROTOBUF_VERSION_MINOR}.${PROTOBUF_VERSION_PATCH})

    set(PROTOBUF_GIT_REPOSITORY "https://github.com/google/protobuf.git")
    set(PROTOBUF_GIT_TAG "v${PROTOBUF_VERSION}")

    set(GFLAGS_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${GFLAGS_EXTRA_COMPILER_FLAGS})
    set(GFLAGS_C_FLAGS ${CMAKE_C_FLAGS} ${GFLAGS_EXTRA_COMPILER_FLAGS})

    message(STATUS "Will fetch Protobuf ${PROTOBUF_VERSION} from GIT at build stage...")

    ExternalProject_Add(protobuf
      PREFIX ${protobuf_PREFIX}
      GIT_REPOSITORY ${PROTOBUF_GIT_REPOSITORY}
      GIT_TAG ${PROTOBUF_GIT_TAG}
      UPDATE_COMMAND ""
      INSTALL_DIR ${protobuf_INSTALL}
      SOURCE_SUBDIR cmake
      CMAKE_ARGS -Dprotobuf_VERBOSE=ON
                 -DCMAKE_INSTALL_PREFIX=${protobuf_INSTALL}
                 -DBUILD_SHARED_LIBS=OFF
                 -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
                 -Dprotobuf_BUILD_TESTS=OFF
                 -BUILD_CONFIG_TESTS=OFF
                 -DCMAKE_C_FLAGS=${GFLAGS_C_FLAGS}
                 -DCMAKE_CXX_FLAGS=${GFLAGS_CXX_FLAGS}
      LOG_DOWNLOAD 1
      LOG_INSTALL 1
      )

    set(PROTOBUF_EXTERNAL TRUE)
    set(PROTOBUF_FOUND TRUE)
    set(PROTOBUF_INCLUDE_DIR ${protobuf_INSTALL}/include)
    set(PROTOBUF_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIR})
    set(PROTOBUF_LIBRARY_DIRS ${protobuf_INSTALL}/lib)
if(WIN32)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
        set(PROTOBUFNAME libprotobuf.lib)
    endif()
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        set(PROTOBUFNAME libprotobufd.lib)
    endif()
    set(PROTOBUF_LIBRARIES ${protobuf_INSTALL}/lib/${PROTOBUFNAME} ${CMAKE_THREAD_LIBS_INIT})
    set(PROTOBUF_PROTOC_EXECUTABLE ${protobuf_INSTALL}/bin/protoc.exe)
else()
        set(PROTOBUF_LIBRARIES ${protobuf_INSTALL}/lib/libprotobuf.a ${CMAKE_THREAD_LIBS_INIT})
endif()

    list(APPEND external_project_dependencies protobuf)
  endif()

endif()

if(NOT WIN32)
    # As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package
    # and should be installed separately as in: sudo apt-get install protobuf-compiler
    if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
        message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Could not find PROTOBUF Compiler")
    endif()
endif()

if(PROTOBUF_FOUND)
  set(PROTOBUF_VERSION "${PROTOBUF_VERSION_MAJOR}.${PROTOBUF_VERSION_MINOR}.${PROTOBUF_VERSION_PATCH}")
  unset(GOOGLE_PROTOBUF_VERSION)
endif()

# place where to generate protobuf sources
set(proto_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/proto")
include_directories("${PROJECT_BINARY_DIR}/include")

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

################################################################################################
# Modification of standard 'protobuf_generate_cpp()' with output dir parameter and python support
# Usage:
#   caffe_protobuf_generate_cpp_py(<output_dir> <srcs_var> <hdrs_var> <python_var> <proto_files>)
function(caffe_protobuf_generate_cpp_py output_dir srcs_var hdrs_var python_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: caffe_protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${output_dir}/${fil_we}_pb2.py")

    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}.pb.cc"
             "${output_dir}/${fil_we}.pb.h"
             "${output_dir}/${fil_we}_pb2.py"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out    ${output_dir} ${_protoc_include} ${abs_fil}
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${output_dir} ${_protoc_include} ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()
