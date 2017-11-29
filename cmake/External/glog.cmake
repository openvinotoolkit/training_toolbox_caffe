# glog depends on gflags
include("cmake/External/gflags.cmake")

if (NOT __GLOG_INCLUDED)
  set(__GLOG_INCLUDED TRUE)

  # try the system-wide glog first
  find_package(Glog)
  if (GLOG_FOUND)
      set(GLOG_EXTERNAL FALSE)
  else()
    # fetch and build glog from github

    # build directory
    set(glog_PREFIX ${CMAKE_BINARY_DIR}/external/glog-prefix)
    # install directory
    set(glog_INSTALL ${CMAKE_BINARY_DIR}/external/glog-install)

    # we build glog statically, but want to link it into the caffe shared library
    # this requires position-independent code
    if (UNIX)
      set(GLOG_EXTRA_COMPILER_FLAGS "-fPIC")
    endif()

    set(GLOG_VERSION_MAJOR 0)
    set(GLOG_VERSION_MINOR 3)
    set(GLOG_VERSION_PATCH 5)
    set(GLOG_VERSION ${GLOG_VERSION_MAJOR}.${GLOG_VERSION_MINOR}.${GLOG_VERSION_PATCH})

    set(GLOG_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})
    set(GLOG_C_FLAGS ${CMAKE_C_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})

    # depend on gflags if we're also building it
    if (GFLAGS_EXTERNAL)
      set(GLOG_DEPENDS gflags)
    endif()

    set(GLOG_GIT_REPOSITORY "https://github.com/google/glog")
    set(GLOG_GIT_TAG "v${GLOG_VERSION}")

    message(STATUS "Will fetch Glog ${GLOG_GIT_TAG} from GIT at build stage...")

if(NOT WIN32)
    ExternalProject_Add(glog
      DEPENDS ${GLOG_DEPENDS}
      PREFIX ${glog_PREFIX}
      GIT_REPOSITORY ${GLOG_GIT_REPOSITORY}
      GIT_TAG ${GLOG_GIT_TAG}
      UPDATE_COMMAND ""
      INSTALL_DIR ${gflags_INSTALL}
      PATCH_COMMAND autoreconf -i ${glog_PREFIX}/src/glog
      CONFIGURE_COMMAND env "CFLAGS=${GLOG_C_FLAGS}" "CXXFLAGS=${GLOG_CXX_FLAGS}" ${glog_PREFIX}/src/glog/configure --prefix=${glog_INSTALL} --enable-shared=no --enable-static=yes --with-gflags=${GFLAGS_LIBRARY_DIRS}/..
      LOG_DOWNLOAD 1
      LOG_CONFIGURE 1
      LOG_INSTALL 1
      )
else()
    ExternalProject_Add(glog
      DEPENDS ${GLOG_DEPENDS}
      PREFIX ${glog_PREFIX}
      GIT_REPOSITORY ${GLOG_GIT_REPOSITORY}
      GIT_TAG ${GLOG_GIT_TAG}
      UPDATE_COMMAND ""
      INSTALL_DIR ${gflags_INSTALL}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_INSTALL_PREFIX=${glog_INSTALL}
                 -DWITH_GFLAGS=OFF
                 -DBUILD_SHARED_LIBS=OFF
                 -DBUILD_STATIC_LIBS=ON
                 -DBUILD_PACKAGING=OFF
                 -DBUILD_TESTING=OFF
                 -DINSTALL_HEADERS=ON
                 -DCMAKE_C_FLAGS=${GLOG_C_FLAGS}
                 -DCMAKE_CXX_FLAGS=${GLOG_CXX_FLAGS}
      LOG_DOWNLOAD 1
      LOG_CONFIGURE 1
      LOG_INSTALL 1
      )
endif()

    set(GLOG_EXTERNAL TRUE)
    set(GLOG_FOUND TRUE)
    set(GLOG_INCLUDE_DIRS ${glog_INSTALL}/include)
    set(GLOG_LIBRARY_DIRS ${glog_INSTALL}/lib)
if(WIN32)
        set(GLOG_LIBRARIES ${GFLAGS_LIBRARIES} ${glog_INSTALL}/lib/glog.lib)
else()
        set(GLOG_LIBRARIES ${GFLAGS_LIBRARIES} ${glog_INSTALL}/lib/libglog.a)
endif()

    list(APPEND external_project_dependencies glog)
  endif()

endif()
