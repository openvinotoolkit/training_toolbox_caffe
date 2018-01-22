if (NOT __GFLAGS_INCLUDED) # guard against multiple includes
  set(__GFLAGS_INCLUDED TRUE)

  # use the system-wide gflags if present
  find_package(GFlags)
  if (GFLAGS_FOUND)
    set(GFLAGS_EXTERNAL FALSE)
  else()
    # gflags will use pthreads if it's available in the system, so we must link with it
    find_package(Threads)

    # build directory
    set(gflags_PREFIX ${CMAKE_BINARY_DIR}/external/gflags-prefix)
    # install directory
    set(gflags_INSTALL ${CMAKE_BINARY_DIR}/external/gflags-install)

    # we build gflags statically, but want to link it into the caffe shared library
    # this requires position-independent code
    if (UNIX)
        set(GFLAGS_EXTRA_COMPILER_FLAGS "-fPIC")
    endif()

    set(GFLAGS_VERSION_MAJOR 2)
    set(GFLAGS_VERSION_MINOR 2)
    set(GFLAGS_VERSION_PATCH 1)
    set(GFLAGS_VERSION ${GFLAGS_VERSION_MAJOR}.${GFLAGS_VERSION_MINOR}.${GFLAGS_VERSION_PATCH})

    set(GFLAGS_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${GFLAGS_EXTRA_COMPILER_FLAGS})

    set(GFLAGS_GIT_REPOSITORY "https://github.com/gflags/gflags.git")
    set(GFLAGS_GIT_TAG "v${GFLAGS_VERSION}")

    set(GFLAGS_IS_A_DLL ON)

    if(${GFLAGS_IS_A_DLL})
        set(_BUILD_SHARED_LIBS ON)
        set(_BUILD_STATIC_LIBS OFF)
    else()
        set(_BUILD_SHARED_LIBS OFF)
        set(_BUILD_STATIC_LIBS ON)
    endif()

    message(STATUS "Will fetch GFlags ${GFLAGS_GIT_TAG} from GIT at build stage...")

    ExternalProject_Add(
      gflags
      PREFIX ${gflags_PREFIX}
      GIT_REPOSITORY ${GFLAGS_GIT_REPOSITORY}
      GIT_TAG ${GFLAGS_GIT_TAG}
      UPDATE_COMMAND ""
      INSTALL_DIR ${gflags_INSTALL}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_INSTALL_PREFIX=${gflags_INSTALL}
                 -DBUILD_SHARED_LIBS=${_BUILD_SHARED_LIBS}
                 -DGFLAGS_IS_A_DLL=${GFLAGS_IS_A_DLL}
                 -DBUILD_STATIC_LIBS=${_BUILD_STATIC_LIBS}
                 -DBUILD_gflags_LIB=ON
                 -DBUILD_gflags_nothreads_LIB=OFF
                 -DBUILD_PACKAGING=OFF
                 -DBUILD_TESTING=OFF
                 -DINSTALL_HEADERS=ON
                 -DCMAKE_CXX_FLAGS=${GFLAGS_CXX_FLAGS}
      GIT_PROGRESS 1
      LOG_DOWNLOAD 1
      LOG_BUILD 1
      LOG_INSTALL 1
    )

    set(GFLAGS_EXTERNAL TRUE)
    set(GFLAGS_FOUND TRUE)
    set(GFLAGS_INCLUDE_DIRS ${gflags_INSTALL}/include)
    set(GFLAGS_LIBRARY_DIRS ${gflags_INSTALL}/lib)
if(WIN32)
        if(${GFLAGS_IS_A_DLL})
            set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/gflags.lib ${CMAKE_THREAD_LIBS_INIT})
        else()
            set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/gflags_static.lib shlwapi.lib ${CMAKE_THREAD_LIBS_INIT})
        endif()
else()
        set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/libgflags.a ${CMAKE_THREAD_LIBS_INIT})
endif()
    unset(_BUILD_SHARED_LIBS)
    unset(_BUILD_STATIC_LIBS)
    list(APPEND external_project_dependencies gflags)
  endif()

endif()
