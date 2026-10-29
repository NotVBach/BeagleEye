set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7)  # cpuinfo only builds its ARM code for armv*

# gcc-12 matches the board's Debian 12 libstdc++; newer g++ emits symbols the board lacks.
set(CMAKE_C_COMPILER   arm-linux-gnueabihf-gcc-12)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++-12)

if(NOT CMAKE_SYSROOT)
    set(CMAKE_SYSROOT $ENV{HOME}/bbb)
endif()

# Rebuild gcc's include list without /usr/arm-linux-gnueabihf/include: its glibc headers are newer than the board's and clash with the sysroot's.
set(_cross /usr/arm-linux-gnueabihf/include/c++/12)
set(_sys   "-isystem /usr/lib/gcc-cross/arm-linux-gnueabihf/12/include -isystem ${CMAKE_SYSROOT}/usr/include/arm-linux-gnueabihf -isystem ${CMAKE_SYSROOT}/usr/include")
set(_arch  "-march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=hard -nostdinc")
set(CMAKE_C_FLAGS_INIT   "${_arch} ${_sys}")
set(CMAKE_CXX_FLAGS_INIT "${_arch} -isystem ${_cross} -isystem ${_cross}/arm-linux-gnueabihf -isystem ${_cross}/backward ${_sys}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Debian picks liblapack/libblas through /etc/alternatives, which the sysroot can't follow; point ld at the real dirs.
set(_lib ${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,-rpath-link,${_lib}/lapack -Wl,-rpath-link,${_lib}/blas")
