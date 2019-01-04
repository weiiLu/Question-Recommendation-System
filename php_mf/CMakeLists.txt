find_path(php_path NAMES php_ini.h)
#set(php_path "/usr/local/php-7.0.20/include/php/main/")
if(${php_path} STREQUAL php_path_NOTFOUND)
  message(FATAL_ERROR "cannot find the path contain the php_ini.h, please mannual set to the php_src/main")
endif()

include_directories(${php_path} ${php_path}/.. ${php_path}/../TSRM ${php_path}/../Zend)

if(WIN32)
  add_definitions(-DZEND_WIN32 -DPHP_WIN32 -DPHP_EXTENSION -DZTS -DCOMPILE_DL_MF=1 -DHAVE_MF=1 -DZEND_DEBUG=$<CONFIG:DEBUG>)
else()
  add_definitions(-DPHP_EXTENSION -DZTS -DCOMPILE_DL_MF=1 -DHAVE_MF=1 -DZEND_DEBUG=$<CONFIG:DEBUG>)
endif()

set( cur_files mfWarp.cpp php_mf.c mfWarp.h php_mf.h)
add_library(php_mf SHARED ${cur_files})

if(WIN32)
  set(mf_lib "${CMAKE_BINARY_DIR}/bin/Release/mf.lib")
  set(mf_lib_debug "${CMAKE_BINARY_DIR}/bin/Debug/mf.lib")
else()
  set(mf_lib "${CMAKE_BINARY_DIR}/bin/libmf.so")
  set(mf_lib_debug "${CMAKE_BINARY_DIR}/bin/libmf.so")
endif()


if(WIN32)
	find_library(php_lib NAMES php)
	target_link_libraries(php_mf ${php_lib} $<$<CONFIG:Release>:${mf_lib}>$<$<CONFIG:Debug>:${mf_lib_debug}>)
else()
	target_link_libraries(php_mf $<$<CONFIG:Release>:${mf_lib}>$<$<CONFIG:Debug>:${mf_lib_debug}>)
endif()

if(WIN32)
install(TARGETS php_mf
	RUNTIME DESTINATION ext)
#else()
	#install(TARGETS php_lib_mf
	#LIBRARY DESTINATION ./)
endif()
