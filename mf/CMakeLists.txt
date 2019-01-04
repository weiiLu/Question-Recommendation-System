include_directories(include)

file(GLOB mf_src "./*.cpp")
file(GLOB mf_head "./*.h")


if(WIN32)
	set(CMAKE_CXX_FLAGS "/W4 /nologo /O2 /EHsc /arch:AVX /openmp")
else()
	set(CMAKE_CXX_FLAGS "-Wall -O3 -pthread -std=c++0x  -mavx -fopenmp")
endif()
add_definitions(-DUSESSE -DUSEAVX -DUSEOMP -D_CRT_SECURE_NO_DEPRECATE)

add_library(mf SHARED ${mf_src} ${mf_head})

if(WIN32)
	install(TARGETS mf
		RUNTIME DESTINATION ./)
endif()