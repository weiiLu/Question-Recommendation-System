# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lw/桌面/phpmf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lw/桌面/phpmf/Release

# Include any dependencies generated for this target.
include mfTest/CMakeFiles/mfTest.dir/depend.make

# Include the progress variables for this target.
include mfTest/CMakeFiles/mfTest.dir/progress.make

# Include the compile flags for this target's objects.
include mfTest/CMakeFiles/mfTest.dir/flags.make

mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o: mfTest/CMakeFiles/mfTest.dir/flags.make
mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o: ../mfTest/mfTest.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/lw/桌面/phpmf/Release/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o"
	cd /home/lw/桌面/phpmf/Release/mfTest && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mfTest.dir/mfTest.cpp.o -c /home/lw/桌面/phpmf/mfTest/mfTest.cpp

mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mfTest.dir/mfTest.cpp.i"
	cd /home/lw/桌面/phpmf/Release/mfTest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/lw/桌面/phpmf/mfTest/mfTest.cpp > CMakeFiles/mfTest.dir/mfTest.cpp.i

mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mfTest.dir/mfTest.cpp.s"
	cd /home/lw/桌面/phpmf/Release/mfTest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/lw/桌面/phpmf/mfTest/mfTest.cpp -o CMakeFiles/mfTest.dir/mfTest.cpp.s

mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.requires:
.PHONY : mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.requires

mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.provides: mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.requires
	$(MAKE) -f mfTest/CMakeFiles/mfTest.dir/build.make mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.provides.build
.PHONY : mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.provides

mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.provides.build: mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o

# Object files for target mfTest
mfTest_OBJECTS = \
"CMakeFiles/mfTest.dir/mfTest.cpp.o"

# External object files for target mfTest
mfTest_EXTERNAL_OBJECTS =

bin/mfTest: mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o
bin/mfTest: mfTest/CMakeFiles/mfTest.dir/build.make
bin/mfTest: bin/libmf.so
bin/mfTest: mfTest/CMakeFiles/mfTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/mfTest"
	cd /home/lw/桌面/phpmf/Release/mfTest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mfTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
mfTest/CMakeFiles/mfTest.dir/build: bin/mfTest
.PHONY : mfTest/CMakeFiles/mfTest.dir/build

mfTest/CMakeFiles/mfTest.dir/requires: mfTest/CMakeFiles/mfTest.dir/mfTest.cpp.o.requires
.PHONY : mfTest/CMakeFiles/mfTest.dir/requires

mfTest/CMakeFiles/mfTest.dir/clean:
	cd /home/lw/桌面/phpmf/Release/mfTest && $(CMAKE_COMMAND) -P CMakeFiles/mfTest.dir/cmake_clean.cmake
.PHONY : mfTest/CMakeFiles/mfTest.dir/clean

mfTest/CMakeFiles/mfTest.dir/depend:
	cd /home/lw/桌面/phpmf/Release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lw/桌面/phpmf /home/lw/桌面/phpmf/mfTest /home/lw/桌面/phpmf/Release /home/lw/桌面/phpmf/Release/mfTest /home/lw/桌面/phpmf/Release/mfTest/CMakeFiles/mfTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : mfTest/CMakeFiles/mfTest.dir/depend

