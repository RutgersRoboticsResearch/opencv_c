# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

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
CMAKE_SOURCE_DIR = /home/earl/Desktop/openCVTut

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/earl/Desktop/openCVTut

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running interactive CMake command-line interface..."
	/usr/bin/cmake -i .
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/earl/Desktop/openCVTut/CMakeFiles /home/earl/Desktop/openCVTut/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/earl/Desktop/openCVTut/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named color_detection

# Build rule for target.
color_detection: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 color_detection
.PHONY : color_detection

# fast build rule for target.
color_detection/fast:
	$(MAKE) -f CMakeFiles/color_detection.dir/build.make CMakeFiles/color_detection.dir/build
.PHONY : color_detection/fast

color_detection.o: color_detection.cpp.o
.PHONY : color_detection.o

# target to build an object file
color_detection.cpp.o:
	$(MAKE) -f CMakeFiles/color_detection.dir/build.make CMakeFiles/color_detection.dir/color_detection.cpp.o
.PHONY : color_detection.cpp.o

color_detection.i: color_detection.cpp.i
.PHONY : color_detection.i

# target to preprocess a source file
color_detection.cpp.i:
	$(MAKE) -f CMakeFiles/color_detection.dir/build.make CMakeFiles/color_detection.dir/color_detection.cpp.i
.PHONY : color_detection.cpp.i

color_detection.s: color_detection.cpp.s
.PHONY : color_detection.s

# target to generate assembly for a file
color_detection.cpp.s:
	$(MAKE) -f CMakeFiles/color_detection.dir/build.make CMakeFiles/color_detection.dir/color_detection.cpp.s
.PHONY : color_detection.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... color_detection"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... color_detection.o"
	@echo "... color_detection.i"
	@echo "... color_detection.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
