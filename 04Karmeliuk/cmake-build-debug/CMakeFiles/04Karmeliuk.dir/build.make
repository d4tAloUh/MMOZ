# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/164/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/164/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alouh/Projects/MMOZ/04Karmeliuk

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alouh/Projects/MMOZ/04Karmeliuk/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/04Karmeliuk.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/04Karmeliuk.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/04Karmeliuk.dir/flags.make

CMakeFiles/04Karmeliuk.dir/main.cpp.o: CMakeFiles/04Karmeliuk.dir/flags.make
CMakeFiles/04Karmeliuk.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alouh/Projects/MMOZ/04Karmeliuk/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/04Karmeliuk.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/04Karmeliuk.dir/main.cpp.o -c /home/alouh/Projects/MMOZ/04Karmeliuk/main.cpp

CMakeFiles/04Karmeliuk.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/04Karmeliuk.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alouh/Projects/MMOZ/04Karmeliuk/main.cpp > CMakeFiles/04Karmeliuk.dir/main.cpp.i

CMakeFiles/04Karmeliuk.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/04Karmeliuk.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alouh/Projects/MMOZ/04Karmeliuk/main.cpp -o CMakeFiles/04Karmeliuk.dir/main.cpp.s

# Object files for target 04Karmeliuk
04Karmeliuk_OBJECTS = \
"CMakeFiles/04Karmeliuk.dir/main.cpp.o"

# External object files for target 04Karmeliuk
04Karmeliuk_EXTERNAL_OBJECTS =

04Karmeliuk: CMakeFiles/04Karmeliuk.dir/main.cpp.o
04Karmeliuk: CMakeFiles/04Karmeliuk.dir/build.make
04Karmeliuk: /usr/local/lib/libopencv_gapi.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_stitching.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_aruco.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_barcode.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_bgsegm.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_bioinspired.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_ccalib.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_dnn_superres.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_dpm.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_face.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_freetype.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_fuzzy.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_hfs.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_img_hash.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_intensity_transform.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_line_descriptor.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_mcc.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_quality.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_rapid.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_reg.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_rgbd.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_saliency.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_stereo.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_structured_light.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_superres.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_surface_matching.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_tracking.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_videostab.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_wechat_qrcode.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_xfeatures2d.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_xobjdetect.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_xphoto.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_shape.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_highgui.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_datasets.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_plot.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_text.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_ml.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_optflow.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_ximgproc.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_video.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_videoio.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_dnn.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_imgcodecs.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_objdetect.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_calib3d.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_features2d.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_flann.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_photo.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_imgproc.so.4.5.3
04Karmeliuk: /usr/local/lib/libopencv_core.so.4.5.3
04Karmeliuk: CMakeFiles/04Karmeliuk.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alouh/Projects/MMOZ/04Karmeliuk/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 04Karmeliuk"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/04Karmeliuk.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/04Karmeliuk.dir/build: 04Karmeliuk
.PHONY : CMakeFiles/04Karmeliuk.dir/build

CMakeFiles/04Karmeliuk.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/04Karmeliuk.dir/cmake_clean.cmake
.PHONY : CMakeFiles/04Karmeliuk.dir/clean

CMakeFiles/04Karmeliuk.dir/depend:
	cd /home/alouh/Projects/MMOZ/04Karmeliuk/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alouh/Projects/MMOZ/04Karmeliuk /home/alouh/Projects/MMOZ/04Karmeliuk /home/alouh/Projects/MMOZ/04Karmeliuk/cmake-build-debug /home/alouh/Projects/MMOZ/04Karmeliuk/cmake-build-debug /home/alouh/Projects/MMOZ/04Karmeliuk/cmake-build-debug/CMakeFiles/04Karmeliuk.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/04Karmeliuk.dir/depend

