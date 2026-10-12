echo "Building ROS nodes"

cd Examples_old/ROS/TIB_SLAM
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
