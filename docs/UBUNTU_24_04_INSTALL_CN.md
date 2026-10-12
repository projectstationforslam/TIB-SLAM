# Ubuntu 24.04 安装与复现实验

本文命令面向干净的 Ubuntu 24.04 x86-64。当前代码在本机 WSL Ubuntu
26.04、GCC 15.2、OpenCV 4.10、PCL 1.15 上完成编译验证；这不等同于已经
在 24.04 实机验证，建议投稿前用 GitHub Actions 或独立 24.04 容器复核。

## 依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config ninja-build \
  libopencv-dev libeigen3-dev libboost-all-dev libssl-dev \
  libpcl-dev libglew-dev libgl1-mesa-dev libwayland-dev \
  libxkbcommon-dev libepoxy-dev python3-pip python3-venv
```

Pangolin 建议固定到与 TIB-SLAM 兼容的版本：

```bash
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cmake -S Pangolin -B Pangolin/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF
cmake --build Pangolin/build
sudo cmake --install Pangolin/build
```

## 编译 TIB-SLAM

```bash
cd ~/TIB-SLAM
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_STRUCTURAL_TESTS=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j"$(nproc)" --target \
  information_budget_scheduler_test rgbd_tum
./build/information_budget_scheduler_test
```

如果 `Vocabulary/ORBvoc.txt` 不存在，先执行上游 TIB-SLAM 的词袋下载或
解压步骤。若无 RealSense，CMake 的 SDK 警告不影响 TUM/ICL RGB-D 运行。

## TUM 运行示例

```bash
cd ~/TIB-SLAM
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt \
  Examples/RGB-D/TUM3_structure.yaml \
  /data/rgbd_dataset_freiburg3_cabinet \
  /data/rgbd_dataset_freiburg3_cabinet/associate.txt no_viewer
```

轨迹写入当前目录的 `CameraTrajectory.txt` 和
`KeyFrameTrajectory.txt`。APE 应使用相同时间关联、相同 `--align` 设置和
同一版 evo 对所有方法计算，并保留命令、日志和随机重复结果。
