# TIB-SLAM 高精度结构 RGB-D 扩展

本分支保留 TIB-SLAM 的 Tracking、Atlas、IMU 和多相机框架，只以可关闭模块的方式加入线、面和 Manhattan 约束。`Structure.Enable: 0` 时不生成结构观测，标准 TIB-SLAM 路径保持不变。

## 已实现内容

- 直线：OpenCV contrib `line_descriptor` 的 LSD KeyLine 与 LBD 二进制描述子；端点邻域深度中值反投影；地图线维护最多 20 次观测的 Hamming medoid 描述子；匹配联合使用 LBD、投影方向、像素距离和三维端点距离。
- 平面：有组织深度网格上的 PEAC/AHC 连通分割；PCL VoxelGrid 降采样和 SACSegmentation/RANSAC 系数精化；保留像素成员、支撑点、质心、协方差和精化内点率。
- 平面匹配：联合使用法向角、平面距离和支撑质心，默认阈值接近 ManhattanSLAM，降低相似柜门等重复结构的误关联。
- Manhattan 方向：LocalMapping 保存正交二平面/三平面及其历史关键帧；当前帧只使用已经匹配到持久 `MapPlane` 的正交平面对，与历史关键帧平面基恢复旋转，经 SVD 投影到 SO(3)，并检查与预测旋转及上一可靠观测的连续性。
- 混合位姿求解：可靠 Manhattan 帧先解析设置旋转，再由专用 g2o `VertexTranslation` 优化三维平移；平移约束不足或发散时恢复原姿态并回退完整六自由度 SE(3) 非线性优化。
- g2o：完整平面采用两个法向切空间分量加一个距离分量的最小三维残差；平行关系采用二维切空间残差；点、双目点、线端点和平面距离的平移边均有解析雅可比。
- 平面权重：`AngleInfo`/`ParallelInfo`/`PerpendicularInfo` 按原版换算为 `1/sigma_rad^2`，`DistanceInfo` 按原版平方；同时兼容 ManhattanSLAM 的 `Plane.*` 旧配置键。
- 安全性：顶点、相机、索引、描述子尺寸、PCL 系数长度、空点云、零法向、非有限值、负深度和优化发散均有显式防护；结构地标由 `Map` 独占，避免并发剔除后的悬空指针。
- 跟踪状态机：结构模式下，点、线和直接平面内点共同参与参考帧、运动模型和局部地图的成功判定；正常门槛与 ManhattanSLAM 一致为 7，刚重定位后的保护门槛为 20。新平面会触发关键帧，LocalMapping 按 found/visible 与关键帧观测数剔除新线面地标。

当前结构约束进入当前帧 Pose-only/Translation-only 优化。与原 ManhattanSLAM 一样，线面地标本身不作为 Local BA 变量；TIB-SLAM 的点 Local BA、回环和 IMU 滑窗仍由上游实现负责。结构地标和 Manhattan 历史索引尚未序列化进 Atlas。

## Ubuntu 24.04 依赖

```bash
sudo apt update
sudo apt install \
  build-essential cmake git \
  libopencv-dev libopencv-contrib-dev \
  libeigen3-dev libboost-all-dev libpcl-dev libssl-dev
```

`libopencv-contrib-dev` 必须安装，它在 OpenCV 4.x 中提供 `line_descriptor`；不需要降级到 OpenCV 3。PCL 为本版本必需依赖。CUDA 不是必需依赖。Pangolin 按 TIB-SLAM 上游方式安装，Ubuntu 24.04 推荐 0.9.x。

本次已在 Ubuntu 24.04 的 GCC 13.3、OpenCV 4.6、Eigen 3.4 和 PCL 1.14 上完成全量构建。项目统一使用 C++17。OpenCV 4.x 的 LSD 金字塔检测尺度必须为整数，示例配置使用 `Structure.Line.Scale: 2`；不要沿用旧代码的 1.2。

## 构建

先按上游 `build.sh` 构建 DBoW2、g2o 和主库，或使用独立 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

CMake 会强制检查 OpenCV `line_descriptor` 和 PCL；缺失时会在配置阶段给出错误，而不是到链接或运行时崩溃。

## 雅可比与 sanitizer

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_STRUCTURAL_TESTS=ON
cmake --build build-debug --target structural_jacobian_test -j2
ASAN_OPTIONS=detect_leaks=1 ./build-debug/structural_jacobian_test
```

测试对完整 SE(3) 的线、最小平面、平行、垂直和 Manhattan 边，以及固定旋转的单目点、双目点、线和平面距离边逐列做中心有限差分，并覆盖非法输入保护。不要脱离 `SparseOptimizer/JacobianWorkspace` 在业务代码中直接调用 g2o 边的 `linearizeOplus()`；旧 g2o 的雅可比成员是映射到工作区的 `Eigen::Map`，无工作区调用是典型段错误来源。

## 实验口径

TUM cabinet 容易发生 Atlas 多 Map。必须至少运行三次，并同时报告：主要/最大 Map ATE、全轨迹 ATE、覆盖率、Map 数、跟踪失败次数、结构提取和优化耗时。不要把多个未对齐 Map 直接拼接后的单一 ATE 当作算法精度。

推荐逐项消融：点基线；点+线；点+面；点+线+面；再开启 Manhattan 混合求解。调参入口见 `Examples/RGB-D/TUM3_structure.yaml`。
