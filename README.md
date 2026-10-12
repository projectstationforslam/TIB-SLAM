# TIB-SLAM

TIB-SLAM is a real-time visual and visual-inertial SLAM library that supports
**monocular**, **stereo** and **RGB-D** cameras with **pinhole** and **fisheye**
lens models, including multi-map operation and loop closing. On top of this
core, the repository provides two opt-in research extensions:

- **Observability-guided Gaussian mapping** — an asynchronous RGB-D
  Gaussian-splatting map backend whose insertion permission, fusion weight,
  opacity and sampling density are driven by pose confidence.
- **Structural RGB-D** — line, plane and Manhattan-world constraints for
  low-texture and highly repetitive scenes.

See [`STRUCTURAL_RGBD.md`](STRUCTURAL_RGBD.md) for the structural extension and
[`docs/`](docs/) for installation and reproduction notes.

## Features

- Visual, visual-inertial and multi-map SLAM in a single system.
- Monocular, stereo, RGB-D and inertial sensor configurations.
- Pinhole and Kannala-Brandt (fisheye) camera models.
- Real-time operation with a Pangolin-based viewer.
- Opt-in Gaussian-splatting map and structural (line / plane / Manhattan) backends.

## Dependencies

Tested on Ubuntu 24.04. The following are required:

- CMake >= 3.16 and a C++17 compiler
- OpenCV >= 4.4 with the `contrib` modules (`line_descriptor`)
- Eigen3 >= 3.1
- Pangolin 0.9.x
- Boost
- PCL (for the structural extension)
- OpenSSL

Install the system dependencies on Ubuntu with:

```bash
sudo apt update
sudo apt install build-essential cmake git \
  libopencv-dev libopencv-contrib-dev \
  libeigen3-dev libboost-all-dev libpcl-dev libssl-dev
```

Pangolin must be built from source (0.9.x). DBoW2 and g2o are vendored in
`Thirdparty/` and built by `build.sh`.

## Building

```bash
./build.sh
```

This builds DBoW2, g2o and Sophus, unpacks the vocabulary, and compiles the
TIB-SLAM shared library (`lib/libTIB_SLAM.so`) together with the example
programs in `Examples/`.

To build only the main targets without the script:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Running

The example programs take a vocabulary file, a settings file and the input
sequence. For an RGB-D sequence with timestamp associations (TUM format):

```bash
./Examples/RGB-D/rgbd_tum \
  Vocabulary/ORBvoc.txt \
  Examples/RGB-D/<settings>.yaml \
  /path/to/sequence \
  /path/to/sequence/associate.txt
```

RGB-D settings for the TUM and ICL datasets are provided under `Examples/RGB-D/`
(e.g. `TUM3_structure.yaml`, `ICL_structure.yaml`).

Optional arguments: pass `no_viewer` to run without the GUI, and an optional
degradation mode (`none|blur|lowlight|depthdrop|mixed`) for robustness
experiments:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3_structure.yaml \
  /path/to/sequence /path/to/sequence/associate.txt no_viewer
```

On exit, the estimated trajectory is written to `CameraTrajectory.txt`, the
keyframe trajectory to `KeyFrameTrajectory.txt`, and tracking statistics to
CSV/JSON files in the working directory.

## Configuration

Calibration and algorithmic settings live in YAML files under `Examples/`.
RGB-D configurations for the TUM and ICL datasets are provided as
`Examples/RGB-D/TUM3_*.yaml` and `Examples/RGB-D/ICL_*.yaml`.

## Structural RGB-D extension

An opt-in line / plane / Manhattan-world module for RGB-D. It is disabled by
default (`Structure.Enable: 0`) and leaves the standard path unchanged when
off. See [`STRUCTURAL_RGBD.md`](STRUCTURAL_RGBD.md) for architecture,
configuration and build instructions.

## License

TIB-SLAM is released under the GNU General Public License v3. See
[`LICENSE`](LICENSE).
