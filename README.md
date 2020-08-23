# ICNet: A Framework for running an inference module using TensorFlow C++ API in ROS. (Semantic Segmentation)

This code provides a framework to easily add architectures and datasets, in order to run an inference module using TensorFlow C++ API in ROS(Robotic operating system).
It contains a full pipeline of different nodes performing different tasks in C++ using Tensorflow and OpenCV.
There are four different nodes which take input data as an image and deliver a segmented output image data. 
The whole process is carried out in a ROS Environment [ROS_Environment](https://gitlab.kpit.com/ravinas/production_mldl/-/tree/ICNet/catkin_ws).
It can also be performed outside ROS environment, a separate directory is provided for the same [Non ROS-Environment](https://gitlab.kpit.com/ravinas/production_mldl/-/tree/ICNet/Icnet).

## Getting Started

To get started you need to:

1. Clone the repository.
2. Install dependencies and Prerequisites
3. Bazel Build
4. Compile code (certain changes with build_tutorial_cpp.sh file should be done for file location).
5. Run Executable

### Prerequisites

1. Ubuntu 18.04 bionic
2. Python Dependencies
3. Bazel 0.21.0
4. Tensorflow v1.13.1
5. Docker

### Installing

**Python Dependencies**
```
sudo apt install python3-dev python3-pip
pip3 install -U --user six numpy wheel mock
pip3 install -U --user keras_applications==1.0.6 --no-deps
pip3 install -U --user keras_preprocessing==1.0.5 --no-deps
```

**Bazel**

sudo apt update && sudo apt install bazel 
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
chmod +x bazel-0.21.0-installer-linux-x86_64.sh
./bazel-0.21.0-installer-linux-x86_64.sh --user
```

**Downloading TensorFlow source code**
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v1.13.1
```
**abseil-cpp**
```
cd tensorflow
git clone https://github.com/abseil/abseil-cpp.git
ln -s abseil-cpp/absl ./absl
```
**Copy Tensorflow Lib i,e .SO file to Shared Library Path**
```
sudo cp /home/ranjeet/libtensorflow_cc.so /usr/lib/
sudo cp /home/ranjeet/libtensorflow_framework.so /usr/lib/
```

**Configure the build**
```
./configure
```
Note: While configuring set everthing as default as it is CPU use only ([Reference](https://www.tensorflow.org/install/source)).

**Bazel Build**
```
bazel build --jobs=6 --verbose_failures -c opt --copt=-mavx --copt=-mfpmath=both --copt=-msse4.2 //tensorflow:libtensorflow_cc.so
```

**Compilation**
```
sudo ./build_tutorial_cpp.sh
```

**Run executable**
```
./icnet_exe
```

**Docker**
Note: The docker instructions are provided in the docker floder.

## References

1. [Building an inference module using TensorFlow C++ API](https://medium.com/@amsokol.com/how-to-build-and-install-tensorflow-gpu-cpu-for-windows-from-source-code-using-bazel-d047d9342b44)
2. [Bonnet](https://github.com/PRBonn/bonnet)
3. [Unet](https://github.com/jakeret/tf_unet)
4. [Tensorflow](https://www.tensorflow.org/install/source)
5. [Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html)
