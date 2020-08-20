g++ -c -pipe -g -std=gnu++11 -Wall -W -fPIC -I. -I./tensorflow -I./tensorflow/bazel-tensorflow/external/eigen_archive -I./tensorflow/bazel-tensorflow/external/protobuf_archive/src -I./tensorflow/bazel-genfiles -o icnet_obj.o ./Icnet/inference_from_pb.cc
g++  -o icnet_exe icnet_obj.o   -L./tensorflow/bazel-bin/tensorflow  -ltensorflow_cc -ltensorflow_framework
cp -r ./tensorflow/bazel-bin/tensorflow/libtensorflow* .
