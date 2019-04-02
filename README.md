# windows_tensorflow_cpp_demo
a tensorflow cpp demo in windows

1. build libtensorflow_cc.so
2. put this directory in /tensorflow/cc/, and look like this /tensorflow/cc/windows_tensorflow_cpp_demo

3. in tensorflow root directory run bazel build //tensorflow/cc/windows_tensorflow_cpp_demo:recoglib.dll

4. use python or c++ test this dll
