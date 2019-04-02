load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "tf_recog",
    srcs = ["recog.cc"],
    hdrs = ["recog.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ]
)

tf_cc_binary(
    name = "recog",
    srcs = ["main.cc"],
    deps = [
        ":tf_recog",
    ],
)

tf_cc_binary(
    name = "recoglib.dll",
    srcs = ["dll.cc"],
    deps = [
        ":tf_recog",
    ],
    linkshared = 1,
)