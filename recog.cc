/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It has been stripped down from the tensorflow/examples/label_image sample
// code to remove features and ops not included in the mobile/embedded core
// library available on the Raspberry Pi.
//
// Full build instructions are at tensorflow/contrib/pi_examples/README.md.

// #include <cstddef>

#include <stdio.h>
// #include <jpeglib.h>
#include <setjmp.h>

#include <fstream>

#include "recog.h"
#include "tensorflow/core/framework/graph.pb.h"
// #include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
// #include "tensorflow/core/platform/types.h"
// #include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/core/lib/strings/str_util.h"


// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

Recog::Recog()
{

}
Recog::~Recog()
{

}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status Recog::ReadLabelsFile(string file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}


// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status Recog::ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string original_name = "identity";
  string output_name = "normalized";
  auto file_reader =
      tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 1;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }

  // Also return identity so that we can know the original dimensions and
  // optionally save the image out with bounding boxes overlaid.
  auto original_image = Identity(root.WithOpName(original_name), image_reader);

  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = Cast(root.WithOpName("float_caster"), original_image,
                           tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(
      session->Run({}, {output_name, original_name}, {}, out_tensors));
  return Status::OK();
}

Status Recog::SaveImage(const Tensor& tensor, const string& file_path) {
  LOG(INFO) << "Saving image to " << file_path;
  CHECK(tensorflow::str_util::EndsWith(file_path, ".png"))
      << "Only saving of png files is supported.";

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string encoder_name = "encode";
  string output_name = "file_writer";

  tensorflow::Output image_encoder =
      EncodePng(root.WithOpName(encoder_name), tensor);
  tensorflow::ops::WriteFile file_saver = tensorflow::ops::WriteFile(
      root.WithOpName(output_name), file_path, image_encoder);

  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run({}, {}, {output_name}, &outputs));

  return Status::OK();
}


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status Recog::LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status Recog::GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* out_indices, Tensor* out_scores) {
  const Tensor& unsorted_scores_tensor = outputs[0];
  auto unsorted_scores_flat = unsorted_scores_tensor.flat<float>();
  std::vector<std::pair<int, float>> scores;
  for (int i = 0; i < unsorted_scores_flat.size(); ++i) {
    scores.push_back(std::pair<int, float>({i, unsorted_scores_flat(i)}));
  }
  std::sort(scores.begin(), scores.end(),
            [](const std::pair<int, float>& left,
               const std::pair<int, float>& right) {
              return left.second > right.second;
            });
  scores.resize(how_many_labels);
  Tensor sorted_indices(tensorflow::DT_INT32, {how_many_labels});
  Tensor sorted_scores(tensorflow::DT_FLOAT, {how_many_labels});
  for (int i = 0; i < scores.size(); ++i) {
    sorted_indices.flat<int>()(i) = scores[i].first;
    sorted_scores.flat<float>()(i) = scores[i].second;
  }
  *out_indices = sorted_indices;
  *out_scores = sorted_scores;
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status Recog::PrintTopLabels(const std::vector<Tensor>& outputs, string labels_file_name, int &label_index, float &score) 
{
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  // for (int pos = 0; pos < how_many_labels; ++pos) {
  // int label_index0 = indices_flat(pos);
  // float score0 = scores_flat(pos);
  // LOG(INFO) << labels[label_index0] << " (" << label_index0 << "): " << score0;
  // }
  // LOG(INFO) <<  "-------------";
  label_index = indices_flat(0);
  score = scores_flat(0);
  // LOG(INFO) <<  "label_index: " << label_index << " score: " << score;
  // LOG(INFO) <<  "+++++++++++++++";
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status Recog::CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 2;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

int Recog::init(std::string model_path, std::string label_path)
{
    // First we load and initialize the model.
  
  string graph_path = tensorflow::io::JoinPath(root_dir, model_path);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // std::ifstream file(label_path);
  // if (!file) {
  //   return tensorflow::errors::NotFound("Labels file ", label_path,
  //                                       " not found.");
  // }
  labels = label_path;
  return 0;
}

int Recog::recog(std::string img_path, int &label_index, float &score)
{
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, img_path);
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  } 
  // else {
  //   // LOG(INFO) << "Running model succeeded!";
  // }

  // This is for automated testing to make sure we get the expected result with
  // the default settings. We know that label 866 (military uniform) should be
  // the top label for the Admiral Hopper image.
  if (self_test) {
    bool expected_matches;
    Status check_status = CheckTopLabel(outputs, 866, &expected_matches);
    if (!check_status.ok()) {
      LOG(ERROR) << "Running check failed: " << check_status;
      return -1;
    }
    if (!expected_matches) {
      LOG(ERROR) << "Self-test failed!";
      return -1;
    }
  }

  // Do something interesting with the results we've generated.
  //   int* label_index;
  //   float* score;
  Status print_status = PrintTopLabels(outputs, labels, label_index, score);
  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
  return 0;
}

// int main(int argc, char* argv[]) {
//   // These are the command-line flags the program can understand.
//   // They define where the graph and input data is located, and what kind of
//   // input the model expects. If you train your own model, or use something
//   // other than GoogLeNet you'll need to update these.
//   string image = argv[1];
//   string graph = "/home/swls/work_dir/git/socket_recog/train_model/saved_model/output.pb";


// //   std::vector<tensorflow::Flag> flag_list = {
// //       Flag("image", &image, "image to be processed"),
// //       Flag("graph", &graph, "graph to be executed"),
// //       Flag("labels", &labels, "name of file containing labels"),
// //       Flag("input_width", &input_width, "resize image to this width in pixels"),
// //       Flag("input_height", &input_height,
// //            "resize image to this height in pixels"),
// //       Flag("input_mean", &input_mean, "scale pixel values to this mean"),
// //       Flag("input_std", &input_std, "scale pixel values to this std deviation"),
// //       Flag("input_layer", &input_layer, "name of input layer"),
// //       Flag("output_layer", &output_layer, "name of output layer"),
// //       Flag("self_test", &self_test, "run a self test"),
// //       Flag("root_dir", &root_dir,
// //            "interpret image and graph file names relative to this directory"),
// //   };
// //   string usage = tensorflow::Flags::Usage(argv[0], flag_list);
// //   const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
// //   if (!parse_result) {
// //     LOG(ERROR) << "\n" << usage;
// //     return -1;
// //   }

//   // We need to call this to set up global state for TensorFlow.
// //   tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
// //   if (argc > 2) {
// //     LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
// //     return -1;
// //   }

  



  

//   return 0;
// }



// // Decompresses a JPEG file from disk.
// Status LoadJpegFile(string file_name, std::vector<tensorflow::uint8>* data,
//                     int* width, int* height, int* channels) {
//   struct jpeg_decompress_struct cinfo;
//   FILE* infile;
//   JSAMPARRAY buffer;
//   int row_stride;

//   if ((infile = fopen(file_name.c_str(), "rb")) == NULL) {
//     LOG(ERROR) << "Can't open " << file_name;
//     return tensorflow::errors::NotFound("JPEG file ", file_name, " not found");
//   }

//   struct jpeg_error_mgr jerr;
//   jmp_buf jpeg_jmpbuf;  // recovery point in case of error
//   cinfo.err = jpeg_std_error(&jerr);
//   cinfo.client_data = &jpeg_jmpbuf;
//   jerr.error_exit = CatchError;
//   if (setjmp(jpeg_jmpbuf)) {
//     fclose(infile);
//     return tensorflow::errors::Unknown("JPEG decoding failed");
//   }

//   jpeg_create_decompress(&cinfo);
//   jpeg_stdio_src(&cinfo, infile);
//   jpeg_read_header(&cinfo, TRUE);
//   jpeg_start_decompress(&cinfo);
//   *width = cinfo.output_width;
//   *height = cinfo.output_height;
//   *channels = cinfo.output_components;
//   data->resize((*height) * (*width) * (*channels));

//   row_stride = cinfo.output_width * cinfo.output_components;
//   buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
//                                       row_stride, 1);
//   while (cinfo.output_scanline < cinfo.output_height) {
//     tensorflow::uint8* row_address =
//         &((*data)[cinfo.output_scanline * row_stride]);
//     jpeg_read_scanlines(&cinfo, buffer, 1);
//     memcpy(row_address, buffer[0], row_stride);
//   }

//   jpeg_finish_decompress(&cinfo);
//   jpeg_destroy_decompress(&cinfo);
//   fclose(infile);
//   return Status::OK();
// }


// Error handling for JPEG decoding.
// void CatchError(j_common_ptr cinfo) {
//   (*cinfo->err->output_message)(cinfo);
//   jmp_buf* jpeg_jmpbuf = reinterpret_cast<jmp_buf*>(cinfo->client_data);
//   jpeg_destroy(cinfo);
//   longjmp(*jpeg_jmpbuf, 1);
// }



// // Given an image file name, read in the data, try to decode it as an image,
// // resize it to the requested size, and then scale the values as desired.
// Status ReadTensorFromImageFile(string file_name, const int wanted_height,
//                                const int wanted_width, const float input_mean,
//                                const float input_std,
//                                std::vector<Tensor>* out_tensors) {
//   std::vector<tensorflow::uint8> image_data;
//   int image_width;
//   int image_height;
//   int image_channels;
//   TF_RETURN_IF_ERROR(LoadJpegFile(file_name, &image_data, &image_width,
//                                   &image_height, &image_channels));
//   LOG(INFO) << "Loaded JPEG: " << image_width << "x" << image_height << "x"
//             << image_channels;
//   const int wanted_channels = 1;
//   if (image_channels < wanted_channels) {
//     return tensorflow::errors::FailedPrecondition(
//         "Image needs to have at least ", wanted_channels, " but only has ",
//         image_channels);
//   }
//   // In these loops, we convert the eight-bit data in the image into float,
//   // resize it using bilinear filtering, and scale it numerically to the float
//   // range that the model expects (given by input_mean and input_std).
//   tensorflow::Tensor image_tensor(
//       tensorflow::DT_FLOAT,
//       tensorflow::TensorShape(
//           {1, wanted_height, wanted_width, wanted_channels}));
//   auto image_tensor_mapped = image_tensor.tensor<float, 4>();
//   tensorflow::uint8* in = image_data.data();
//   float* out = image_tensor_mapped.data();
//   const size_t image_rowlen = image_width * image_channels;
//   const float width_scale = static_cast<float>(image_width) / wanted_width;
//   const float height_scale = static_cast<float>(image_height) / wanted_height;
//   for (int y = 0; y < wanted_height; ++y) {
//     const float in_y = y * height_scale;
//     const int top_y_index = static_cast<int>(floorf(in_y));
//     const int bottom_y_index =
//         std::min(static_cast<int>(ceilf(in_y)), (image_height - 1));
//     const float y_lerp = in_y - top_y_index;
//     tensorflow::uint8* in_top_row = in + (top_y_index * image_rowlen);
//     tensorflow::uint8* in_bottom_row = in + (bottom_y_index * image_rowlen);
//     float* out_row = out + (y * wanted_width * wanted_channels);
//     for (int x = 0; x < wanted_width; ++x) {
//       const float in_x = x * width_scale;
//       const int left_x_index = static_cast<int>(floorf(in_x));
//       const int right_x_index =
//           std::min(static_cast<int>(ceilf(in_x)), (image_width - 1));
//       tensorflow::uint8* in_top_left_pixel =
//           in_top_row + (left_x_index * wanted_channels);
//       tensorflow::uint8* in_top_right_pixel =
//           in_top_row + (right_x_index * wanted_channels);
//       tensorflow::uint8* in_bottom_left_pixel =
//           in_bottom_row + (left_x_index * wanted_channels);
//       tensorflow::uint8* in_bottom_right_pixel =
//           in_bottom_row + (right_x_index * wanted_channels);
//       const float x_lerp = in_x - left_x_index;
//       float* out_pixel = out_row + (x * wanted_channels);
//       for (int c = 0; c < wanted_channels; ++c) {
//         const float top_left((in_top_left_pixel[c] - input_mean) / input_std);
//         const float top_right((in_top_right_pixel[c] - input_mean) / input_std);
//         const float bottom_left((in_bottom_left_pixel[c] - input_mean) /
//                                 input_std);
//         const float bottom_right((in_bottom_right_pixel[c] - input_mean) /
//                                  input_std);
//         const float top = top_left + (top_right - top_left) * x_lerp;
//         const float bottom =
//             bottom_left + (bottom_right - bottom_left) * x_lerp;
//         out_pixel[c] = top + (bottom - top) * y_lerp;
//       }
//     }
//   }

//   out_tensors->push_back(image_tensor);
//   return Status::OK();
// }