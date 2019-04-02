
#include "recog.h"

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than GoogLeNet you'll need to update these.
  string image = argv[1];
  string graph = "C:/Users/vanlance/Documents/Visual Studio 2013/Projects/watershed_algorithm/x64/Release/models/output.pb";
  string label_path = "C:/Users/vanlance/Documents/Visual Studio 2013/Projects/watershed_algorithm/x64/Release/models/label.txt";
  int label_index = 10;
  float score = 10;
  Recog recog;
  int code = recog.init(graph, label_path);
  // LOG(INFO) << "init code is: " << code;
  
  code = recog.recog(image, label_index, score);
  LOG(INFO) << "recog code is: " << label_index;
  // LOG(INFO) << "label_index --> " << label_index << " score --> " << code;
  return code;
//   std::vector<tensorflow::Flag> flag_list = {
//       Flag("image", &image, "image to be processed"),
//       Flag("graph", &graph, "graph to be executed"),
//       Flag("labels", &labels, "name of file containing labels"),
//       Flag("input_width", &input_width, "resize image to this width in pixels"),
//       Flag("input_height", &input_height,
//            "resize image to this height in pixels"),
//       Flag("input_mean", &input_mean, "scale pixel values to this mean"),
//       Flag("input_std", &input_std, "scale pixel values to this std deviation"),
//       Flag("input_layer", &input_layer, "name of input layer"),
//       Flag("output_layer", &output_layer, "name of output layer"),
//       Flag("self_test", &self_test, "run a self test"),
//       Flag("root_dir", &root_dir,
//            "interpret image and graph file names relative to this directory"),
//   };
//   string usage = tensorflow::Flags::Usage(argv[0], flag_list);
//   const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
//   if (!parse_result) {
//     LOG(ERROR) << "/n" << usage;
//     return -1;
//   }

  // We need to call this to set up global state for TensorFlow.
//   tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
//   if (argc > 2) {
//     LOG(ERROR) << "Unknown argument " << argv[1] << "/n" << usage;
//     return -1;
//   }

  



  

//   return 0;
}