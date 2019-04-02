#include <string>
#include <vector>
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::string;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::int32;
class Recog
{
    public:
    Recog();
    ~Recog();

    public:
    int init(std::string model_path, std::string label_path);
    int recog(std::string img_path, int &label_index, float &score);

    private:
    Status ReadLabelsFile(string file_name, std::vector<string>* result,
                      size_t* found_label_count);
    Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors);
    Status SaveImage(const Tensor& tensor, const string& file_path);
    Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session);
    Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* out_indices, Tensor* out_scores) ;
    Status PrintTopLabels(const std::vector<Tensor>& outputs, string labels_file_name, int &label_index, float &score); 
    Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected);

    private:
    std::unique_ptr<tensorflow::Session> session;
    string labels =  "";

    int32 input_width = 32;
    int32 input_height = 32;
    int32 input_mean = 0;
    int32 input_std = 255;
    string input_layer = "input_node";
    string output_layer = "Dense2/output_node";
    bool self_test = false;
    string root_dir = "";
};