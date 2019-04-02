
#include "recog.h"
#include <iostream>
Recog tf_recog;

// extern "C" {
//     int init(char* model_path);
//     int recog(char* image);
// }


extern "C" __declspec(dllexport)
int init(char* model_path, char* label_path)
{
    string mp(model_path);
    string sp(label_path);
    if (tf_recog.init(mp, sp) != 0){
        return -1;
    }
    //std::cout << "init ok" << std::endl;
    return 0;
}

extern "C" __declspec(dllexport)
int reco(char* image) 
{

    string image_path(image);
    //   string graph = "/home/swls/work_dir/git/socket_recog/train_model/saved_model/output.pb";
    int label_index = 10;
    float score = 10;
   // std::cout << "image: " << image_path << std::endl;
   // std::cout << "one ok" << std::endl;
    int code = tf_recog.recog(image_path, label_index, score);
  //  std::cout << "two ok" << std::endl;
    //   LOG(INFO) << "recog code is: " << code;
    //   LOG(INFO) << "label_index --> " << label_index << " score --> " << code;
    return label_index;
}