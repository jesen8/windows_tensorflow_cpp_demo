
#include "Reed_Switch.h"
#include <iostream>

using namespace recog;

int main(int argc, char** argv)
{
    ReedSwitch reed_switch;

    int ret = reed_switch.init(".\\models\\output.pb", _T(".\\models\\recoglib.dll"), ".\\models\\text_map.txt");
    if (ret != 0){
        return -1;
    }

    // for (int i = 0; i < 10; i++){
    //         int out = reed_switch.recog(new_path.c_str());
    //     }
    int out = reed_switch.recog(argv[1]);
    std:: cout << "out " << out std::endl;
    return 0;
}