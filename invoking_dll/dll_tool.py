from ctypes import CDLL,byref,create_string_buffer,cdll
import os


def recog_dll(img_path_dir):
    recog_dll_path = r'recoglib.dll'
    ocr_recog_dll = cdll.LoadLibrary(recog_dll_path)
    model_path = r"output.pb"
    model_string_buffer = create_string_buffer(model_path.encode('utf-8'), 65536)
    text_map_path = r"label.txt"
    text_map_string_buffer = create_string_buffer(text_map_path.encode('utf-8'), 65536)
    
    out = ocr_recog_dll.init(model_string_buffer, text_map_string_buffer)
    print('out: ', out)

    for idx, p in enumerate(os.listdir(img_path_dir)):
        path = os.path.join(img_path_dir, p)
        input_string_buffer = create_string_buffer(path.encode('utf-8'), 65536)
        out2 = ocr_recog_dll.reco(byref(input_string_buffer))
        print('out2: ', out2)


if __name__ == "__main__":
    img_path_dir = r'./test/1'
    recog_dll(img_path_dir)
