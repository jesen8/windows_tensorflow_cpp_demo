

#include "Reed_Switch.h"

#define MODEL_NOT_INIT_ERROR 20
using namespace std;


namespace recog
{
	ReedSwitch::ReedSwitch()
	{
		
	}
	ReedSwitch::~ReedSwitch()
	{

		tf_recog = NULL;
	}

	int ReedSwitch::init(const char* tf_model_path,const  wchar_t* tf_so_path, const char* label_path)
	{
		tf_handle = LoadLibrary(tf_so_path);
		if (tf_handle == NULL)
		{
			FreeLibrary(tf_handle);
			return DLL_OPEN_ERROR;
		}
		tf_init = (TF_INIT)GetProcAddress(tf_handle, "init");
		if (tf_init == NULL)
		{
			FreeLibrary(tf_handle);
			return DLL_FIND_INIT_FCN_ERROR;
		}

		int out = tf_init(tf_model_path, label_path);
		if (out != 0){
			return DLL_MODEL_INIT_ERROR;
		}

		tf_recog = (TF_RECOG)GetProcAddress(tf_handle, "recog");
		if (tf_recog == NULL)
		{
			FreeLibrary(tf_handle);
			return DLL_FIND_RECOG_FCN_ERROR;
		}

		
		is_init_model = true;

		return 0;
	}


	int ReedSwitch::manager(std::string& img_path)
	{
		if (!is_init_model){ return MODEL_NOT_INIT_ERROR; }
		
		
        int out = tf_recog(img_path.c_str());   
		return out;
	}




}
