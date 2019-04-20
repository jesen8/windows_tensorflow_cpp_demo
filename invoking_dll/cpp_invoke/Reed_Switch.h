
#include <Windows.h>
//#include <dlfcn.h>

//#define LINUX



namespace recog
{
	class ReedSwitch
	{
	public:
		ReedSwitch();
		~ReedSwitch();

	public:
		int init(const char* tf_model_path, const wchar_t* tf_so_path, const char* label_path);
		int run(const std::string path){ return manager(path); };
    
    private:
		int manager(const std::string& img_path);
	

	private:
		// 初始化参数
		bool is_init_model = false;
		//

		//网络参数
		typedef int(*TF_RECOG)(const char* img_path);
		typedef int(*TF_INIT)(const char* model_path, const char* label_path);
		TF_RECOG tf_recog = NULL;
		TF_INIT tf_init = NULL;

		HINSTANCE tf_handle;
		char* tf_error;

		//void* tf_handle;
	};
}; // name
