//Initialize some constants
static float PI=3.142f;
static float TWOPI=1.0f;

float* read_data(const char *);
float* copy_data_to_device(float*, const int);
float* copy_data_to_host(const float*,const int);
float gamma(float& );
float unwrap(float,float);
float* unwrap_cpu(float*, int&);
void export_data(const char*,float*,int&);
