#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <random>
#include "cnpy.h"
#include <cstring>

#define VOCAB_SIZE 41
#define HIDDEN_DIM 16

#define CL_CHECK(_expr) do { cl_int _err = (_expr); if (_err != CL_SUCCESS) { \
  printf("OpenCL Error: %d at %s:%d\n", _err, __FILE__, __LINE__); exit(-1); }} while(0)
#define CL_CHECK2(_expr) ({ cl_int _err = CL_INVALID_VALUE; auto _ret = (_expr); \
  if (_err != CL_SUCCESS) { printf("OpenCL Error: %d at %s:%d\n", _err, __FILE__, __LINE__); exit(-1); } _ret; })

static inline void safe_rel(cl_mem& m){ if(m){ clReleaseMemObject(m); m=nullptr; } }
static inline void safe_rel(cl_kernel& k){ if(k){ clReleaseKernel(k); k=nullptr; } }
static inline void safe_rel(cl_program& p){ if(p){ clReleaseProgram(p); p=nullptr; } }
static inline void safe_rel(cl_command_queue& q){ if(q){ clReleaseCommandQueue(q); q=nullptr; } }
static inline void safe_rel(cl_context& c){ if(c){ clReleaseContext(c); c=nullptr; } }

void cleanup(cl_kernel& kernel, cl_program& program,
             cl_mem& w1, cl_mem& b1, cl_mem& w2, cl_mem& b2,
             cl_mem& x, cl_mem& h1, cl_mem& out,
             cl_command_queue& queue, cl_context& context) {
  safe_rel(w1); safe_rel(b1); safe_rel(w2); safe_rel(b2);
  safe_rel(x);  safe_rel(h1); safe_rel(out);
  safe_rel(kernel); safe_rel(program); safe_rel(queue); safe_rel(context);
}

static cl_platform_id pick_pocl_platform() {
  cl_uint n=0; CL_CHECK(clGetPlatformIDs(0,nullptr,&n));
  if(!n) return nullptr;
  std::vector<cl_platform_id> P(n);
  CL_CHECK(clGetPlatformIDs(n,P.data(),nullptr));
  char name[256]={0}, vendor[256]={0};
  for(auto p: P){
    std::memset(name,0,sizeof(name));
    std::memset(vendor,0,sizeof(vendor));
    clGetPlatformInfo(p, CL_PLATFORM_NAME,   sizeof(name),   name,   nullptr);
    clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
    std::string s(name); std::string v(vendor);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (s.find("pocl")!=std::string::npos || s.find("portable")!=std::string::npos) return p;
  }
  return P[0];
}

std::vector<std::string> vocab = {
  "gpt","xrt","vortex","fpga","kernel","inference","runs","executes",
  "tinygpt","opencl","thread","warp","core","fast","slow","int8",
  "matvec","quant","weight","bias","input","token","prompt","next",
  "device","platform","bitstream","compile","loop","memory","clock",
  "latency","model","load","store","u280","gemm","sgemm","activation",
  "softmax","output"
};
std::unordered_map<std::string,int> token_to_id;
void build_token_map(){ for(size_t i=0;i<vocab.size();++i) token_to_id[vocab[i]]=(int)i; }

void softmax(std::vector<float>& vec, float temp=1.0f){
  float mx = *std::max_element(vec.begin(), vec.end());
  float sum=0.f;
  for(auto& v: vec){ v = std::exp((v-mx)/temp); sum += v; }
  if(sum==0.f) return;
  for(auto& v: vec) v /= sum;
}

int sample_top_k(const std::vector<float>& probs, int k, int vocab_size, int& chosen_index){
  std::vector<std::pair<float,int> > scored; scored.reserve(vocab_size);
  for(int i=0;i<vocab_size;++i) scored.push_back({probs[i], i});
  if (k>vocab_size) k=vocab_size;
  std::partial_sort(scored.begin(), scored.begin()+k, scored.end(),
    [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; });
  std::vector<float> tp(k); std::vector<int> ti(k); float tot=0.f;
  for(int i=0;i<k;++i){ tp[i]=scored[i].first; ti[i]=scored[i].second; tot+=tp[i]; }
  if(tot<=0.f){ chosen_index=ti[0]; return chosen_index; }
  for(size_t i=0;i<tp.size();++i) tp[i] /= tot;
  static thread_local std::mt19937 gen{std::random_device{}()};
  std::discrete_distribution<> dist(tp.begin(), tp.end());
  chosen_index = ti[dist(gen)];
  return chosen_index;
}

int main(int argc, char** argv){
  build_token_map();

  // Defaults
  float temperature = 0.8f;
  int   top_k       = 5;
  float penalty_strength = 0.9f;
  int   steps       = 15; // default token count

  // Parse args
  for (int i=1; i<argc; ++i) {
    std::string a = argv[i];
    auto take_val = [&](const std::string& key)->std::string {
      if (a == key && i+1<argc) return argv[++i];
      if (a.rfind(key,0)==0 && a.size()>key.size()) return a.substr(key.size());
      return "";
    };
   std::string v;
   v = take_val("-temp");    if (!v.empty()) temperature      = std::stof(v);
   v = take_val("-topk");    if (!v.empty()) top_k            = std::stoi(v);
   v = take_val("-penalty"); if (!v.empty()) penalty_strength = std::stof(v);
   v = take_val("-tokens");  if (!v.empty()) steps            = std::stoi(v);
  }

  printf("TinyGPT — (%d tokens)\n--------------\n", steps);

  // Load weights
  auto W1_npy = cnpy::npy_load("weights1.npy"); 
  auto W2_npy = cnpy::npy_load("weights2.npy"); 
  auto B1_npy = cnpy::npy_load("bias1.npy");    
  auto B2_npy = cnpy::npy_load("bias2.npy");    
  auto E_npy  = cnpy::npy_load("embedding.npy");

  std::vector<float> W1(W1_npy.data<float>(), W1_npy.data<float>() + HIDDEN_DIM*HIDDEN_DIM);
  std::vector<float> W2(W2_npy.data<float>(), W2_npy.data<float>() + VOCAB_SIZE*HIDDEN_DIM);
  std::vector<float> B1(B1_npy.data<float>(), B1_npy.data<float>() + HIDDEN_DIM);
  std::vector<float> B2(B2_npy.data<float>(), B2_npy.data<float>() + VOCAB_SIZE);
  std::vector<float> E (E_npy .data<float>(), E_npy .data<float>() + VOCAB_SIZE*HIDDEN_DIM);

  std::vector<float> h_x(HIDDEN_DIM), h_out(VOCAB_SIZE);

  // OpenCL init
  cl_platform_id platform = pick_pocl_platform();
  if (!platform) { fprintf(stderr,"No OpenCL platform found.\n"); return -1; }
  cl_device_id device_id=nullptr;
  cl_int derr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, nullptr);
  if (derr != CL_SUCCESS) {
    derr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, nullptr);
    if (derr != CL_SUCCESS) CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device_id, nullptr));
  }

  cl_context context = CL_CHECK2(clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &_err));
  cl_command_queue queue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Build kernel
  FILE* fp = fopen("kernel.cl","rb");
  if(!fp){ perror("kernel.cl"); return -1; }
  fseek(fp,0,SEEK_END); size_t kernel_size = ftell(fp); rewind(fp);
  std::vector<char> kernel_src(kernel_size);
  fread(kernel_src.data(),1,kernel_size,fp); fclose(fp);

  const char* kernel_str = kernel_src.data();
  cl_program program = CL_CHECK2(clCreateProgramWithSource(context, 1, &kernel_str, &kernel_size, &_err));
  cl_int bstat = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
  if (bstat != CL_SUCCESS) {
    size_t logsz=0; clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
    std::vector<char> blog(logsz+1,0);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logsz, blog.data(), nullptr);
    fprintf(stderr, "Build failed (%d). Build log:\n%s\n", bstat, blog.data());
    return -1;
  }
  cl_kernel kernel = CL_CHECK2(clCreateKernel(program, "matvec2layer", &_err));

  // Buffers
  cl_mem w1_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, W1.size()*sizeof(float), W1.data(), &_err));
  cl_mem b1_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, B1.size()*sizeof(float), B1.data(), &_err));
  cl_mem w2_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, W2.size()*sizeof(float), W2.data(), &_err));
  cl_mem b2_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, B2.size()*sizeof(float), B2.data(), &_err));
  cl_mem x_buf  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,                        HIDDEN_DIM*sizeof(float), nullptr, &_err));
  cl_mem h1_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,                       HIDDEN_DIM*sizeof(float), nullptr, &_err));
  cl_mem out_buf= CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY,                       VOCAB_SIZE*sizeof(float), nullptr, &_err));

  // Prompt
  std::string line;
  std::cout << "\nEnter prompt: ";
  std::getline(std::cin, line);
  std::istringstream iss(line);
  std::string word; std::vector<std::string> words;
  while (iss >> word) words.push_back(word);
  if (words.empty() || !token_to_id.count(words.back())) { std::cerr << "Invalid prompt.\n"; return 1; }
  int current_token = token_to_id[words.back()];

  std::cout << "\nPrompt: \"" << line << "\"\n(Top-k = " << top_k
            << ", Temp = " << temperature
            << ", Penalty = " << penalty_strength
            << ")\nGenerating " << steps << " tokens:\n\n";

  // Static args
  int H = HIDDEN_DIM, V = VOCAB_SIZE;
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &w1_buf));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b1_buf));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &w2_buf));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &b2_buf));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &x_buf));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &h1_buf));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &out_buf));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &H));
  CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &V));

  std::vector<int> token_history;

  for (int step=0; step<steps; ++step) {
    for (int i=0;i<HIDDEN_DIM;++i) h_x[i] = E[current_token*HIDDEN_DIM + i];
    CL_CHECK(clEnqueueWriteBuffer(queue, x_buf, CL_TRUE, 0, HIDDEN_DIM*sizeof(float), h_x.data(), 0, nullptr, nullptr));

    int phase = 0; size_t gH = HIDDEN_DIM;
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &phase));
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gH, nullptr, 0, nullptr, nullptr));

    phase = 1; size_t gV = VOCAB_SIZE;
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &phase));
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gV, nullptr, 0, nullptr, nullptr));

    CL_CHECK(clFinish(queue));

    CL_CHECK(clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, VOCAB_SIZE*sizeof(float), h_out.data(), 0, nullptr, nullptr));

    for (int t : token_history) h_out[t] -= penalty_strength;

    softmax(h_out, temperature);
    int next_token = -1;
    sample_top_k(h_out, top_k, VOCAB_SIZE, next_token);

    std::cout << vocab[next_token] << " ";
    token_history.push_back(next_token);
    if ((int)token_history.size() > 5) token_history.erase(token_history.begin());
    current_token = next_token;
  }

  std::cout << "\n\nDone.\n";
  // Fast exit to avoid double-free in downstream atexit/dtors
  if (std::getenv("VX_FAST_EXIT")) {
   fflush(stdout);
   fflush(stderr);
   _exit(0);  // bypass all atexit/dynamic destructor paths
  }
  cleanup(kernel, program, w1_buf, b1_buf, w2_buf, b2_buf, x_buf, h1_buf, out_buf, queue, context);
  return 0;
}
