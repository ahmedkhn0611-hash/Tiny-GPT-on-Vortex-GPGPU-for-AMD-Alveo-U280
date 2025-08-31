#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <random>
#include "cnpy.h"

// ====== model dims ======
#define VOCAB_SIZE 41
#define HIDDEN_DIM 16

// ====== OpenCL helpers ======
#define CL_CHECK(_expr) do { cl_int _err = (_expr); if (_err != CL_SUCCESS) { \
  printf("OpenCL Error: %d at %s:%d\n", _err, __FILE__, __LINE__); exit(-1); }} while(0)
#define CL_CHECK2(_expr) ({ cl_int _err = CL_INVALID_VALUE; auto _ret = (_expr); \
  if (_err != CL_SUCCESS) { printf("OpenCL Error: %d at %s:%d\n", _err, __FILE__, __LINE__); exit(-1); } _ret; })

static inline void safe_rel(cl_mem& m){ if(m){ clReleaseMemObject(m); m=nullptr; } }
static inline void safe_rel(cl_kernel& k){ if(k){ clReleaseKernel(k); k=nullptr; } }
static inline void safe_rel(cl_program& p){ if(p){ clReleaseProgram(p); p=nullptr; } }
static inline void safe_rel(cl_command_queue& q){ if(q){ clReleaseCommandQueue(q); q=nullptr; } }
static inline void safe_rel(cl_context& c){ if(c){ clReleaseContext(c); c=nullptr; } }

void cleanup_all(cl_kernel& k_persist, cl_kernel& k_slice, cl_program& program,
                 cl_mem& w1, cl_mem& b1, cl_mem& w2, cl_mem& b2,
                 cl_mem& emb, cl_mem& io_tokens, cl_mem& logits,
                 cl_command_queue& queue, cl_context& context) {
  safe_rel(w1); safe_rel(b1); safe_rel(w2); safe_rel(b2);
  safe_rel(emb); safe_rel(io_tokens); safe_rel(logits);
  safe_rel(k_persist); safe_rel(k_slice); safe_rel(program); safe_rel(queue); safe_rel(context);
}

static cl_platform_id pick_platform() {
  cl_uint n=0; CL_CHECK(clGetPlatformIDs(0,nullptr,&n));
  if(!n) return nullptr;
  std::vector<cl_platform_id> P(n);
  CL_CHECK(clGetPlatformIDs(n,P.data(),nullptr));
  return P[0];
}

// ====== tiny vocab ======
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

// ====== host softmax / sampling ======
static void softmax_host(std::vector<float>& v, float temp=1.0f){
  float mx = *std::max_element(v.begin(), v.end());
  double sum = 0.0;
  for (float &x: v){ x = expf((x - mx)/temp); sum += x; }
  if (sum > 0) for (float &x: v) x = float(x / sum);
}
static int topk_sample_host(const std::vector<float>& p, int k=5){
  std::vector<int> idx(p.size()); std::iota(idx.begin(), idx.end(), 0);
  if (k > (int)idx.size()) k = (int)idx.size();
  std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
                    [&](int a,int b){ return p[a] > p[b]; });
  // deterministic top-1 (you can swap for stochastic if you prefer)
  return idx[0];
}

int main(int argc, char** argv){
  build_token_map();

  // ====== args ======
  int   steps = 15;
  float temperature = 0.8f;
  int   top_k       = 5;
  float penalty     = 0.0f;    // set >0.0 if you want repetition penalty
  std::string engine = "persist"; // "persist" (1-core) or "slice" (multi-core)
  int groups = 4;               // number of work-groups for "slice" engine

  for (int i=1; i<argc; ++i) {
    std::string a = argv[i];
    auto take = [&](const char* key)->const char*{
      if (a == key && i+1<argc) return argv[++i];
      return nullptr;
    };
    if (a=="-tokens")  { const char* v=take("-tokens");  if(v) steps = std::stoi(v); }
    else if (a=="-temp"){ const char* v=take("-temp");   if(v) temperature = std::stof(v); }
    else if (a=="-topk"){ const char* v=take("-topk");   if(v) top_k = std::stoi(v); }
    else if (a=="-penalty"){ const char* v=take("-penalty"); if(v) penalty = std::stof(v); }
    else if (a=="-engine"){ const char* v=take("-engine"); if(v) engine = v; }
    else if (a=="-groups"){ const char* v=take("-groups"); if(v) groups = std::stoi(v); }
  }

  printf("TinyGPT — (%d tokens)\n--------------\n", steps);

  // ====== weights ======
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

  // ====== OpenCL ======
  cl_platform_id platform = pick_platform();
  if (!platform) { fprintf(stderr,"No OpenCL platform found.\n"); return -1; }
  cl_device_id device_id=nullptr;
  cl_int derr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, nullptr);
  if (derr != CL_SUCCESS) derr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, nullptr);

  cl_int _err = CL_SUCCESS;
  cl_context       context = CL_CHECK2(clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &_err));
  cl_command_queue queue   = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // build program (kernel.cl contains both kernels)
  FILE* fp = fopen("kernel.cl","rb");
  if(!fp){ perror("kernel.cl"); return -1; }
  fseek(fp,0,SEEK_END); size_t kernel_size = ftell(fp); rewind(fp);
  std::vector<char> kernel_src(kernel_size);
  fread(kernel_src.data(),1,kernel_size,fp); fclose(fp);

  const char* kernel_str = kernel_src.data();
  cl_program program = CL_CHECK2(clCreateProgramWithSource(context, 1, &kernel_str, &kernel_size, &_err));
  const char* opts = "-cl-std=CL1.2 -cl-fast-relaxed-math";
  cl_int bstat = clBuildProgram(program, 1, &device_id, opts, nullptr, nullptr);
  if (bstat != CL_SUCCESS) {
    size_t logsz=0; clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
    std::vector<char> blog(logsz+1,0);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logsz, blog.data(), nullptr);
    fprintf(stderr, "Build failed (%d). Build log:\n%s\n", bstat, blog.data());
    return -1;
  }

  cl_kernel k_persist = CL_CHECK2(clCreateKernel(program, "tinygpt_persist_fused", &_err));
  cl_kernel k_slice   = CL_CHECK2(clCreateKernel(program, "ffn_to_logits_slice",   &_err));

  // buffers
  cl_mem w1_buf  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, W1.size()*sizeof(float), W1.data(), &_err));
  cl_mem b1_buf  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, B1.size()*sizeof(float), B1.data(), &_err));
  cl_mem w2_buf  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, W2.size()*sizeof(float), W2.data(), &_err));
  cl_mem b2_buf  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, B2.size()*sizeof(float), B2.data(), &_err));
  cl_mem emb_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, E.size()*sizeof(float),  E.data(),  &_err));
  cl_mem io_buf  = nullptr;
  cl_mem logits_buf = nullptr;

  // prompt
  std::string line;
  std::cout << "\nEnter prompt: ";
  std::getline(std::cin, line);
  std::istringstream iss(line);
  std::string word; std::vector<std::string> words;
  while (iss >> word) words.push_back(word);
  if (words.empty() || !token_to_id.count(words.back())) { std::cerr << "Invalid prompt.\n"; return 1; }
  int current_token = token_to_id[words.back()];

  printf("\nPrompt: \"%s\"\n", line.c_str());
  printf("Engine: %s  Groups: %d  (Temp=%.2f, TopK=%d, Penalty=%.2f)\n\n",
         engine.c_str(), groups, temperature, top_k, penalty);

  if (engine == "persist") {
    // ---------- single-core persistent fused ----------
    std::vector<int> io_tokens(steps+1, 0);
    io_tokens[0] = current_token;
    io_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      io_tokens.size()*sizeof(int), io_tokens.data(), &_err));

    cl_int H = HIDDEN_DIM, V = VOCAB_SIZE, T = steps;
    CL_CHECK(clSetKernelArg(k_persist, 0, sizeof(cl_mem), &w1_buf));
    CL_CHECK(clSetKernelArg(k_persist, 1, sizeof(cl_mem), &b1_buf));
    CL_CHECK(clSetKernelArg(k_persist, 2, sizeof(cl_mem), &w2_buf));
    CL_CHECK(clSetKernelArg(k_persist, 3, sizeof(cl_mem), &b2_buf));
    CL_CHECK(clSetKernelArg(k_persist, 4, sizeof(cl_mem), &emb_buf));
    CL_CHECK(clSetKernelArg(k_persist, 5, sizeof(cl_mem), &io_buf));
    CL_CHECK(clSetKernelArg(k_persist, 6, sizeof(cl_int), &H));
    CL_CHECK(clSetKernelArg(k_persist, 7, sizeof(cl_int), &V));
    CL_CHECK(clSetKernelArg(k_persist, 8, sizeof(cl_int), &T));

    // 1 WG with 16 WIs
    size_t local  = HIDDEN_DIM;
    size_t max_wg = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(k_persist, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr));
    if (local > max_wg) local = max_wg ? max_wg : 1;
    size_t global = local;
    printf("Launching persist: global=%zu, local=%zu\n", global, local);

    CL_CHECK(clEnqueueNDRangeKernel(queue, k_persist, 1, nullptr, &global, &local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    CL_CHECK(clEnqueueReadBuffer(queue, io_buf, CL_TRUE, 0, io_tokens.size()*sizeof(int), io_tokens.data(), 0, nullptr, nullptr));

    for (int i=1; i<=steps; ++i) {
      int tok = io_tokens[i];
      if (tok < 0 || tok >= (int)vocab.size()) tok = 0;
      std::cout << vocab[tok] << " ";
    }
    std::cout << "\n\nDone.\n";
  } else {
    // ---------- multi-core slice (per-token host loop) ----------
    logits_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, VOCAB_SIZE*sizeof(float), nullptr, &_err));

    size_t local  = HIDDEN_DIM;         // 16 work-items per WG
    size_t max_wg = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(k_slice, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr));
    if (local > max_wg) local = max_wg ? max_wg : 1;

    size_t global = local * (size_t)groups; // groups work-groups (=> cores)
    printf("Launching slice per token: global=%zu, local=%zu, groups=%d\n", global, local, groups);

    std::vector<float> logits(VOCAB_SIZE);
    std::vector<int>   history;

    for (int t=0; t<steps; ++t) {
      cl_int Hc = HIDDEN_DIM, Vc = VOCAB_SIZE, tok = current_token;
      CL_CHECK(clSetKernelArg(k_slice, 0, sizeof(cl_mem), &w1_buf));
      CL_CHECK(clSetKernelArg(k_slice, 1, sizeof(cl_mem), &b1_buf));
      CL_CHECK(clSetKernelArg(k_slice, 2, sizeof(cl_mem), &w2_buf));
      CL_CHECK(clSetKernelArg(k_slice, 3, sizeof(cl_mem), &b2_buf));
      CL_CHECK(clSetKernelArg(k_slice, 4, sizeof(cl_mem), &emb_buf));
      CL_CHECK(clSetKernelArg(k_slice, 5, sizeof(cl_mem), &logits_buf));
      CL_CHECK(clSetKernelArg(k_slice, 6, sizeof(cl_int), &Hc));
      CL_CHECK(clSetKernelArg(k_slice, 7, sizeof(cl_int), &Vc));
      CL_CHECK(clSetKernelArg(k_slice, 8, sizeof(cl_int), &tok));

      CL_CHECK(clEnqueueNDRangeKernel(queue, k_slice, 1, nullptr, &global, &local, 0, nullptr, nullptr));
      CL_CHECK(clFinish(queue));

      CL_CHECK(clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, 0, VOCAB_SIZE*sizeof(float), logits.data(), 0, nullptr, nullptr));

      // repetition penalty (optional)
      if (penalty > 0.0f) {
        for (int h : history) logits[h] -= penalty;
      }

      softmax_host(logits, temperature);
      int next = topk_sample_host(logits, top_k);
      std::cout << vocab[next] << " ";
      history.push_back(next);
      if ((int)history.size() > 5) history.erase(history.begin());

      current_token = next;
    }
    std::cout << "\n\nDone.\n";
  }

  cleanup_all(k_persist, k_slice, program, w1_buf, b1_buf, w2_buf, b2_buf,
              emb_buf, io_buf, logits_buf, queue, context);
  return 0;
}
