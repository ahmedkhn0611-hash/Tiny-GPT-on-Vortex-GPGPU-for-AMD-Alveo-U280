// kernel.cl — TinyGPT fused kernels (CL 1.2)

// ====== config (fits your current H=16, V=41) ======
#define D_MAX       64     // >= H
#define VOCAB_MAX  128     // >= V

// ====== helpers reused by both kernels ======
inline float fast_tanh(float x) { return tanh(x); }

inline float dot_row_vec4(const __global float* restrict w_row,
                          const __local  float* restrict x_loc,
                          int d) {
  const __global float4* w4 = (const __global float4*)w_row;
  const __local  float4* x4 = (const __local  float4*)x_loc;
  int d4 = d >> 2;
  float acc = 0.0f;
  #pragma unroll
  for (int i=0;i<d4;++i) {
    float4 a = w4[i];
    float4 b = x4[i];
    acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
  }
  for (int i=(d4<<2); i<d; ++i) acc += w_row[i] * x_loc[i];
  return acc;
}

inline float reduce_max_local(__local float* vals, int n) {
  float m = vals[0];
  for (int i=1;i<n;++i) m = fmax(m, vals[i]);
  return m;
}

inline int softmax_sample_top1(__local float* logits, int vocab) {
  float m = reduce_max_local(logits, vocab);
  float s = 0.0f;
  for (int i=0;i<vocab;++i) { logits[i] = exp(logits[i] - m); s += logits[i]; }
  float best = -1.0f; int best_id = 0;
  for (int i=0;i<vocab;++i) {
    float p = logits[i] / s;
    if (p > best) { best = p; best_id = i; }
  }
  return best_id;
}

// ====== Kernel 1: persistent fused (best for 1-core runs) ======
__kernel void tinygpt_persist_fused(
  __global const float* restrict W1, // [H x H], row-major
  __global const float* restrict B1, // [H]
  __global const float* restrict W2, // [V x H], row-major
  __global const float* restrict B2, // [V]
  __global const float* restrict E,  // [V x H] embedding
  __global int*         restrict io_tokens, // [T+1], io_tokens[0]=prompt
  const int H,
  const int V,
  const int T)
{
  if (H > D_MAX || V > VOCAB_MAX) return;
  // This baseline assumes a single work-group; host should launch global=local (1 WG).
  if (get_num_groups(0) != 1)     return;

  const int lid   = get_local_id(0);
  const int lsize = get_local_size(0);

  __local float x_loc[D_MAX];
  __local float hidden_loc[D_MAX];
  __local float logits_loc[VOCAB_MAX];

  int token = io_tokens[0];

  for (int t = 0; t < T; ++t) {
    // 1) Embed
    if (lid == 0) {
      const __global float* e_row = E + token * H;
      int i=0;
      for (; i+3 < H; i+=4) {
        float4 v = vload4(0, e_row + i);
        vstore4(v, 0, x_loc + i);
      }
      for (; i < H; ++i) x_loc[i] = e_row[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2) hidden = tanh(W1*x + B1)
    for (int r = lid; r < H; r += lsize) {
      const __global float* w1_row = W1 + r * H;
      float acc = dot_row_vec4(w1_row, x_loc, H) + B1[r];
      hidden_loc[r] = fast_tanh(acc);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3) logits = W2*hidden + B2
    for (int v = lid; v < V; v += lsize) {
      const __global float* w2_row = W2 + v * H;
      float acc = 0.0f;
      int i=0;
      for (; i+3 < H; i+=4) {
        float4 w = vload4(0, w2_row + i);
        float4 h = vload4(0, hidden_loc + i);
        acc += w.x*h.x + w.y*h.y + w.z*h.z + w.w*h.w;
      }
      for (; i < H; ++i) acc += w2_row[i] * hidden_loc[i];
      __local float* row = logits_loc; // alias to keep code simple
      row[v] = acc + B2[v];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4) softmax + sample (top-1)
    if (lid == 0) {
      int next_token = softmax_sample_top1(logits_loc, V);
      io_tokens[t+1] = next_token;
      token = next_token;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// ====== Kernel 2: multi-WG slice (scales across cores) ======
__kernel void ffn_to_logits_slice(
  __global const float* restrict W1, // [H x H]
  __global const float* restrict B1, // [H]
  __global const float* restrict W2, // [V x H]
  __global const float* restrict B2, // [V]
  __global const float* restrict E,  // [V x H]
  __global float*       restrict logits, // [V], overwritten each token
  const int H, const int V, const int token_id)
{
  const int G   = get_num_groups(0);
  const int gid = get_group_id(0);
  const int lid = get_local_id(0);
  const int lsz = get_local_size(0);

  if (H > D_MAX || V > VOCAB_MAX) return;

  __local float x_loc[D_MAX];
  __local float hidden_loc[D_MAX];

  // 1) Embed
  if (lid == 0) {
    const __global float* e_row = E + token_id * H;
    int i=0;
    for (; i+3 < H; i+=4) { float4 v=vload4(0, e_row+i); vstore4(v,0,x_loc+i); }
    for (; i < H; ++i) x_loc[i] = e_row[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // 2) hidden = tanh(W1*x + B1)
  for (int r = lid; r < H; r += lsz) {
    const __global float* w1_row = W1 + r * H;
    float acc = dot_row_vec4(w1_row, x_loc, H) + B1[r];
    hidden_loc[r] = fast_tanh(acc);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // 3) Disjoint vocab slice per group
  const int chunk = (V + G - 1) / G;
  const int begin = gid * chunk;
  const int end   = min(begin + chunk, V);

  for (int v = begin + lid; v < end; v += lsz) {
    const __global float* w2_row = W2 + v * H;
    float acc = 0.0f;
    int i=0;
    for (; i+3 < H; i+=4) {
      float4 w = vload4(0, w2_row + i);
      float4 h = vload4(0, hidden_loc + i);
      acc += w.x*h.x + w.y*h.y + w.z*h.z + w.w*h.w;
    }
    for (; i < H; ++i) acc += w2_row[i] * hidden_loc[i];
    logits[v] = acc + B2[v];
  }
}
