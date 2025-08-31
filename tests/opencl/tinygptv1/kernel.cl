__kernel void matvec2layer(
    __global const float* W1,   // H x H (row-major)
    __global const float* B1,   // H
    __global const float* W2,   // V x H (row-major)
    __global const float* B2,   // V
    __global const float* x,    // H
    __global float*       h1,   // H
    __global float*       out,  // V
    const int H,
    const int V,
    const int phase          // 0 = compute h1, 1 = compute out
) {
    int gid = get_global_id(0);

    if (phase == 0) {
        if (gid < H) {
            float acc = 0.0f;
            const __global float* row = W1 + gid * H;
            for (int k = 0; k < H; ++k)
                acc += row[k] * x[k];
            acc += B1[gid];

            // tanh(acc)
            float e2x = exp(-2.0f * acc);
            h1[gid] = (1.0f - e2x) / (1.0f + e2x);
        }
    } else if (phase == 1) {
        if (gid < V) {
            float acc = 0.0f;
            const __global float* row = W2 + gid * H;
            for (int k = 0; k < H; ++k)
                acc += row[k] * h1[k];
            out[gid] = acc + B2[gid];
        }
    }
}
