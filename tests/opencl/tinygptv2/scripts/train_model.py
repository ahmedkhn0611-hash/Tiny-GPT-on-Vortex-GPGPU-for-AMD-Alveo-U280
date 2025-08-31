import numpy as np

# === Step 1: Define vocabulary ===
VOCAB = [
    "gpt", "xrt", "vortex", "fpga", "kernel", "inference", "runs", "executes",
    "tinygpt", "opencl", "thread", "warp", "core", "fast", "slow", "int8",
    "matvec", "quant", "weight", "bias", "input", "token", "prompt", "next",
    "device", "platform", "bitstream", "compile", "loop", "memory", "clock",
    "latency", "model", "load", "store", "u280", "gemm", "sgemm", "activation",
    "softmax", "output"
]

VOCAB_SIZE = len(VOCAB)
HIDDEN_DIM = 16
LR = 0.05
EPOCHS = 1000

token_to_id = {tok: i for i, tok in enumerate(VOCAB)}
id_to_token = {i: tok for tok, i in token_to_id.items()}

# === Step 2: Longer, richer training sequences ===
SEQUENCES = [
    ["gpt", "runs", "inference", "on", "vortex", "core", "using", "int8", "matvec", "activation", "softmax", "output"],
    ["xrt", "executes", "kernel", "on", "fpga", "platform", "for", "inference", "tasks", "with", "low", "latency"],
    ["bitstream", "loads", "onto", "device", "before", "model", "runs", "int8", "matvec", "operation"],
    ["tinygpt", "predicts", "next", "token", "from", "prompt", "using", "model", "stored", "in", "memory"],
    ["opencl", "kernel", "executes", "matvec", "on", "warp", "thread", "in", "core", "of", "vortex"],
    ["prompt", "token", "next", "token", "predicted", "by", "softmax", "output", "layer"],
    ["load", "model", "weights", "from", "memory", "into", "device", "before", "running", "inference"],
    ["clock", "speed", "and", "latency", "affect", "fpga", "performance", "in", "kernel", "execution"],
    ["compile", "bitstream", "for", "target", "platform", "before", "model", "inference"],
    ["core", "runs", "int8", "matvec", "with", "quant", "weight", "and", "bias", "values"],
    ["thread", "executes", "on", "warp", "inside", "fpga", "device", "to", "perform", "gemm"],
    ["tinygpt", "uses", "embedding", "and", "weights", "to", "generate", "next", "token", "from", "prompt"],
    ["activation", "softmax", "output", "used", "for", "classification", "in", "model"],
    ["memory", "stores", "model", "parameters", "for", "quick", "load", "into", "core"],
    ["device", "executes", "inference", "using", "bitstream", "loaded", "via", "xrt", "tools"],
    ["load", "weights", "bias", "into", "matvec", "operation", "before", "model", "runs"],
    ["prompt", "executes", "through", "embedding", "into", "matvec", "producing", "next", "token"],
    ["gpt", "uses", "softmax", "to", "predict", "token", "from", "quantized", "activation", "vector"],
    ["opencl", "kernel", "launches", "from", "xrt", "with", "inputs", "from", "host"],
    ["tinygpt", "runs", "loop", "over", "tokens", "predicting", "each", "with", "softmax", "and", "matvec"],
    ["bitstream", "load", "executes", "onto", "fpga", "from", "compile", "step", "on", "host"],
    ["device", "memory", "stores", "u280", "model", "compiled", "using", "opencl", "tools"],
    ["core", "executes", "thread", "matvec", "activation", "softmax", "output", "token"],
    ["xrt", "loads", "bitstream", "before", "running", "kernel", "inference"],
    ["int8", "matvec", "used", "in", "quant", "inference", "by", "tinygpt"],
    ["model", "load", "store", "weights", "bias", "into", "device"],
    ["softmax", "output", "activation", "used", "in", "next", "token", "prediction"],
    ["fpga", "platform", "runs", "gpt", "kernel", "with", "int8", "speed"],
    ["store", "quant", "weight", "bias", "into", "memory", "before", "execution"],
    ["sgemm", "runs", "on", "device", "using", "int8", "weights"],
    ["load", "embedding", "before", "running", "inference", "on", "prompt"],
]

# === Step 3: Create (input, target) pairs ===
pairs = []
for seq in SEQUENCES:
    ids = [token_to_id[w] for w in seq if w in token_to_id]
    for i in range(len(ids) - 1):
        pairs.append((ids[i], ids[i + 1]))

print(f"✅ Created {len(pairs)} training pairs.")

# === Step 4: Init model params ===
np.random.seed(42)
embedding = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32)
W1 = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32)
b1 = np.zeros((HIDDEN_DIM,), dtype=np.float32)
W2 = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32)
b2 = np.zeros((VOCAB_SIZE,), dtype=np.float32)

# === Step 5: Training ===
for epoch in range(EPOCHS):
    total_loss = 0
    for x_id, y_id in pairs:
        x_embed = embedding[x_id]
        h1 = np.tanh(np.dot(W1, x_embed) + b1)
        logits = np.dot(W2, h1) + b2

        # Softmax + cross-entropy
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        loss = -np.log(probs[y_id] + 1e-8)
        total_loss += loss

        # Backprop
        dlogits = probs
        dlogits[y_id] -= 1

        dW2 = np.outer(dlogits, h1)
        db2 = dlogits
        dh1 = np.dot(W2.T, dlogits)
        dtanh = (1 - h1 ** 2) * dh1

        dW1 = np.outer(dtanh, x_embed)
        db1 = dtanh
        dembed = np.dot(W1.T, dtanh)

        # Update
        W2 -= LR * dW2
        b2 -= LR * db2
        W1 -= LR * dW1
        b1 -= LR * db1
        embedding[x_id] -= LR * dembed

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(pairs):.4f}")

# === Step 6: Save model ===
np.save("embedding.npy", embedding)
np.save("weights1.npy", W1)
np.save("bias1.npy", b1)
np.save("weights2.npy", W2)
np.save("bias2.npy", b2)

print("\n✅ Exported model: embedding.npy, weights1.npy, bias1.npy, weights2.npy, bias2.npy")
