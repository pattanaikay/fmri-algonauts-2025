Floating-Point Precision in Training

Floating-point precision controls how numbers are represented in GPU memory. FP32 (single precision) uses 32 bits per number (1 sign, 8 exponent, 23 mantissa), while FP16 (float16) and BF16 (bfloat16) use 16 bits. FP16 has 5 exponent bits and 10 mantissa bits, giving smaller dynamic range but higher precision bits than BF16. In contrast, BF16 has the same 8-bit exponent as FP32 (8+7+1=16 bits) but only 7 mantissa bits, so it retains FP32’s wide range but with lower precision. Modern NVIDIA GPUs also support TF32 (TensorFloat-32) on Ampere/Hopper: this uses 8 exponent bits and 10 mantissa bits (like FP32 range but reduced precision) and is enabled by default for tensor core ops on Ampere. In practice, TF32 can accelerate FP32 workloads with essentially no code changes
developer.nvidia.com
huggingface.co
.

Choosing lower-precision (FP16/BF16/TF32) dramatically reduces memory usage and boosts speed on GPUs with Tensor Cores. Storing tensors in 16-bit precision typically cuts memory by about half compared to FP32
acecloud.ai
. For example, on an NVIDIA A100 the tensor-core throughput is ~312 TFLOPS in FP16/BF16 vs ~19.5 TFLOPS in FP32 (over 15× more compute throughput)
acecloud.ai
. In practice, mixed-precision training (with FP16 or BF16) often gives ~2–3× higher training speed on modern GPUs, and PyTorch reports speedups of 1.5–5.5× on V100 and 1.3–2.5× on A100 when switching from FP32 to mixed precision
pytorch.org
acecloud.ai
. Enabling TF32 on Ampere/Hopper also speeds up FP32 matrix multiplies (NVIDIA shows up to 2–6× faster FP32 training on A100 vs V100 with TF32)
developer.nvidia.com
developer.nvidia.com
.

However, lower precision can affect convergence stability and numerical accuracy. FP16’s narrower range (max ≈6.55×10^4) can cause overflow/underflow; gradient values may underflow to zero and cause training to “stall” unless loss scaling is used. PyTorch’s AMP (torch.cuda.amp) automatically keeps sensitive ops (layer norms, softmax, losses) in FP32 to improve stability
medium.com
medium.com
. BF16 avoids many of these issues by virtue of its large exponent range (same as FP32); it almost never overflows and usually does not require manual loss scaling. In practice, BF16 often trains “out of the box” with similar accuracy to FP32, while FP16 may need GradScaler adjustments
medium.com
medium.com
. For example, many large models (BERT, GPT, ViT, etc.) now default to BF16 on hardware like A100/H100 to ensure stability, since BF16 “behaves closer to FP32”
acecloud.ai
medium.com
.

Critically, when done correctly mixed precision need not hurt final model quality. NVIDIA reports that TF32 gives essentially identical accuracy to FP32 on a wide range of models (differences are within run-to-run variance)
developer.nvidia.com
huggingface.co
. Likewise, numerous experiments find that using FP16/BF16 (with appropriate scaling) yields negligible loss in final performance – “accuracy impact: none, if done correctly”
medium.com
. In summary: FP16/BF16 training can double or triple speed and halve memory, with minimal accuracy change if using AMP (grad scaling for FP16, or BF16 to avoid scaling). TF32 boosts FP32 throughput on Ampere without code changes, with no loss in convergence
developer.nvidia.com
huggingface.co
. For downstream ridge regression (a linear solve), it is safest to use higher precision (e.g. FP32 or double) when computing the solution, since those closed-form solvers can be sensitive to round-off. In practice one often extracts features via the transformer (in FP16/BF16) and then performs the ridge fit in full precision to ensure numerical stability.