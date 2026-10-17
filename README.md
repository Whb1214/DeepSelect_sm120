# DeepSelect SM120 Port

An experimental CUDA port that builds and runs DeepSelect TopK on NVIDIA RTX 50 Series GPUs (compute capability SM120). The port was built and tested on an RTX 5060 Laptop GPU under WSL2.

This repository is based on [DeepSeek-AI/DeepSelect](https://github.com/deepseek-ai/DeepSelect). It retains the upstream V3 selection algorithm and CUDA kernels; the changes here add SM120 build support and architecture-specific configurations. This project does not claim authorship of the original TopK algorithm.

## What changed in this port

- Added an `sm120` architecture option to the CUDA extension build and generated dedicated BF16/FP32 kernel instantiations.
- Added a lower shared-memory V3 configuration for SM120 and dispatch paths for those configurations.
- Disabled the upstream 16-CTA BF16 cluster path for SM120.
- Prevented SM120 from selecting the SM100-family 256-bit global load/store path; the port uses the 128-bit path.
- Added explicit current limits: `topk <= 1024` and `vocab_size < 5,242,880`.

The current specializations use 256 threads, occupancy 1, a 4096-element block, a 1024-element compaction threshold, and TMA depth 2. These are initial compatibility settings, not a claim that the configuration is fully tuned for every RTX 50 Series GPU.

## Build on Linux or WSL2

Requirements:

- NVIDIA RTX 50 Series GPU and a compatible NVIDIA driver
- CUDA Toolkit 12.8 or newer
- CUDA-enabled PyTorch and Triton
- `pciutils` (`lspci`), including on WSL2

Clone this branch and build the extension:

```bash
git clone --branch sm120-rtx50 --recurse-submodules https://github.com/Whb1214/DeepSelect_sm120.git
cd DeepSelect_sm120

DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA \
DEEP_SELECT_CUDA_ARCH=sm120 \
NVCC_THREADS=4 MAX_JOBS=4 \
python -m pip install --no-build-isolation --no-deps -v .
```

Run the following from outside the checkout after activating the same Python environment:

```python
import torch
import deep_select

x = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
values, indices = deep_select.topk(
    x, 512, return_value=False, indices_type=torch.int32
)
print(values, indices.shape, indices.dtype)
```

On WSL2, force `DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA` because PCI auto-detection may not see the Windows GPU. When checking the installed package, run Python outside the checkout (for example, `cd ~`) so the source tree does not shadow the installed extension.

## Validation on RTX 5060 Laptop GPU

Environment: WSL2 Ubuntu 24.04, CUDA Toolkit 12.8, PyTorch 2.11.0+cu128, RTX 5060 Laptop GPU (SM120).

- BF16 and FP32 smoke checks matched `torch.topk`, including `topk=1024`, sorted-index output, and indices-only output.
- Ran 28 representative performance cases against `torch.topk` using generated `torch.randn` inputs, 8 warmups, and 30 timed runs per case.
- Observed speedups ranged from **1.37x to 6.22x**, with a **4.76x median** on this GPU.

The benchmark covers a selected subset of shapes, not the full upstream performance matrix. Results are specific to this device, build, and input distribution; broader coverage and repeated runs are still needed. Raw results are in [`benchmark_results_sm120_rtx5060.csv`](benchmark_results_sm120_rtx5060.csv).

## Current limitations

- `topk` is limited to 1024 and `vocab_size` must be below 5,242,880.
- The current build contains only the generated SM120 BF16 and FP32 specializations described above.
- Performance has not been tuned or characterized across all RTX 50 Series models and workloads.

## Upstream attribution and license

The TopK algorithm, base V3 CUDA implementation, and original project documentation are from [DeepSeek-AI/DeepSelect](https://github.com/deepseek-ai/DeepSelect). Please refer to the upstream project for algorithm details and its citation. The upstream MIT license is retained in [`LICENSE`](LICENSE).
