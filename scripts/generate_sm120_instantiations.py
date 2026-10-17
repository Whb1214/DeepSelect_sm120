"""Generate the reduced-shared-memory V3 specializations used by SM120."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_VOCAB_SIZE = 5 * 1024 * 1024


def write_instantiation(
    directory: str,
    include: str,
    namespace: str,
    value_type: str,
    index_type: str,
    sorted_value: bool,
    sorted_index: bool,
    return_value: bool,
    max_topk: int,
) -> None:
    abbrev = "bf16" if value_type == "nv_bfloat16" else "fp32"
    idx_abbrev = "i32" if index_type == "int32_t" else "i64"
    filename = (
        f"value_{abbrev}_outidx_{idx_abbrev}_sv{int(sorted_value)}"
        f"_si{int(sorted_index)}_rv{int(return_value)}_maxtopk_{max_topk}"
        f"_numthreads_256_occupancy_1_b_4096_b2_1024_tma_2.cu"
    )
    path = ROOT / "csrc/cuda_kernels" / directory / "instantiations" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#include \"{include}\"\n
namespace {namespace} {{

template
void run_topk_select_kernel<
    TopkSelectConfig<{value_type}, {index_type}, {str(sorted_value).lower()},
                     {str(sorted_index).lower()}, {str(return_value).lower()},
                     {max_topk}, 256, 1, 4096, 1024, 2, 512, 1, {MAX_VOCAB_SIZE}>
>(const TopkSelectArgs &args);

}}   // namespace {namespace}
""",
        encoding="utf-8",
    )


for index_type in ("int32_t", "int64_t"):
    for sorted_index in (False, True):
        for return_value in (False, True):
            for max_topk in (512, 1024):
                write_instantiation(
                    "v3_sm120",
                    "../../v3/topk_select.cuh",
                    "topk_select_bf16_normal",
                    "nv_bfloat16",
                    index_type,
                    False,
                    sorted_index,
                    return_value,
                    max_topk,
                )

    for sorted_index in (False, True):
        for return_value in (False, True):
            for max_topk in (512, 1024):
                write_instantiation(
                    "v3_fp32_sm120",
                    "../../v3_fp32/topk_select.cuh",
                    "topk_select_fp32",
                    "float",
                    index_type,
                    False,
                    sorted_index,
                    return_value,
                    max_topk,
                )
    for max_topk in (512, 1024):
        write_instantiation(
            "v3_fp32_sm120",
            "../../v3_fp32/topk_select.cuh",
            "topk_select_fp32",
            "float",
            index_type,
            True,
            False,
            True,
            max_topk,
        )
