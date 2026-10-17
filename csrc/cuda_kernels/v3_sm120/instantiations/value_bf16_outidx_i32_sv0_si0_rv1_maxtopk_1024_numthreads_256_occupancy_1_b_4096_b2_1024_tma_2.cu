#include "../../v3/topk_select.cuh"

namespace topk_select_bf16_normal {

template
void run_topk_select_kernel<
    TopkSelectConfig<nv_bfloat16, int32_t, false,
                     false, true,
                     1024, 256, 1, 4096, 1024, 2, 512, 1, 5242880>
>(const TopkSelectArgs &args);

}   // namespace topk_select_bf16_normal
