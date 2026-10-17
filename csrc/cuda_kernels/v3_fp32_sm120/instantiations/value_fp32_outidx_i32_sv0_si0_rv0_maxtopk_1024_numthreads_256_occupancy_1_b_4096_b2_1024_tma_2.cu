#include "../../v3_fp32/topk_select.cuh"

namespace topk_select_fp32 {

template
void run_topk_select_kernel<
    TopkSelectConfig<float, int32_t, false,
                     false, false,
                     1024, 256, 1, 4096, 1024, 2, 512, 1, 5242880>
>(const TopkSelectArgs &args);

}   // namespace topk_select_fp32
