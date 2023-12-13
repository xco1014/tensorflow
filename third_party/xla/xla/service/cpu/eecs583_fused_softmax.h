#ifndef XLA_SERVICE_CPU_EECS583_FUSED_SOFTMAX_H_
#define XLA_SERVICE_CPU_EECS583_FUSED_SOFTMAX_H_

namespace xla {
namespace cpu {

extern "C" {
extern void __xla_cpu_runtime_EECS583FusedSoftmax(const void* run_options_ptr,
                                            void* input, void* output);
}  

}
}
#endif  
