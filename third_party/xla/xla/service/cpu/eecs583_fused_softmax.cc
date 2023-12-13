#include "absl/base/dynamic_annotations.h"
#include "dnnl.hpp"
#include "tsl/util/onednn_threadpool.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/executable_run_options.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/runtime_lightweight_check.h"

namespace xla {
namespace cpu {

using namespace dnnl;

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EECS583FusedSoftmax(
    const void* run_options_ptr, void* input, void* output) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options != nullptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tsl::OneDnnThreadPool thread_pool(
      run_options->intra_op_thread_pool()->getPool(), false);
  engine cpu_engine(engine::kind::cpu, 0);
  auto onednn_stream = stream(cpu_engine);
  MemrefInfo input_minfo(input);
  MemrefInfo output_minfo(output);
  auto src_md = input_minfo.GetOneDnnMemDesc();
  auto dst_md = output_minfo.GetOneDnnMemDesc();
  auto src_mem = memory(src_md, cpu_engine, input_minfo.Data());
  auto dst_mem = memory(dst_md, cpu_engine, output_minfo.Data());
  int axis = (input_minfo.GetOneDnnDims().size()) - 1;
  auto softmax_pd = softmax_forward::primitive_desc(
      cpu_engine, prop_kind::forward_inference,
      dnnl::algorithm::softmax_accurate, src_md, dst_md, axis);
  auto softmax_prim = softmax_forward(softmax_pd);
  std::unordered_map<int, memory> softmax_args;
  softmax_args.insert({DNNL_ARG_SRC, src_mem});
  softmax_args.insert({DNNL_ARG_DST, dst_mem});
  softmax_prim.execute(onednn_stream, softmax_args);
}
}  
}  
