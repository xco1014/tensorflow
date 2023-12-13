#include "tsl/platform/cpu_info.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;

std::optional<HloInstruction*> MatchSoftmax(HloInstruction* instr) {
  HloInstruction* le, re, lp, rp;

  if (!Match(
          instr,
          m::Divide(m::Exp(&le, m::Op()),
          m::Broadcast(m::Reshape(m::Broadcast(m::Reshape(
          m::Reduce(m::Exp(&re, m::Op()), m::Op())
          .WithPredicate([](const HloInstruction* reduce) {
          HloComputation* reducer = reduce->to_apply();
        return (reducer->root_instruction()->opcode() ==
        HloOpcode::kAdd &&
        reduce->dimensions().size() == 1 &&
        reduce->dimensions()[0] !=
        reduce->shape().rank() - 1);}).WithOneUse()))))))) {
    return std::nullopt;
  }

  if (!Match(
          le->mutable_operand(0),
          m::Subtract(m::Op(&lp),
          m::Broadcast(m::Reshape(
          m::Broadcast(m::Reshape(
          m::Reduce(m::Op(&rp), m::Op())
          .WithPredicate([](const HloInstruction* reduce) {
          HloComputation* reducer = reduce->to_apply();
        return (reducer->root_instruction()->opcode() ==
        HloOpcode::kMaximum &&
        reduce->dimensions().size() == 1 &&
        reduce->dimensions()[0] !=
        reduce->shape().rank() - 1);
        }).WithOneUse())))).WithOneUse()).WithOneUse())) {
    return std::nullopt;
  }
  return lp;
}
}  

class EECS583Visitor : public DfsHloRewriteVisitor {
 public:
  Status HandleDivide(HloInstruction* divide_instr) override {
    if (divide_instr->HasControlDependencies()) return OkStatus();
    if (!IsSupportedType(divide_instr->shape().element_type()))
      return OkStatus();
    std::optional<HloInstruction*> producer;
    bool found_pattern = false;
    if (producer = MatchSoftmax(divide_instr))
      found_pattern = true;
    if (!found_pattern) return OkStatus();
    const Shape& output_shape = divide_instr->shape();
    HloInstruction* softmax_call =
        divide_instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {producer.value()}, "__eecs583$fusedsoftmax"));
    TF_RETURN_IF_ERROR(ReplaceInstruction(divide_instr, softmax_call));
    return OkStatus();
  }
};

StatusOr<bool> EECS583Optimization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  EECS583Visitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}
}  
}  

