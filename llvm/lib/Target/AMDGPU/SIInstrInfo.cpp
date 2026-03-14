//===- SIInstrInfo.cpp - SI Instruction Information  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// SI Implementation of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "SIInstrInfo.h"
#include "AMDGPU.h"
#include "AMDGPUInstrInfo.h"
#include "AMDGPULaneMaskUtils.h"
#include "GCNHazardRecognizer.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-instr-info"

#define GET_INSTRINFO_CTOR_DTOR
#include "AMDGPUGenInstrInfo.inc"

namespace llvm::AMDGPU {
#define GET_D16ImageDimIntrinsics_IMPL
#define GET_ImageDimIntrinsicTable_IMPL
#define GET_RsrcIntrinsics_IMPL
#include "AMDGPUGenSearchableTables.inc"
} // namespace llvm::AMDGPU

// Must be at least 4 to be able to branch over minimum unconditional branch
// code. This is only for making it possible to write reasonably small tests for
// long branches.
static cl::opt<unsigned>
BranchOffsetBits("amdgpu-s-branch-bits", cl::ReallyHidden, cl::init(16),
                 cl::desc("Restrict range of branch instructions (DEBUG)"));

static cl::opt<bool> Fix16BitCopies(
  "amdgpu-fix-16-bit-physreg-copies",
  cl::desc("Fix copies between 32 and 16 bit registers by extending to 32 bit"),
  cl::init(true),
  cl::ReallyHidden);

SIInstrInfo::SIInstrInfo(const GCNSubtarget &ST)
    : AMDGPUGenInstrInfo(ST, RI, AMDGPU::ADJCALLSTACKUP,
                         AMDGPU::ADJCALLSTACKDOWN),
      RI(ST), ST(ST) {
  SchedModel.init(&ST);
}

//===----------------------------------------------------------------------===//
// TargetInstrInfo callbacks
//===----------------------------------------------------------------------===//

static unsigned getNumOperandsNoGlue(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Glue)
    --N;
  return N;
}

/// Returns true if both nodes have the same value for the given
///        operand \p Op, or if both nodes do not have this operand.
static bool nodesHaveSameOperandValue(SDNode *N0, SDNode *N1,
                                      AMDGPU::OpName OpName) {
  unsigned Opc0 = N0->getMachineOpcode();
  unsigned Opc1 = N1->getMachineOpcode();

  int Op0Idx = AMDGPU::getNamedOperandIdx(Opc0, OpName);
  int Op1Idx = AMDGPU::getNamedOperandIdx(Opc1, OpName);

  if (Op0Idx == -1 && Op1Idx == -1)
    return true;


  if ((Op0Idx == -1 && Op1Idx != -1) ||
      (Op1Idx == -1 && Op0Idx != -1))
    return false;

  // getNamedOperandIdx returns the index for the MachineInstr's operands,
  // which includes the result as the first operand. We are indexing into the
  // MachineSDNode's operands, so we need to skip the result operand to get
  // the real index.
  --Op0Idx;
  --Op1Idx;

  return N0->getOperand(Op0Idx) == N1->getOperand(Op1Idx);
}

static bool canRemat(const MachineInstr &MI) {

  if (SIInstrInfo::isVOP1(MI) || SIInstrInfo::isVOP2(MI) ||
      SIInstrInfo::isVOP3(MI) || SIInstrInfo::isSDWA(MI) ||
      SIInstrInfo::isSALU(MI))
    return true;

  if (SIInstrInfo::isSMRD(MI)) {
    return !MI.memoperands_empty() &&
           llvm::all_of(MI.memoperands(), [](const MachineMemOperand *MMO) {
             return MMO->isLoad() && MMO->isInvariant();
           });
  }

  return false;
}

bool SIInstrInfo::isReMaterializableImpl(
    const MachineInstr &MI) const {

  if (canRemat(MI)) {
    // Normally VALU use of exec would block the rematerialization, but that
    // is OK in this case to have an implicit exec read as all VALU do.
    // We really want all of the generic logic for this except for this.

    // Another potential implicit use is mode register. The core logic of
    // the RA will not attempt rematerialization if mode is set anywhere
    // in the function, otherwise it is safe since mode is not changed.

    // There is difference to generic method which does not allow
    // rematerialization if there are virtual register uses. We allow this,
    // therefore this method includes SOP instructions as well.
    if (!MI.hasImplicitDef() &&
        MI.getNumImplicitOperands() == MI.getDesc().implicit_uses().size() &&
        !MI.mayRaiseFPException())
      return true;
  }

  return TargetInstrInfo::isReMaterializableImpl(MI);
}

// Returns true if the scalar result of a VALU instruction depends on exec.
bool SIInstrInfo::resultDependsOnExec(const MachineInstr &MI) const {
  // Ignore comparisons which are only used masked with exec.
  // This allows some hoisting/sinking of VALU comparisons.
  if (MI.isCompare()) {
    const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::sdst);
    if (!Dst)
      return true;

    Register DstReg = Dst->getReg();
    if (!DstReg.isVirtual())
      return true;

    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    for (MachineInstr &Use : MRI.use_nodbg_instructions(DstReg)) {
      switch (Use.getOpcode()) {
      case AMDGPU::S_AND_SAVEEXEC_B32:
      case AMDGPU::S_AND_SAVEEXEC_B64:
        break;
      case AMDGPU::S_AND_B32:
      case AMDGPU::S_AND_B64:
        if (!Use.readsRegister(AMDGPU::EXEC, /*TRI=*/nullptr))
          return true;
        break;
      default:
        return true;
      }
    }
    return false;
  }

  // If it is not convergent it does not depend on EXEC.
  if (!MI.isConvergent())
    return false;

  switch (MI.getOpcode()) {
  default:
    break;
  case AMDGPU::V_READFIRSTLANE_B32:
    return true;
  }

  return false;
}

bool SIInstrInfo::isIgnorableUse(const MachineOperand &MO) const {
  // Any implicit use of exec by VALU is not a real register read.
  return MO.getReg() == AMDGPU::EXEC && MO.isImplicit() &&
         isVALU(*MO.getParent()) && !resultDependsOnExec(*MO.getParent());
}

bool SIInstrInfo::isSafeToSink(MachineInstr &MI,
                               MachineBasicBlock *SuccToSinkTo,
                               MachineCycleInfo *CI) const {
  // Allow sinking if MI edits lane mask (divergent i1 in sgpr).
  if (MI.getOpcode() == AMDGPU::SI_IF_BREAK)
    return true;

  MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  // Check if sinking of MI would create temporal divergent use.
  for (auto Op : MI.uses()) {
    if (Op.isReg() && Op.getReg().isVirtual() &&
        RI.isSGPRClass(MRI.getRegClass(Op.getReg()))) {
      MachineInstr *SgprDef = MRI.getVRegDef(Op.getReg());

      // SgprDef defined inside cycle
      MachineCycle *FromCycle = CI->getCycle(SgprDef->getParent());
      if (FromCycle == nullptr)
        continue;

      MachineCycle *ToCycle = CI->getCycle(SuccToSinkTo);
      // Check if there is a FromCycle that contains SgprDef's basic block but
      // does not contain SuccToSinkTo and also has divergent exit condition.
      while (FromCycle && !FromCycle->contains(ToCycle)) {
        SmallVector<MachineBasicBlock *, 1> ExitingBlocks;
        FromCycle->getExitingBlocks(ExitingBlocks);

        // FromCycle has divergent exit condition.
        for (MachineBasicBlock *ExitingBlock : ExitingBlocks) {
          if (hasDivergentBranch(ExitingBlock))
            return false;
        }

        FromCycle = FromCycle->getParentCycle();
      }
    }
  }

  return true;
}

bool SIInstrInfo::areLoadsFromSameBasePtr(SDNode *Load0, SDNode *Load1,
                                          int64_t &Offset0,
                                          int64_t &Offset1) const {
  if (!Load0->isMachineOpcode() || !Load1->isMachineOpcode())
    return false;

  unsigned Opc0 = Load0->getMachineOpcode();
  unsigned Opc1 = Load1->getMachineOpcode();

  // Make sure both are actually loads.
  if (!get(Opc0).mayLoad() || !get(Opc1).mayLoad())
    return false;

  // A mayLoad instruction without a def is not a load. Likely a prefetch.
  if (!get(Opc0).getNumDefs() || !get(Opc1).getNumDefs())
    return false;

  if (isDS(Opc0) && isDS(Opc1)) {

    // FIXME: Handle this case:
    if (getNumOperandsNoGlue(Load0) != getNumOperandsNoGlue(Load1))
      return false;

    // Check base reg.
    if (Load0->getOperand(0) != Load1->getOperand(0))
      return false;

    // Skip read2 / write2 variants for simplicity.
    // TODO: We should report true if the used offsets are adjacent (excluded
    // st64 versions).
    int Offset0Idx = AMDGPU::getNamedOperandIdx(Opc0, AMDGPU::OpName::offset);
    int Offset1Idx = AMDGPU::getNamedOperandIdx(Opc1, AMDGPU::OpName::offset);
    if (Offset0Idx == -1 || Offset1Idx == -1)
      return false;

    // XXX - be careful of dataless loads
    // getNamedOperandIdx returns the index for MachineInstrs.  Since they
    // include the output in the operand list, but SDNodes don't, we need to
    // subtract the index by one.
    Offset0Idx -= get(Opc0).NumDefs;
    Offset1Idx -= get(Opc1).NumDefs;
    Offset0 = Load0->getConstantOperandVal(Offset0Idx);
    Offset1 = Load1->getConstantOperandVal(Offset1Idx);
    return true;
  }

  if (isSMRD(Opc0) && isSMRD(Opc1)) {
    // Skip time and cache invalidation instructions.
    if (!AMDGPU::hasNamedOperand(Opc0, AMDGPU::OpName::sbase) ||
        !AMDGPU::hasNamedOperand(Opc1, AMDGPU::OpName::sbase))
      return false;

    unsigned NumOps = getNumOperandsNoGlue(Load0);
    if (NumOps != getNumOperandsNoGlue(Load1))
      return false;

    // Check base reg.
    if (Load0->getOperand(0) != Load1->getOperand(0))
      return false;

    // Match register offsets, if both register and immediate offsets present.
    assert(NumOps == 4 || NumOps == 5);
    if (NumOps == 5 && Load0->getOperand(1) != Load1->getOperand(1))
      return false;

    const ConstantSDNode *Load0Offset =
        dyn_cast<ConstantSDNode>(Load0->getOperand(NumOps - 3));
    const ConstantSDNode *Load1Offset =
        dyn_cast<ConstantSDNode>(Load1->getOperand(NumOps - 3));

    if (!Load0Offset || !Load1Offset)
      return false;

    Offset0 = Load0Offset->getZExtValue();
    Offset1 = Load1Offset->getZExtValue();
    return true;
  }

  // MUBUF and MTBUF can access the same addresses.
  if ((isMUBUF(Opc0) || isMTBUF(Opc0)) && (isMUBUF(Opc1) || isMTBUF(Opc1))) {

    // MUBUF and MTBUF have vaddr at different indices.
    if (!nodesHaveSameOperandValue(Load0, Load1, AMDGPU::OpName::soffset) ||
        !nodesHaveSameOperandValue(Load0, Load1, AMDGPU::OpName::vaddr) ||
        !nodesHaveSameOperandValue(Load0, Load1, AMDGPU::OpName::srsrc))
      return false;

    int OffIdx0 = AMDGPU::getNamedOperandIdx(Opc0, AMDGPU::OpName::offset);
    int OffIdx1 = AMDGPU::getNamedOperandIdx(Opc1, AMDGPU::OpName::offset);

    if (OffIdx0 == -1 || OffIdx1 == -1)
      return false;

    // getNamedOperandIdx returns the index for MachineInstrs.  Since they
    // include the output in the operand list, but SDNodes don't, we need to
    // subtract the index by one.
    OffIdx0 -= get(Opc0).NumDefs;
    OffIdx1 -= get(Opc1).NumDefs;

    SDValue Off0 = Load0->getOperand(OffIdx0);
    SDValue Off1 = Load1->getOperand(OffIdx1);

    // The offset might be a FrameIndexSDNode.
    if (!isa<ConstantSDNode>(Off0) || !isa<ConstantSDNode>(Off1))
      return false;

    Offset0 = Off0->getAsZExtVal();
    Offset1 = Off1->getAsZExtVal();
    return true;
  }

  return false;
}

static bool isStride64(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::DS_READ2ST64_B32:
  case AMDGPU::DS_READ2ST64_B64:
  case AMDGPU::DS_WRITE2ST64_B32:
  case AMDGPU::DS_WRITE2ST64_B64:
    return true;
  default:
    return false;
  }
}

bool SIInstrInfo::getMemOperandsWithOffsetWidth(
    const MachineInstr &LdSt, SmallVectorImpl<const MachineOperand *> &BaseOps,
    int64_t &Offset, bool &OffsetIsScalable, LocationSize &Width,
    const TargetRegisterInfo *TRI) const {
  if (!LdSt.mayLoadOrStore())
    return false;

  unsigned Opc = LdSt.getOpcode();
  OffsetIsScalable = false;
  const MachineOperand *BaseOp, *OffsetOp;
  int DataOpIdx;

  if (isDS(LdSt)) {
    BaseOp = getNamedOperand(LdSt, AMDGPU::OpName::addr);
    OffsetOp = getNamedOperand(LdSt, AMDGPU::OpName::offset);
    if (OffsetOp) {
      // Normal, single offset LDS instruction.
      if (!BaseOp) {
        // DS_CONSUME/DS_APPEND use M0 for the base address.
        // TODO: find the implicit use operand for M0 and use that as BaseOp?
        return false;
      }
      BaseOps.push_back(BaseOp);
      Offset = OffsetOp->getImm();
      // Get appropriate operand, and compute width accordingly.
      DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
      if (DataOpIdx == -1)
        DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data0);
      if (Opc == AMDGPU::DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64)
        Width = LocationSize::precise(64);
      else
        Width = LocationSize::precise(getOpSize(LdSt, DataOpIdx));
    } else {
      // The 2 offset instructions use offset0 and offset1 instead. We can treat
      // these as a load with a single offset if the 2 offsets are consecutive.
      // We will use this for some partially aligned loads.
      const MachineOperand *Offset0Op =
          getNamedOperand(LdSt, AMDGPU::OpName::offset0);
      const MachineOperand *Offset1Op =
          getNamedOperand(LdSt, AMDGPU::OpName::offset1);

      unsigned Offset0 = Offset0Op->getImm() & 0xff;
      unsigned Offset1 = Offset1Op->getImm() & 0xff;
      if (Offset0 + 1 != Offset1)
        return false;

      // Each of these offsets is in element sized units, so we need to convert
      // to bytes of the individual reads.

      unsigned EltSize;
      if (LdSt.mayLoad())
        EltSize = TRI->getRegSizeInBits(*getOpRegClass(LdSt, 0)) / 16;
      else {
        assert(LdSt.mayStore());
        int Data0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data0);
        EltSize = TRI->getRegSizeInBits(*getOpRegClass(LdSt, Data0Idx)) / 8;
      }

      if (isStride64(Opc))
        EltSize *= 64;

      BaseOps.push_back(BaseOp);
      Offset = EltSize * Offset0;
      // Get appropriate operand(s), and compute width accordingly.
      DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
      if (DataOpIdx == -1) {
        DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data0);
        Width = LocationSize::precise(getOpSize(LdSt, DataOpIdx));
        DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data1);
        Width = LocationSize::precise(
            Width.getValue() + TypeSize::getFixed(getOpSize(LdSt, DataOpIdx)));
      } else {
        Width = LocationSize::precise(getOpSize(LdSt, DataOpIdx));
      }
    }
    return true;
  }

  if (isMUBUF(LdSt) || isMTBUF(LdSt)) {
    const MachineOperand *RSrc = getNamedOperand(LdSt, AMDGPU::OpName::srsrc);
    if (!RSrc) // e.g. BUFFER_WBINVL1_VOL
      return false;
    BaseOps.push_back(RSrc);
    BaseOp = getNamedOperand(LdSt, AMDGPU::OpName::vaddr);
    if (BaseOp && !BaseOp->isFI())
      BaseOps.push_back(BaseOp);
    const MachineOperand *OffsetImm =
        getNamedOperand(LdSt, AMDGPU::OpName::offset);
    Offset = OffsetImm->getImm();
    const MachineOperand *SOffset =
        getNamedOperand(LdSt, AMDGPU::OpName::soffset);
    if (SOffset) {
      if (SOffset->isReg())
        BaseOps.push_back(SOffset);
      else
        Offset += SOffset->getImm();
    }
    // Get appropriate operand, and compute width accordingly.
    DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
    if (DataOpIdx == -1)
      DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdata);
    if (DataOpIdx == -1) // LDS DMA
      return false;
    Width = LocationSize::precise(getOpSize(LdSt, DataOpIdx));
    return true;
  }

  if (isImage(LdSt)) {
    auto RsrcOpName =
        isMIMG(LdSt) ? AMDGPU::OpName::srsrc : AMDGPU::OpName::rsrc;
    int SRsrcIdx = AMDGPU::getNamedOperandIdx(Opc, RsrcOpName);
    BaseOps.push_back(&LdSt.getOperand(SRsrcIdx));
    int VAddr0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr0);
    if (VAddr0Idx >= 0) {
      // GFX10 possible NSA encoding.
      for (int I = VAddr0Idx; I < SRsrcIdx; ++I)
        BaseOps.push_back(&LdSt.getOperand(I));
    } else {
      BaseOps.push_back(getNamedOperand(LdSt, AMDGPU::OpName::vaddr));
    }
    Offset = 0;
    // Get appropriate operand, and compute width accordingly.
    DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdata);
    if (DataOpIdx == -1)
      return false; // no return sampler
    Width = LocationSize::precise(getOpSize(LdSt, DataOpIdx));
    return true;
  }

  if (isSMRD(LdSt)) {
    BaseOp = getNamedOperand(LdSt, AMDGPU::OpName::sbase);
    if (!BaseOp) // e.g. S_MEMTIME
      return false;
    BaseOps.push_back(BaseOp);
    OffsetOp = getNamedOperand(LdSt, AMDGPU::OpName::offset);
    Offset = OffsetOp ? OffsetOp->getImm() : 0;
    // Get appropriate operand, and compute width accordingly.
    DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::sdst);
    if (DataOpIdx == -1)
      return false;
    Width = LocationSize::precise(getOpSize(LdSt, DataOpIdx));
    return true;
  }

  if (isFLAT(LdSt)) {
    // Instructions have either vaddr or saddr or both or none.
    BaseOp = getNamedOperand(LdSt, AMDGPU::OpName::vaddr);
    if (BaseOp)
      BaseOps.push_back(BaseOp);
    BaseOp = getNamedOperand(LdSt, AMDGPU::OpName::saddr);
    if (BaseOp)
      BaseOps.push_back(BaseOp);
    Offset = getNamedOperand(LdSt, AMDGPU::OpName::offset)->getImm();
    // Get appropriate operand, and compute width accordingly.
    DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
    if (DataOpIdx == -1)
      DataOpIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdata);
    if (DataOpIdx == -1) // LDS DMA
      return false;
    Width = LocationSize::precise(getOpSize(LdSt, DataOpIdx));
    return true;
  }

  return false;
}

static bool memOpsHaveSameBasePtr(const MachineInstr &MI1,
                                  ArrayRef<const MachineOperand *> BaseOps1,
                                  const MachineInstr &MI2,
                                  ArrayRef<const MachineOperand *> BaseOps2) {
  // Only examine the first "base" operand of each instruction, on the
  // assumption that it represents the real base address of the memory access.
  // Other operands are typically offsets or indices from this base address.
  if (BaseOps1.front()->isIdenticalTo(*BaseOps2.front()))
    return true;

  if (!MI1.hasOneMemOperand() || !MI2.hasOneMemOperand())
    return false;

  auto *MO1 = *MI1.memoperands_begin();
  auto *MO2 = *MI2.memoperands_begin();
  if (MO1->getAddrSpace() != MO2->getAddrSpace())
    return false;

  const auto *Base1 = MO1->getValue();
  const auto *Base2 = MO2->getValue();
  if (!Base1 || !Base2)
    return false;
  Base1 = getUnderlyingObject(Base1);
  Base2 = getUnderlyingObject(Base2);

  if (isa<UndefValue>(Base1) || isa<UndefValue>(Base2))
    return false;

  return Base1 == Base2;
}

bool SIInstrInfo::shouldClusterMemOps(ArrayRef<const MachineOperand *> BaseOps1,
                                      int64_t Offset1, bool OffsetIsScalable1,
                                      ArrayRef<const MachineOperand *> BaseOps2,
                                      int64_t Offset2, bool OffsetIsScalable2,
                                      unsigned ClusterSize,
                                      unsigned NumBytes) const {
  // If the mem ops (to be clustered) do not have the same base ptr, then they
  // should not be clustered
  unsigned MaxMemoryClusterDWords = DefaultMemoryClusterDWordsLimit;
  if (!BaseOps1.empty() && !BaseOps2.empty()) {
    const MachineInstr &FirstLdSt = *BaseOps1.front()->getParent();
    const MachineInstr &SecondLdSt = *BaseOps2.front()->getParent();
    if (!memOpsHaveSameBasePtr(FirstLdSt, BaseOps1, SecondLdSt, BaseOps2))
      return false;

    const SIMachineFunctionInfo *MFI =
        FirstLdSt.getMF()->getInfo<SIMachineFunctionInfo>();
    MaxMemoryClusterDWords = MFI->getMaxMemoryClusterDWords();
  } else if (!BaseOps1.empty() || !BaseOps2.empty()) {
    // If only one base op is empty, they do not have the same base ptr
    return false;
  }

  // In order to avoid register pressure, on an average, the number of DWORDS
  // loaded together by all clustered mem ops should not exceed
  // MaxMemoryClusterDWords. This is an empirical value based on certain
  // observations and performance related experiments.
  // The good thing about this heuristic is - it avoids clustering of too many
  // sub-word loads, and also avoids clustering of wide loads. Below is the
  // brief summary of how the heuristic behaves for various `LoadSize` when
  // MaxMemoryClusterDWords is 8.
  //
  // (1) 1 <= LoadSize <= 4: cluster at max 8 mem ops
  // (2) 5 <= LoadSize <= 8: cluster at max 4 mem ops
  // (3) 9 <= LoadSize <= 12: cluster at max 2 mem ops
  // (4) 13 <= LoadSize <= 16: cluster at max 2 mem ops
  // (5) LoadSize >= 17: do not cluster
  const unsigned LoadSize = NumBytes / ClusterSize;
  const unsigned NumDWords = ((LoadSize + 3) / 4) * ClusterSize;
  return NumDWords <= MaxMemoryClusterDWords;
}

// FIXME: This behaves strangely. If, for example, you have 32 load + stores,
// the first 16 loads will be interleaved with the stores, and the next 16 will
// be clustered as expected. It should really split into 2 16 store batches.
//
// Loads are clustered until this returns false, rather than trying to schedule
// groups of stores. This also means we have to deal with saying different
// address space loads should be clustered, and ones which might cause bank
// conflicts.
//
// This might be deprecated so it might not be worth that much effort to fix.
bool SIInstrInfo::shouldScheduleLoadsNear(SDNode *Load0, SDNode *Load1,
                                          int64_t Offset0, int64_t Offset1,
                                          unsigned NumLoads) const {
  assert(Offset1 > Offset0 &&
         "Second offset should be larger than first offset!");
  // If we have less than 16 loads in a row, and the offsets are within 64
  // bytes, then schedule together.

  // A cacheline is 64 bytes (for global memory).
  return (NumLoads <= 16 && (Offset1 - Offset0) < 64);
}

static void reportIllegalCopy(const SIInstrInfo *TII, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              const DebugLoc &DL, MCRegister DestReg,
                              MCRegister SrcReg, bool KillSrc,
                              const char *Msg = "illegal VGPR to SGPR copy") {
  MachineFunction *MF = MBB.getParent();

  LLVMContext &C = MF->getFunction().getContext();
  C.diagnose(DiagnosticInfoUnsupported(MF->getFunction(), Msg, DL, DS_Error));

  BuildMI(MBB, MI, DL, TII->get(AMDGPU::SI_ILLEGAL_COPY), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}

/// Handle copying from SGPR to AGPR, or from AGPR to AGPR on GFX908. It is not
/// possible to have a direct copy in these cases on GFX908, so an intermediate
/// VGPR copy is required.
static void indirectCopyToAGPR(const SIInstrInfo &TII,
                               MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI,
                               const DebugLoc &DL, MCRegister DestReg,
                               MCRegister SrcReg, bool KillSrc,
                               RegScavenger &RS, bool RegsOverlap,
                               Register ImpDefSuperReg = Register(),
                               Register ImpUseSuperReg = Register()) {
  assert((TII.getSubtarget().hasMAIInsts() &&
          !TII.getSubtarget().hasGFX90AInsts()) &&
         "Expected GFX908 subtarget.");

  assert((AMDGPU::SReg_32RegClass.contains(SrcReg) ||
          AMDGPU::AGPR_32RegClass.contains(SrcReg)) &&
         "Source register of the copy should be either an SGPR or an AGPR.");

  assert(AMDGPU::AGPR_32RegClass.contains(DestReg) &&
         "Destination register of the copy should be an AGPR.");

  const SIRegisterInfo &RI = TII.getRegisterInfo();

  // First try to find defining accvgpr_write to avoid temporary registers.
  // In the case of copies of overlapping AGPRs, we conservatively do not
  // reuse previous accvgpr_writes. Otherwise, we may incorrectly pick up
  // an accvgpr_write used for this same copy due to implicit-defs
  if (!RegsOverlap) {
    for (auto Def = MI, E = MBB.begin(); Def != E; ) {
      --Def;

      if (!Def->modifiesRegister(SrcReg, &RI))
        continue;

      if (Def->getOpcode() != AMDGPU::V_ACCVGPR_WRITE_B32_e64 ||
          Def->getOperand(0).getReg() != SrcReg)
        break;

      MachineOperand &DefOp = Def->getOperand(1);
      assert(DefOp.isReg() || DefOp.isImm());

      if (DefOp.isReg()) {
        bool SafeToPropagate = true;
        // Check that register source operand is not clobbered before MI.
        // Immediate operands are always safe to propagate.
        for (auto I = Def; I != MI && SafeToPropagate; ++I)
          if (I->modifiesRegister(DefOp.getReg(), &RI))
            SafeToPropagate = false;

        if (!SafeToPropagate)
          break;

        for (auto I = Def; I != MI; ++I)
          I->clearRegisterKills(DefOp.getReg(), &RI);
      }

      MachineInstrBuilder Builder =
        BuildMI(MBB, MI, DL, TII.get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), DestReg)
        .add(DefOp);
      if (ImpDefSuperReg)
        Builder.addReg(ImpDefSuperReg, RegState::Define | RegState::Implicit);

      if (ImpUseSuperReg) {
        Builder.addReg(ImpUseSuperReg,
                      getKillRegState(KillSrc) | RegState::Implicit);
      }

      return;
    }
  }

  RS.enterBasicBlockEnd(MBB);
  RS.backward(std::next(MI));

  // Ideally we want to have three registers for a long reg_sequence copy
  // to hide 2 waitstates between v_mov_b32 and accvgpr_write.
  unsigned MaxVGPRs = RI.getRegPressureLimit(&AMDGPU::VGPR_32RegClass,
                                             *MBB.getParent());

  // Registers in the sequence are allocated contiguously so we can just
  // use register number to pick one of three round-robin temps.
  unsigned RegNo = (DestReg - AMDGPU::AGPR0) % 3;
  Register Tmp =
      MBB.getParent()->getInfo<SIMachineFunctionInfo>()->getVGPRForAGPRCopy();
  assert(MBB.getParent()->getRegInfo().isReserved(Tmp) &&
         "VGPR used for an intermediate copy should have been reserved.");

  // Only loop through if there are any free registers left. We don't want to
  // spill.
  while (RegNo--) {
    Register Tmp2 = RS.scavengeRegisterBackwards(AMDGPU::VGPR_32RegClass, MI,
                                                 /* RestoreAfter */ false, 0,
                                                 /* AllowSpill */ false);
    if (!Tmp2 || RI.getHWRegIndex(Tmp2) >= MaxVGPRs)
      break;
    Tmp = Tmp2;
    RS.setRegUsed(Tmp);
  }

  // Insert copy to temporary VGPR.
  unsigned TmpCopyOp = AMDGPU::V_MOV_B32_e32;
  if (AMDGPU::AGPR_32RegClass.contains(SrcReg)) {
    TmpCopyOp = AMDGPU::V_ACCVGPR_READ_B32_e64;
  } else {
    assert(AMDGPU::SReg_32RegClass.contains(SrcReg));
  }

  MachineInstrBuilder UseBuilder = BuildMI(MBB, MI, DL, TII.get(TmpCopyOp), Tmp)
    .addReg(SrcReg, getKillRegState(KillSrc));
  if (ImpUseSuperReg) {
    UseBuilder.addReg(ImpUseSuperReg,
                      getKillRegState(KillSrc) | RegState::Implicit);
  }

  MachineInstrBuilder DefBuilder
    = BuildMI(MBB, MI, DL, TII.get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), DestReg)
    .addReg(Tmp, RegState::Kill);

  if (ImpDefSuperReg)
    DefBuilder.addReg(ImpDefSuperReg, RegState::Define | RegState::Implicit);
}

static void expandSGPRCopy(const SIInstrInfo &TII, MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, const DebugLoc &DL,
                           MCRegister DestReg, MCRegister SrcReg, bool KillSrc,
                           const TargetRegisterClass *RC, bool Forward) {
  const SIRegisterInfo &RI = TII.getRegisterInfo();
  ArrayRef<int16_t> BaseIndices = RI.getRegSplitParts(RC, 4);
  MachineBasicBlock::iterator I = MI;
  MachineInstr *FirstMI = nullptr, *LastMI = nullptr;

  for (unsigned Idx = 0; Idx < BaseIndices.size(); ++Idx) {
    int16_t SubIdx = BaseIndices[Idx];
    Register DestSubReg = RI.getSubReg(DestReg, SubIdx);
    Register SrcSubReg = RI.getSubReg(SrcReg, SubIdx);
    assert(DestSubReg && SrcSubReg && "Failed to find subregs!");
    unsigned Opcode = AMDGPU::S_MOV_B32;

    // Is SGPR aligned? If so try to combine with next.
    bool AlignedDest = ((DestSubReg - AMDGPU::SGPR0) % 2) == 0;
    bool AlignedSrc = ((SrcSubReg - AMDGPU::SGPR0) % 2) == 0;
    if (AlignedDest && AlignedSrc && (Idx + 1 < BaseIndices.size())) {
      // Can use SGPR64 copy
      unsigned Channel = RI.getChannelFromSubReg(SubIdx);
      SubIdx = RI.getSubRegFromChannel(Channel, 2);
      DestSubReg = RI.getSubReg(DestReg, SubIdx);
      SrcSubReg = RI.getSubReg(SrcReg, SubIdx);
      assert(DestSubReg && SrcSubReg && "Failed to find subregs!");
      Opcode = AMDGPU::S_MOV_B64;
      Idx++;
    }

    LastMI = BuildMI(MBB, I, DL, TII.get(Opcode), DestSubReg)
                 .addReg(SrcSubReg)
                 .addReg(SrcReg, RegState::Implicit);

    if (!FirstMI)
      FirstMI = LastMI;

    if (!Forward)
      I--;
  }

  assert(FirstMI && LastMI);
  if (!Forward)
    std::swap(FirstMI, LastMI);

  FirstMI->addOperand(
      MachineOperand::CreateReg(DestReg, true /*IsDef*/, true /*IsImp*/));

  if (KillSrc)
    LastMI->addRegisterKilled(SrcReg, &RI);
}

void SIInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              const DebugLoc &DL, Register DestReg,
                              Register SrcReg, bool KillSrc, bool RenamableDest,
                              bool RenamableSrc) const {
  const TargetRegisterClass *RC = RI.getPhysRegBaseClass(DestReg);
  unsigned Size = RI.getRegSizeInBits(*RC);
  const TargetRegisterClass *SrcRC = RI.getPhysRegBaseClass(SrcReg);
  unsigned SrcSize = RI.getRegSizeInBits(*SrcRC);

  // The rest of copyPhysReg assumes Src and Dst size are the same size.
  // TODO-GFX11_16BIT If all true 16 bit instruction patterns are completed can
  // we remove Fix16BitCopies and this code block?
  if (Fix16BitCopies) {
    if (((Size == 16) != (SrcSize == 16))) {
      // Non-VGPR Src and Dst will later be expanded back to 32 bits.
      assert(ST.useRealTrue16Insts());
      Register &RegToFix = (Size == 32) ? DestReg : SrcReg;
      MCRegister SubReg = RI.getSubReg(RegToFix, AMDGPU::lo16);
      RegToFix = SubReg;

      if (DestReg == SrcReg) {
        // Identity copy. Insert empty bundle since ExpandPostRA expects an
        // instruction here.
        BuildMI(MBB, MI, DL, get(AMDGPU::BUNDLE));
        return;
      }
      RC = RI.getPhysRegBaseClass(DestReg);
      Size = RI.getRegSizeInBits(*RC);
      SrcRC = RI.getPhysRegBaseClass(SrcReg);
      SrcSize = RI.getRegSizeInBits(*SrcRC);
    }
  }

  if (RC == &AMDGPU::VGPR_32RegClass) {
    assert(AMDGPU::VGPR_32RegClass.contains(SrcReg) ||
           AMDGPU::SReg_32RegClass.contains(SrcReg) ||
           AMDGPU::AGPR_32RegClass.contains(SrcReg));
    unsigned Opc = AMDGPU::AGPR_32RegClass.contains(SrcReg) ?
                     AMDGPU::V_ACCVGPR_READ_B32_e64 : AMDGPU::V_MOV_B32_e32;
    BuildMI(MBB, MI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RC == &AMDGPU::SReg_32_XM0RegClass ||
      RC == &AMDGPU::SReg_32RegClass) {
    if (SrcReg == AMDGPU::SCC) {
      BuildMI(MBB, MI, DL, get(AMDGPU::S_CSELECT_B32), DestReg)
          .addImm(1)
          .addImm(0);
      return;
    }

    if (!AMDGPU::SReg_32RegClass.contains(SrcReg)) {
      if (DestReg == AMDGPU::VCC_LO) {
        // FIXME: Hack until VReg_1 removed.
        assert(AMDGPU::VGPR_32RegClass.contains(SrcReg));
        BuildMI(MBB, MI, DL, get(AMDGPU::V_CMP_NE_U32_e32))
            .addImm(0)
            .addReg(SrcReg, getKillRegState(KillSrc));
        return;
      }

      reportIllegalCopy(this, MBB, MI, DL, DestReg, SrcReg, KillSrc);
      return;
    }

    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B32), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RC == &AMDGPU::SReg_64RegClass) {
    if (SrcReg == AMDGPU::SCC) {
      BuildMI(MBB, MI, DL, get(AMDGPU::S_CSELECT_B64), DestReg)
          .addImm(1)
          .addImm(0);
      return;
    }

    if (!AMDGPU::SReg_64_EncodableRegClass.contains(SrcReg)) {
      if (DestReg == AMDGPU::VCC) {
        // FIXME: Hack until VReg_1 removed.
        assert(AMDGPU::VGPR_32RegClass.contains(SrcReg));
        BuildMI(MBB, MI, DL, get(AMDGPU::V_CMP_NE_U32_e32))
            .addImm(0)
            .addReg(SrcReg, getKillRegState(KillSrc));
        return;
      }

      reportIllegalCopy(this, MBB, MI, DL, DestReg, SrcReg, KillSrc);
      return;
    }

    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B64), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (DestReg == AMDGPU::SCC) {
    // Copying 64-bit or 32-bit sources to SCC barely makes sense,
    // but SelectionDAG emits such copies for i1 sources.
    if (AMDGPU::SReg_64RegClass.contains(SrcReg)) {
      // This copy can only be produced by patterns
      // with explicit SCC, which are known to be enabled
      // only for subtargets with S_CMP_LG_U64 present.
      assert(ST.hasScalarCompareEq64());
      BuildMI(MBB, MI, DL, get(AMDGPU::S_CMP_LG_U64))
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addImm(0);
    } else {
      assert(AMDGPU::SReg_32RegClass.contains(SrcReg));
      BuildMI(MBB, MI, DL, get(AMDGPU::S_CMP_LG_U32))
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addImm(0);
    }

    return;
  }

  if (RC == &AMDGPU::AGPR_32RegClass) {
    if (AMDGPU::VGPR_32RegClass.contains(SrcReg) ||
        (ST.hasGFX90AInsts() && AMDGPU::SReg_32RegClass.contains(SrcReg))) {
      BuildMI(MBB, MI, DL, get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
      return;
    }

    if (AMDGPU::AGPR_32RegClass.contains(SrcReg) && ST.hasGFX90AInsts()) {
      BuildMI(MBB, MI, DL, get(AMDGPU::V_ACCVGPR_MOV_B32), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
      return;
    }

    // FIXME: Pass should maintain scavenger to avoid scan through the block on
    // every AGPR spill.
    RegScavenger RS;
    const bool Overlap = RI.regsOverlap(SrcReg, DestReg);
    indirectCopyToAGPR(*this, MBB, MI, DL, DestReg, SrcReg, KillSrc, RS, Overlap);
    return;
  }

  if (Size == 16) {
    assert(AMDGPU::VGPR_16RegClass.contains(SrcReg) ||
           AMDGPU::SReg_LO16RegClass.contains(SrcReg) ||
           AMDGPU::AGPR_LO16RegClass.contains(SrcReg));

    bool IsSGPRDst = AMDGPU::SReg_LO16RegClass.contains(DestReg);
    bool IsSGPRSrc = AMDGPU::SReg_LO16RegClass.contains(SrcReg);
    bool IsAGPRDst = AMDGPU::AGPR_LO16RegClass.contains(DestReg);
    bool IsAGPRSrc = AMDGPU::AGPR_LO16RegClass.contains(SrcReg);
    bool DstLow = !AMDGPU::isHi16Reg(DestReg, RI);
    bool SrcLow = !AMDGPU::isHi16Reg(SrcReg, RI);
    MCRegister NewDestReg = RI.get32BitRegister(DestReg);
    MCRegister NewSrcReg = RI.get32BitRegister(SrcReg);

    if (IsSGPRDst) {
      if (!IsSGPRSrc) {
        reportIllegalCopy(this, MBB, MI, DL, DestReg, SrcReg, KillSrc);
        return;
      }

      BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B32), NewDestReg)
        .addReg(NewSrcReg, getKillRegState(KillSrc));
      return;
    }

    if (IsAGPRDst || IsAGPRSrc) {
      if (!DstLow || !SrcLow) {
        reportIllegalCopy(this, MBB, MI, DL, DestReg, SrcReg, KillSrc,
                          "Cannot use hi16 subreg with an AGPR!");
      }

      copyPhysReg(MBB, MI, DL, NewDestReg, NewSrcReg, KillSrc);
      return;
    }

    if (ST.useRealTrue16Insts()) {
      if (IsSGPRSrc) {
        assert(SrcLow);
        SrcReg = NewSrcReg;
      }
      // Use the smaller instruction encoding if possible.
      if (AMDGPU::VGPR_16_Lo128RegClass.contains(DestReg) &&
          (IsSGPRSrc || AMDGPU::VGPR_16_Lo128RegClass.contains(SrcReg))) {
        BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B16_t16_e32), DestReg)
            .addReg(SrcReg);
      } else {
        BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B16_t16_e64), DestReg)
            .addImm(0) // src0_modifiers
            .addReg(SrcReg)
            .addImm(0); // op_sel
      }
      return;
    }

    if (IsSGPRSrc && !ST.hasSDWAScalar()) {
      if (!DstLow || !SrcLow) {
        reportIllegalCopy(this, MBB, MI, DL, DestReg, SrcReg, KillSrc,
                          "Cannot use hi16 subreg on VI!");
      }

      BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), NewDestReg)
        .addReg(NewSrcReg, getKillRegState(KillSrc));
      return;
    }

    auto MIB = BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_sdwa), NewDestReg)
      .addImm(0) // src0_modifiers
      .addReg(NewSrcReg)
      .addImm(0) // clamp
      .addImm(DstLow ? AMDGPU::SDWA::SdwaSel::WORD_0
                     : AMDGPU::SDWA::SdwaSel::WORD_1)
      .addImm(AMDGPU::SDWA::DstUnused::UNUSED_PRESERVE)
      .addImm(SrcLow ? AMDGPU::SDWA::SdwaSel::WORD_0
                     : AMDGPU::SDWA::SdwaSel::WORD_1)
      .addReg(NewDestReg, RegState::Implicit | RegState::Undef);
    // First implicit operand is $exec.
    MIB->tieOperands(0, MIB->getNumOperands() - 1);
    return;
  }

  if (RC == RI.getVGPR64Class() && (SrcRC == RC || RI.isSGPRClass(SrcRC))) {
    if (ST.hasMovB64()) {
      BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B64_e32), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
      return;
    }
    if (ST.hasPkMovB32()) {
      BuildMI(MBB, MI, DL, get(AMDGPU::V_PK_MOV_B32), DestReg)
        .addImm(SISrcMods::OP_SEL_1)
        .addReg(SrcReg)
        .addImm(SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1)
        .addReg(SrcReg)
        .addImm(0) // op_sel_lo
        .addImm(0) // op_sel_hi
        .addImm(0) // neg_lo
        .addImm(0) // neg_hi
        .addImm(0) // clamp
        .addReg(SrcReg, getKillRegState(KillSrc) | RegState::Implicit);
      return;
    }
  }

  const bool Forward = RI.getHWRegIndex(DestReg) <= RI.getHWRegIndex(SrcReg);
  if (RI.isSGPRClass(RC)) {
    if (!RI.isSGPRClass(SrcRC)) {
      reportIllegalCopy(this, MBB, MI, DL, DestReg, SrcReg, KillSrc);
      return;
    }
    const bool CanKillSuperReg = KillSrc && !RI.regsOverlap(SrcReg, DestReg);
    expandSGPRCopy(*this, MBB, MI, DL, DestReg, SrcReg, CanKillSuperReg, RC,
                   Forward);
    return;
  }

  unsigned EltSize = 4;
  unsigned Opcode = AMDGPU::V_MOV_B32_e32;
  if (RI.isAGPRClass(RC)) {
    if (ST.hasGFX90AInsts() && RI.isAGPRClass(SrcRC))
      Opcode = AMDGPU::V_ACCVGPR_MOV_B32;
    else if (RI.hasVGPRs(SrcRC) ||
             (ST.hasGFX90AInsts() && RI.isSGPRClass(SrcRC)))
      Opcode = AMDGPU::V_ACCVGPR_WRITE_B32_e64;
    else
      Opcode = AMDGPU::INSTRUCTION_LIST_END;
  } else if (RI.hasVGPRs(RC) && RI.isAGPRClass(SrcRC)) {
    Opcode = AMDGPU::V_ACCVGPR_READ_B32_e64;
  } else if ((Size % 64 == 0) && RI.hasVGPRs(RC) &&
             (RI.isProperlyAlignedRC(*RC) &&
              (SrcRC == RC || RI.isSGPRClass(SrcRC)))) {
    // TODO: In 96-bit case, could do a 64-bit mov and then a 32-bit mov.
    if (ST.hasMovB64()) {
      Opcode = AMDGPU::V_MOV_B64_e32;
      EltSize = 8;
    } else if (ST.hasPkMovB32()) {
      Opcode = AMDGPU::V_PK_MOV_B32;
      EltSize = 8;
    }
  }

  // For the cases where we need an intermediate instruction/temporary register
  // (destination is an AGPR), we need a scavenger.
  //
  // FIXME: The pass should maintain this for us so we don't have to re-scan the
  // whole block for every handled copy.
  std::unique_ptr<RegScavenger> RS;
  if (Opcode == AMDGPU::INSTRUCTION_LIST_END)
    RS = std::make_unique<RegScavenger>();

  ArrayRef<int16_t> SubIndices = RI.getRegSplitParts(RC, EltSize);

  // If there is an overlap, we can't kill the super-register on the last
  // instruction, since it will also kill the components made live by this def.
  const bool Overlap = RI.regsOverlap(SrcReg, DestReg);
  const bool CanKillSuperReg = KillSrc && !Overlap;

  for (unsigned Idx = 0; Idx < SubIndices.size(); ++Idx) {
    unsigned SubIdx;
    if (Forward)
      SubIdx = SubIndices[Idx];
    else
      SubIdx = SubIndices[SubIndices.size() - Idx - 1];
    Register DestSubReg = RI.getSubReg(DestReg, SubIdx);
    Register SrcSubReg = RI.getSubReg(SrcReg, SubIdx);
    assert(DestSubReg && SrcSubReg && "Failed to find subregs!");

    bool IsFirstSubreg = Idx == 0;
    bool UseKill = CanKillSuperReg && Idx == SubIndices.size() - 1;

    if (Opcode == AMDGPU::INSTRUCTION_LIST_END) {
      Register ImpDefSuper = IsFirstSubreg ? Register(DestReg) : Register();
      Register ImpUseSuper = SrcReg;
      indirectCopyToAGPR(*this, MBB, MI, DL, DestSubReg, SrcSubReg, UseKill,
                         *RS, Overlap, ImpDefSuper, ImpUseSuper);
    } else if (Opcode == AMDGPU::V_PK_MOV_B32) {
      MachineInstrBuilder MIB =
          BuildMI(MBB, MI, DL, get(AMDGPU::V_PK_MOV_B32), DestSubReg)
              .addImm(SISrcMods::OP_SEL_1)
              .addReg(SrcSubReg)
              .addImm(SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1)
              .addReg(SrcSubReg)
              .addImm(0) // op_sel_lo
              .addImm(0) // op_sel_hi
              .addImm(0) // neg_lo
              .addImm(0) // neg_hi
              .addImm(0) // clamp
              .addReg(SrcReg, getKillRegState(UseKill) | RegState::Implicit);
      if (IsFirstSubreg)
        MIB.addReg(DestReg, RegState::Define | RegState::Implicit);
    } else {
      MachineInstrBuilder Builder =
          BuildMI(MBB, MI, DL, get(Opcode), DestSubReg).addReg(SrcSubReg);
      if (IsFirstSubreg)
        Builder.addReg(DestReg, RegState::Define | RegState::Implicit);

      Builder.addReg(SrcReg, getKillRegState(UseKill) | RegState::Implicit);
    }
  }
}

int SIInstrInfo::commuteOpcode(unsigned Opcode) const {
  int32_t NewOpc;

  // Try to map original to commuted opcode
  NewOpc = AMDGPU::getCommuteRev(Opcode);
  if (NewOpc != -1)
    // Check if the commuted (REV) opcode exists on the target.
    return pseudoToMCOpcode(NewOpc) != -1 ? NewOpc : -1;

  // Try to map commuted to original opcode
  NewOpc = AMDGPU::getCommuteOrig(Opcode);
  if (NewOpc != -1)
    // Check if the original (non-REV) opcode exists on the target.
    return pseudoToMCOpcode(NewOpc) != -1 ? NewOpc : -1;

  return Opcode;
}

const TargetRegisterClass *
SIInstrInfo::getPreferredSelectRegClass(unsigned Size) const {
  return &AMDGPU::VGPR_32RegClass;
}

void SIInstrInfo::insertVectorSelect(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I,
                                     const DebugLoc &DL, Register DstReg,
                                     ArrayRef<MachineOperand> Cond,
                                     Register TrueReg,
                                     Register FalseReg) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *BoolXExecRC = RI.getWaveMaskRegClass();
  const AMDGPU::LaneMaskConstants &LMC = AMDGPU::LaneMaskConstants::get(ST);
  assert(MRI.getRegClass(DstReg) == &AMDGPU::VGPR_32RegClass &&
         "Not a VGPR32 reg");

  if (Cond.size() == 1) {
    Register SReg = MRI.createVirtualRegister(BoolXExecRC);
    BuildMI(MBB, I, DL, get(AMDGPU::COPY), SReg)
      .add(Cond[0]);
    BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
      .addImm(0)
      .addReg(FalseReg)
      .addImm(0)
      .addReg(TrueReg)
      .addReg(SReg);
  } else if (Cond.size() == 2) {
    assert(Cond[0].isImm() && "Cond[0] is not an immediate");
    switch (Cond[0].getImm()) {
    case SIInstrInfo::SCC_TRUE: {
      Register SReg = MRI.createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, get(LMC.CSelectOpc), SReg).addImm(1).addImm(0);
      BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
        .addImm(0)
        .addReg(FalseReg)
        .addImm(0)
        .addReg(TrueReg)
        .addReg(SReg);
      break;
    }
    case SIInstrInfo::SCC_FALSE: {
      Register SReg = MRI.createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, get(LMC.CSelectOpc), SReg).addImm(0).addImm(1);
      BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
        .addImm(0)
        .addReg(FalseReg)
        .addImm(0)
        .addReg(TrueReg)
        .addReg(SReg);
      break;
    }
    case SIInstrInfo::VCCNZ: {
      MachineOperand RegOp = Cond[1];
      RegOp.setImplicit(false);
      Register SReg = MRI.createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, get(AMDGPU::COPY), SReg)
        .add(RegOp);
      BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
          .addImm(0)
          .addReg(FalseReg)
          .addImm(0)
          .addReg(TrueReg)
          .addReg(SReg);
      break;
    }
    case SIInstrInfo::VCCZ: {
      MachineOperand RegOp = Cond[1];
      RegOp.setImplicit(false);
      Register SReg = MRI.createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, get(AMDGPU::COPY), SReg)
        .add(RegOp);
      BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
          .addImm(0)
          .addReg(TrueReg)
          .addImm(0)
          .addReg(FalseReg)
          .addReg(SReg);
      break;
    }
    case SIInstrInfo::EXECNZ: {
      Register SReg = MRI.createVirtualRegister(BoolXExecRC);
      Register SReg2 = MRI.createVirtualRegister(RI.getBoolRC());
      BuildMI(MBB, I, DL, get(LMC.OrSaveExecOpc), SReg2).addImm(0);
      BuildMI(MBB, I, DL, get(LMC.CSelectOpc), SReg).addImm(1).addImm(0);
      BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
        .addImm(0)
        .addReg(FalseReg)
        .addImm(0)
        .addReg(TrueReg)
        .addReg(SReg);
      break;
    }
    case SIInstrInfo::EXECZ: {
      Register SReg = MRI.createVirtualRegister(BoolXExecRC);
      Register SReg2 = MRI.createVirtualRegister(RI.getBoolRC());
      BuildMI(MBB, I, DL, get(LMC.OrSaveExecOpc), SReg2).addImm(0);
      BuildMI(MBB, I, DL, get(LMC.CSelectOpc), SReg).addImm(0).addImm(1);
      BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
        .addImm(0)
        .addReg(FalseReg)
        .addImm(0)
        .addReg(TrueReg)
        .addReg(SReg);
      llvm_unreachable("Unhandled branch predicate EXECZ");
      break;
    }
    default:
      llvm_unreachable("invalid branch predicate");
    }
  } else {
    llvm_unreachable("Can only handle Cond size 1 or 2");
  }
}

Register SIInstrInfo::insertEQ(MachineBasicBlock *MBB,
                               MachineBasicBlock::iterator I,
                               const DebugLoc &DL,
                               Register SrcReg, int Value) const {
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  Register Reg = MRI.createVirtualRegister(RI.getBoolRC());
  BuildMI(*MBB, I, DL, get(AMDGPU::V_CMP_EQ_I32_e64), Reg)
    .addImm(Value)
    .addReg(SrcReg);

  return Reg;
}

Register SIInstrInfo::insertNE(MachineBasicBlock *MBB,
                               MachineBasicBlock::iterator I,
                               const DebugLoc &DL,
                               Register SrcReg, int Value) const {
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  Register Reg = MRI.createVirtualRegister(RI.getBoolRC());
  BuildMI(*MBB, I, DL, get(AMDGPU::V_CMP_NE_I32_e64), Reg)
    .addImm(Value)
    .addReg(SrcReg);

  return Reg;
}

bool SIInstrInfo::getConstValDefinedInReg(const MachineInstr &MI,
                                          const Register Reg,
                                          int64_t &ImmVal) const {
  switch (MI.getOpcode()) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOVK_I32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::V_MOV_B64_e32:
  case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
  case AMDGPU::AV_MOV_B32_IMM_PSEUDO:
  case AMDGPU::AV_MOV_B64_IMM_PSEUDO:
  case AMDGPU::S_MOV_B64_IMM_PSEUDO:
  case AMDGPU::V_MOV_B64_PSEUDO:
  case AMDGPU::V_MOV_B16_t16_e32: {
    const MachineOperand &Src0 = MI.getOperand(1);
    if (Src0.isImm()) {
      ImmVal = Src0.getImm();
      return MI.getOperand(0).getReg() == Reg;
    }

    return false;
  }
  case AMDGPU::V_MOV_B16_t16_e64: {
    const MachineOperand &Src0 = MI.getOperand(2);
    if (Src0.isImm() && !MI.getOperand(1).getImm()) {
      ImmVal = Src0.getImm();
      return MI.getOperand(0).getReg() == Reg;
    }

    return false;
  }
  case AMDGPU::S_BREV_B32:
  case AMDGPU::V_BFREV_B32_e32:
  case AMDGPU::V_BFREV_B32_e64: {
    const MachineOperand &Src0 = MI.getOperand(1);
    if (Src0.isImm()) {
      ImmVal = static_cast<int64_t>(reverseBits<int32_t>(Src0.getImm()));
      return MI.getOperand(0).getReg() == Reg;
    }

    return false;
  }
  case AMDGPU::S_NOT_B32:
  case AMDGPU::V_NOT_B32_e32:
  case AMDGPU::V_NOT_B32_e64: {
    const MachineOperand &Src0 = MI.getOperand(1);
    if (Src0.isImm()) {
      ImmVal = static_cast<int64_t>(~static_cast<int32_t>(Src0.getImm()));
      return MI.getOperand(0).getReg() == Reg;
    }

    return false;
  }
  default:
    return false;
  }
}

std::optional<int64_t>
SIInstrInfo::getImmOrMaterializedImm(MachineOperand &Op) const {
  if (Op.isImm())
    return Op.getImm();

  if (!Op.isReg() || !Op.getReg().isVirtual())
    return std::nullopt;
  MachineRegisterInfo &MRI = Op.getParent()->getMF()->getRegInfo();
  const MachineInstr *Def = MRI.getVRegDef(Op.getReg());
  if (Def && Def->isMoveImmediate()) {
    const MachineOperand &ImmSrc = Def->getOperand(1);
    if (ImmSrc.isImm())
      return extractSubregFromImm(ImmSrc.getImm(), Op.getSubReg());
  }

  return std::nullopt;
}

unsigned SIInstrInfo::getMovOpcode(const TargetRegisterClass *DstRC) const {

  if (RI.isAGPRClass(DstRC))
    return AMDGPU::COPY;
  if (RI.getRegSizeInBits(*DstRC) == 16) {
    // Assume hi bits are unneeded. Only _e64 true16 instructions are legal
    // before RA.
    return RI.isSGPRClass(DstRC) ? AMDGPU::COPY : AMDGPU::V_MOV_B16_t16_e64;
  }
  if (RI.getRegSizeInBits(*DstRC) == 32)
    return RI.isSGPRClass(DstRC) ? AMDGPU::S_MOV_B32 : AMDGPU::V_MOV_B32_e32;
  if (RI.getRegSizeInBits(*DstRC) == 64 && RI.isSGPRClass(DstRC))
    return AMDGPU::S_MOV_B64;
  if (RI.getRegSizeInBits(*DstRC) == 64 && !RI.isSGPRClass(DstRC))
    return AMDGPU::V_MOV_B64_PSEUDO;
  return AMDGPU::COPY;
}

const MCInstrDesc &
SIInstrInfo::getIndirectGPRIDXPseudo(unsigned VecSize,
                                     bool IsIndirectSrc) const {
  if (IsIndirectSrc) {
    if (VecSize <= 32) // 4 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V1);
    if (VecSize <= 64) // 8 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V2);
    if (VecSize <= 96) // 12 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V3);
    if (VecSize <= 128) // 16 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V4);
    if (VecSize <= 160) // 20 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V5);
    if (VecSize <= 192) // 24 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V6);
    if (VecSize <= 224) // 28 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V7);
    if (VecSize <= 256) // 32 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V8);
    if (VecSize <= 288) // 36 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V9);
    if (VecSize <= 320) // 40 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V10);
    if (VecSize <= 352) // 44 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V11);
    if (VecSize <= 384) // 48 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V12);
    if (VecSize <= 512) // 64 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V16);
    if (VecSize <= 1024) // 128 bytes
      return get(AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V32);

    llvm_unreachable("unsupported size for IndirectRegReadGPRIDX pseudos");
  }

  if (VecSize <= 32) // 4 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V1);
  if (VecSize <= 64) // 8 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V2);
  if (VecSize <= 96) // 12 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V3);
  if (VecSize <= 128) // 16 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V4);
  if (VecSize <= 160) // 20 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V5);
  if (VecSize <= 192) // 24 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V6);
  if (VecSize <= 224) // 28 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V7);
  if (VecSize <= 256) // 32 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V8);
  if (VecSize <= 288) // 36 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V9);
  if (VecSize <= 320) // 40 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V10);
  if (VecSize <= 352) // 44 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V11);
  if (VecSize <= 384) // 48 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V12);
  if (VecSize <= 512) // 64 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V16);
  if (VecSize <= 1024) // 128 bytes
    return get(AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V32);

  llvm_unreachable("unsupported size for IndirectRegWriteGPRIDX pseudos");
}

static unsigned getIndirectVGPRWriteMovRelPseudoOpc(unsigned VecSize) {
  if (VecSize <= 32) // 4 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V1;
  if (VecSize <= 64) // 8 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V2;
  if (VecSize <= 96) // 12 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V3;
  if (VecSize <= 128) // 16 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V4;
  if (VecSize <= 160) // 20 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V5;
  if (VecSize <= 192) // 24 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V6;
  if (VecSize <= 224) // 28 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V7;
  if (VecSize <= 256) // 32 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V8;
  if (VecSize <= 288) // 36 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V9;
  if (VecSize <= 320) // 40 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V10;
  if (VecSize <= 352) // 44 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V11;
  if (VecSize <= 384) // 48 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V12;
  if (VecSize <= 512) // 64 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V16;
  if (VecSize <= 1024) // 128 bytes
    return AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V32;

  llvm_unreachable("unsupported size for IndirectRegWrite pseudos");
}

static unsigned getIndirectSGPRWriteMovRelPseudo32(unsigned VecSize) {
  if (VecSize <= 32) // 4 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V1;
  if (VecSize <= 64) // 8 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V2;
  if (VecSize <= 96) // 12 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V3;
  if (VecSize <= 128) // 16 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V4;
  if (VecSize <= 160) // 20 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V5;
  if (VecSize <= 192) // 24 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V6;
  if (VecSize <= 224) // 28 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V7;
  if (VecSize <= 256) // 32 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V8;
  if (VecSize <= 288) // 36 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V9;
  if (VecSize <= 320) // 40 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V10;
  if (VecSize <= 352) // 44 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V11;
  if (VecSize <= 384) // 48 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V12;
  if (VecSize <= 512) // 64 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V16;
  if (VecSize <= 1024) // 128 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V32;

  llvm_unreachable("unsupported size for IndirectRegWrite pseudos");
}

static unsigned getIndirectSGPRWriteMovRelPseudo64(unsigned VecSize) {
  if (VecSize <= 64) // 8 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V1;
  if (VecSize <= 128) // 16 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V2;
  if (VecSize <= 256) // 32 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V4;
  if (VecSize <= 512) // 64 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V8;
  if (VecSize <= 1024) // 128 bytes
    return AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V16;

  llvm_unreachable("unsupported size for IndirectRegWrite pseudos");
}

const MCInstrDesc &
SIInstrInfo::getIndirectRegWriteMovRelPseudo(unsigned VecSize, unsigned EltSize,
                                             bool IsSGPR) const {
  if (IsSGPR) {
    switch (EltSize) {
    case 32:
      return get(getIndirectSGPRWriteMovRelPseudo32(VecSize));
    case 64:
      return get(getIndirectSGPRWriteMovRelPseudo64(VecSize));
    default:
      llvm_unreachable("invalid reg indexing elt size");
    }
  }

  assert(EltSize == 32 && "invalid reg indexing elt size");
  return get(getIndirectVGPRWriteMovRelPseudoOpc(VecSize));
}

static unsigned getSGPRSpillSaveOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_S32_SAVE;
  case 8:
    return AMDGPU::SI_SPILL_S64_SAVE;
  case 12:
    return AMDGPU::SI_SPILL_S96_SAVE;
  case 16:
    return AMDGPU::SI_SPILL_S128_SAVE;
  case 20:
    return AMDGPU::SI_SPILL_S160_SAVE;
  case 24:
    return AMDGPU::SI_SPILL_S192_SAVE;
  case 28:
    return AMDGPU::SI_SPILL_S224_SAVE;
  case 32:
    return AMDGPU::SI_SPILL_S256_SAVE;
  case 36:
    return AMDGPU::SI_SPILL_S288_SAVE;
  case 40:
    return AMDGPU::SI_SPILL_S320_SAVE;
  case 44:
    return AMDGPU::SI_SPILL_S352_SAVE;
  case 48:
    return AMDGPU::SI_SPILL_S384_SAVE;
  case 64:
    return AMDGPU::SI_SPILL_S512_SAVE;
  case 128:
    return AMDGPU::SI_SPILL_S1024_SAVE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getVGPRSpillSaveOpcode(unsigned Size) {
  switch (Size) {
  case 2:
    return AMDGPU::SI_SPILL_V16_SAVE;
  case 4:
    return AMDGPU::SI_SPILL_V32_SAVE;
  case 8:
    return AMDGPU::SI_SPILL_V64_SAVE;
  case 12:
    return AMDGPU::SI_SPILL_V96_SAVE;
  case 16:
    return AMDGPU::SI_SPILL_V128_SAVE;
  case 20:
    return AMDGPU::SI_SPILL_V160_SAVE;
  case 24:
    return AMDGPU::SI_SPILL_V192_SAVE;
  case 28:
    return AMDGPU::SI_SPILL_V224_SAVE;
  case 32:
    return AMDGPU::SI_SPILL_V256_SAVE;
  case 36:
    return AMDGPU::SI_SPILL_V288_SAVE;
  case 40:
    return AMDGPU::SI_SPILL_V320_SAVE;
  case 44:
    return AMDGPU::SI_SPILL_V352_SAVE;
  case 48:
    return AMDGPU::SI_SPILL_V384_SAVE;
  case 64:
    return AMDGPU::SI_SPILL_V512_SAVE;
  case 128:
    return AMDGPU::SI_SPILL_V1024_SAVE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getAVSpillSaveOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_AV32_SAVE;
  case 8:
    return AMDGPU::SI_SPILL_AV64_SAVE;
  case 12:
    return AMDGPU::SI_SPILL_AV96_SAVE;
  case 16:
    return AMDGPU::SI_SPILL_AV128_SAVE;
  case 20:
    return AMDGPU::SI_SPILL_AV160_SAVE;
  case 24:
    return AMDGPU::SI_SPILL_AV192_SAVE;
  case 28:
    return AMDGPU::SI_SPILL_AV224_SAVE;
  case 32:
    return AMDGPU::SI_SPILL_AV256_SAVE;
  case 36:
    return AMDGPU::SI_SPILL_AV288_SAVE;
  case 40:
    return AMDGPU::SI_SPILL_AV320_SAVE;
  case 44:
    return AMDGPU::SI_SPILL_AV352_SAVE;
  case 48:
    return AMDGPU::SI_SPILL_AV384_SAVE;
  case 64:
    return AMDGPU::SI_SPILL_AV512_SAVE;
  case 128:
    return AMDGPU::SI_SPILL_AV1024_SAVE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getWWMRegSpillSaveOpcode(unsigned Size,
                                         bool IsVectorSuperClass) {
  // Currently, there is only 32-bit WWM register spills needed.
  if (Size != 4)
    llvm_unreachable("unknown wwm register spill size");

  if (IsVectorSuperClass)
    return AMDGPU::SI_SPILL_WWM_AV32_SAVE;

  return AMDGPU::SI_SPILL_WWM_V32_SAVE;
}

unsigned SIInstrInfo::getVectorRegSpillSaveOpcode(
    Register Reg, const TargetRegisterClass *RC, unsigned Size,
    const SIMachineFunctionInfo &MFI) const {
  bool IsVectorSuperClass = RI.isVectorSuperClass(RC);

  // Choose the right opcode if spilling a WWM register.
  if (MFI.checkFlag(Reg, AMDGPU::VirtRegFlag::WWM_REG))
    return getWWMRegSpillSaveOpcode(Size, IsVectorSuperClass);

  // TODO: Check if AGPRs are available
  if (ST.hasMAIInsts())
    return getAVSpillSaveOpcode(Size);

  return getVGPRSpillSaveOpcode(Size);
}

void SIInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
    bool isKill, int FrameIndex, const TargetRegisterClass *RC, Register VReg,
    MachineInstr::MIFlag Flags) const {
  MachineFunction *MF = MBB.getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  const DebugLoc &DL = MBB.findDebugLoc(MI);

  MachinePointerInfo PtrInfo
    = MachinePointerInfo::getFixedStack(*MF, FrameIndex);
  MachineMemOperand *MMO = MF->getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOStore, FrameInfo.getObjectSize(FrameIndex),
      FrameInfo.getObjectAlign(FrameIndex));
  unsigned SpillSize = RI.getSpillSize(*RC);

  MachineRegisterInfo &MRI = MF->getRegInfo();
  if (RI.isSGPRClass(RC)) {
    MFI->setHasSpilledSGPRs();
    assert(SrcReg != AMDGPU::M0 && "m0 should not be spilled");
    assert(SrcReg != AMDGPU::EXEC_LO && SrcReg != AMDGPU::EXEC_HI &&
           SrcReg != AMDGPU::EXEC && "exec should not be spilled");

    // We are only allowed to create one new instruction when spilling
    // registers, so we need to use pseudo instruction for spilling SGPRs.
    const MCInstrDesc &OpDesc = get(getSGPRSpillSaveOpcode(SpillSize));

    // The SGPR spill/restore instructions only work on number sgprs, so we need
    // to make sure we are using the correct register class.
    if (SrcReg.isVirtual() && SpillSize == 4) {
      MRI.constrainRegClass(SrcReg, &AMDGPU::SReg_32_XM0_XEXECRegClass);
    }

    BuildMI(MBB, MI, DL, OpDesc)
      .addReg(SrcReg, getKillRegState(isKill)) // data
      .addFrameIndex(FrameIndex)               // addr
      .addMemOperand(MMO)
      .addReg(MFI->getStackPtrOffsetReg(), RegState::Implicit);

    if (RI.spillSGPRToVGPR())
      FrameInfo.setStackID(FrameIndex, TargetStackID::SGPRSpill);
    return;
  }

  unsigned Opcode =
      getVectorRegSpillSaveOpcode(VReg ? VReg : SrcReg, RC, SpillSize, *MFI);
  MFI->setHasSpilledVGPRs();

  BuildMI(MBB, MI, DL, get(Opcode))
    .addReg(SrcReg, getKillRegState(isKill)) // data
    .addFrameIndex(FrameIndex)               // addr
    .addReg(MFI->getStackPtrOffsetReg())     // scratch_offset
    .addImm(0)                               // offset
    .addMemOperand(MMO);
}

static unsigned getSGPRSpillRestoreOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_S32_RESTORE;
  case 8:
    return AMDGPU::SI_SPILL_S64_RESTORE;
  case 12:
    return AMDGPU::SI_SPILL_S96_RESTORE;
  case 16:
    return AMDGPU::SI_SPILL_S128_RESTORE;
  case 20:
    return AMDGPU::SI_SPILL_S160_RESTORE;
  case 24:
    return AMDGPU::SI_SPILL_S192_RESTORE;
  case 28:
    return AMDGPU::SI_SPILL_S224_RESTORE;
  case 32:
    return AMDGPU::SI_SPILL_S256_RESTORE;
  case 36:
    return AMDGPU::SI_SPILL_S288_RESTORE;
  case 40:
    return AMDGPU::SI_SPILL_S320_RESTORE;
  case 44:
    return AMDGPU::SI_SPILL_S352_RESTORE;
  case 48:
    return AMDGPU::SI_SPILL_S384_RESTORE;
  case 64:
    return AMDGPU::SI_SPILL_S512_RESTORE;
  case 128:
    return AMDGPU::SI_SPILL_S1024_RESTORE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getVGPRSpillRestoreOpcode(unsigned Size) {
  switch (Size) {
  case 2:
    return AMDGPU::SI_SPILL_V16_RESTORE;
  case 4:
    return AMDGPU::SI_SPILL_V32_RESTORE;
  case 8:
    return AMDGPU::SI_SPILL_V64_RESTORE;
  case 12:
    return AMDGPU::SI_SPILL_V96_RESTORE;
  case 16:
    return AMDGPU::SI_SPILL_V128_RESTORE;
  case 20:
    return AMDGPU::SI_SPILL_V160_RESTORE;
  case 24:
    return AMDGPU::SI_SPILL_V192_RESTORE;
  case 28:
    return AMDGPU::SI_SPILL_V224_RESTORE;
  case 32:
    return AMDGPU::SI_SPILL_V256_RESTORE;
  case 36:
    return AMDGPU::SI_SPILL_V288_RESTORE;
  case 40:
    return AMDGPU::SI_SPILL_V320_RESTORE;
  case 44:
    return AMDGPU::SI_SPILL_V352_RESTORE;
  case 48:
    return AMDGPU::SI_SPILL_V384_RESTORE;
  case 64:
    return AMDGPU::SI_SPILL_V512_RESTORE;
  case 128:
    return AMDGPU::SI_SPILL_V1024_RESTORE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getAVSpillRestoreOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_AV32_RESTORE;
  case 8:
    return AMDGPU::SI_SPILL_AV64_RESTORE;
  case 12:
    return AMDGPU::SI_SPILL_AV96_RESTORE;
  case 16:
    return AMDGPU::SI_SPILL_AV128_RESTORE;
  case 20:
    return AMDGPU::SI_SPILL_AV160_RESTORE;
  case 24:
    return AMDGPU::SI_SPILL_AV192_RESTORE;
  case 28:
    return AMDGPU::SI_SPILL_AV224_RESTORE;
  case 32:
    return AMDGPU::SI_SPILL_AV256_RESTORE;
  case 36:
    return AMDGPU::SI_SPILL_AV288_RESTORE;
  case 40:
    return AMDGPU::SI_SPILL_AV320_RESTORE;
  case 44:
    return AMDGPU::SI_SPILL_AV352_RESTORE;
  case 48:
    return AMDGPU::SI_SPILL_AV384_RESTORE;
  case 64:
    return AMDGPU::SI_SPILL_AV512_RESTORE;
  case 128:
    return AMDGPU::SI_SPILL_AV1024_RESTORE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getWWMRegSpillRestoreOpcode(unsigned Size,
                                            bool IsVectorSuperClass) {
  // Currently, there is only 32-bit WWM register spills needed.
  if (Size != 4)
    llvm_unreachable("unknown wwm register spill size");

  if (IsVectorSuperClass) // TODO: Always use this if there are AGPRs
    return AMDGPU::SI_SPILL_WWM_AV32_RESTORE;

  return AMDGPU::SI_SPILL_WWM_V32_RESTORE;
}

unsigned SIInstrInfo::getVectorRegSpillRestoreOpcode(
    Register Reg, const TargetRegisterClass *RC, unsigned Size,
    const SIMachineFunctionInfo &MFI) const {
  bool IsVectorSuperClass = RI.isVectorSuperClass(RC);

  // Choose the right opcode if restoring a WWM register.
  if (MFI.checkFlag(Reg, AMDGPU::VirtRegFlag::WWM_REG))
    return getWWMRegSpillRestoreOpcode(Size, IsVectorSuperClass);

  // TODO: Check if AGPRs are available
  if (ST.hasMAIInsts())
    return getAVSpillRestoreOpcode(Size);

  assert(!RI.isAGPRClass(RC));
  return getVGPRSpillRestoreOpcode(Size);
}

void SIInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       Register DestReg, int FrameIndex,
                                       const TargetRegisterClass *RC,
                                       Register VReg, unsigned SubReg,
                                       MachineInstr::MIFlag Flags) const {
  MachineFunction *MF = MBB.getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  const DebugLoc &DL = MBB.findDebugLoc(MI);
  unsigned SpillSize = RI.getSpillSize(*RC);

  MachinePointerInfo PtrInfo
    = MachinePointerInfo::getFixedStack(*MF, FrameIndex);

  MachineMemOperand *MMO = MF->getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad, FrameInfo.getObjectSize(FrameIndex),
      FrameInfo.getObjectAlign(FrameIndex));

  if (RI.isSGPRClass(RC)) {
    MFI->setHasSpilledSGPRs();
    assert(DestReg != AMDGPU::M0 && "m0 should not be reloaded into");
    assert(DestReg != AMDGPU::EXEC_LO && DestReg != AMDGPU::EXEC_HI &&
           DestReg != AMDGPU::EXEC && "exec should not be spilled");

    // FIXME: Maybe this should not include a memoperand because it will be
    // lowered to non-memory instructions.
    const MCInstrDesc &OpDesc = get(getSGPRSpillRestoreOpcode(SpillSize));
    if (DestReg.isVirtual() && SpillSize == 4) {
      MachineRegisterInfo &MRI = MF->getRegInfo();
      MRI.constrainRegClass(DestReg, &AMDGPU::SReg_32_XM0_XEXECRegClass);
    }

    if (RI.spillSGPRToVGPR())
      FrameInfo.setStackID(FrameIndex, TargetStackID::SGPRSpill);
    BuildMI(MBB, MI, DL, OpDesc, DestReg)
      .addFrameIndex(FrameIndex) // addr
      .addMemOperand(MMO)
      .addReg(MFI->getStackPtrOffsetReg(), RegState::Implicit);

    return;
  }

  unsigned Opcode = getVectorRegSpillRestoreOpcode(VReg ? VReg : DestReg, RC,
                                                   SpillSize, *MFI);
  BuildMI(MBB, MI, DL, get(Opcode), DestReg)
      .addFrameIndex(FrameIndex)           // vaddr
      .addReg(MFI->getStackPtrOffsetReg()) // scratch_offset
      .addImm(0)                           // offset
      .addMemOperand(MMO);
}

void SIInstrInfo::insertNoop(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI) const {
  insertNoops(MBB, MI, 1);
}

void SIInstrInfo::insertNoops(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              unsigned Quantity) const {
  DebugLoc DL = MBB.findDebugLoc(MI);
  unsigned MaxSNopCount = 1u << ST.getSNopBits();
  while (Quantity > 0) {
    unsigned Arg = std::min(Quantity, MaxSNopCount);
    Quantity -= Arg;
    BuildMI(MBB, MI, DL, get(AMDGPU::S_NOP)).addImm(Arg - 1);
  }
}

void SIInstrInfo::insertReturn(MachineBasicBlock &MBB) const {
  auto *MF = MBB.getParent();
  SIMachineFunctionInfo *Info = MF->getInfo<SIMachineFunctionInfo>();

  assert(Info->isEntryFunction());

  if (MBB.succ_empty()) {
    bool HasNoTerminator = MBB.getFirstTerminator() == MBB.end();
    if (HasNoTerminator) {
      if (Info->returnsVoid()) {
        BuildMI(MBB, MBB.end(), DebugLoc(), get(AMDGPU::S_ENDPGM)).addImm(0);
      } else {
        BuildMI(MBB, MBB.end(), DebugLoc(), get(AMDGPU::SI_RETURN_TO_EPILOG));
      }
    }
  }
}

MachineBasicBlock *SIInstrInfo::insertSimulatedTrap(MachineRegisterInfo &MRI,
                                                    MachineBasicBlock &MBB,
                                                    MachineInstr &MI,
                                                    const DebugLoc &DL) const {
  MachineFunction *MF = MBB.getParent();
  constexpr unsigned DoorbellIDMask = 0x3ff;
  constexpr unsigned ECQueueWaveAbort = 0x400;

  MachineBasicBlock *TrapBB = &MBB;
  MachineBasicBlock *HaltLoopBB = MF->CreateMachineBasicBlock();

  if (!MBB.succ_empty() || std::next(MI.getIterator()) != MBB.end()) {
    MBB.splitAt(MI, /*UpdateLiveIns=*/false);
    TrapBB = MF->CreateMachineBasicBlock();
    BuildMI(MBB, MI, DL, get(AMDGPU::S_CBRANCH_EXECNZ)).addMBB(TrapBB);
    MF->push_back(TrapBB);
    MBB.addSuccessor(TrapBB);
  }
  // Start with a `s_trap 2`, if we're in PRIV=1 and we need the workaround this
  // will be a nop.
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_TRAP))
      .addImm(static_cast<unsigned>(GCNSubtarget::TrapID::LLVMAMDHSATrap));
  Register DoorbellReg = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_SENDMSG_RTN_B32),
          DoorbellReg)
      .addImm(AMDGPU::SendMsg::ID_RTN_GET_DOORBELL);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_MOV_B32), AMDGPU::TTMP2)
      .addUse(AMDGPU::M0);
  Register DoorbellRegMasked =
      MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_AND_B32), DoorbellRegMasked)
      .addUse(DoorbellReg)
      .addImm(DoorbellIDMask);
  Register SetWaveAbortBit =
      MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_OR_B32), SetWaveAbortBit)
      .addUse(DoorbellRegMasked)
      .addImm(ECQueueWaveAbort);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_MOV_B32), AMDGPU::M0)
      .addUse(SetWaveAbortBit);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_SENDMSG))
      .addImm(AMDGPU::SendMsg::ID_INTERRUPT);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_MOV_B32), AMDGPU::M0)
      .addUse(AMDGPU::TTMP2);
  BuildMI(*TrapBB, TrapBB->end(), DL, get(AMDGPU::S_BRANCH)).addMBB(HaltLoopBB);
  TrapBB->addSuccessor(HaltLoopBB);

  BuildMI(*HaltLoopBB, HaltLoopBB->end(), DL, get(AMDGPU::S_SETHALT)).addImm(5);
  BuildMI(*HaltLoopBB, HaltLoopBB->end(), DL, get(AMDGPU::S_BRANCH))
      .addMBB(HaltLoopBB);
  MF->push_back(HaltLoopBB);
  HaltLoopBB->addSuccessor(HaltLoopBB);

  return MBB.getNextNode();
}

unsigned SIInstrInfo::getNumWaitStates(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    if (MI.isMetaInstruction())
      return 0;
    return 1; // FIXME: Do wait states equal cycles?

  case AMDGPU::S_NOP:
    return MI.getOperand(0).getImm() + 1;
  // SI_RETURN_TO_EPILOG is a fallthrough to code outside of the function. The
  // hazard, even if one exist, won't really be visible. Should we handle it?
  }
}

bool SIInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MBB.findDebugLoc(MI);
  const AMDGPU::LaneMaskConstants &LMC = AMDGPU::LaneMaskConstants::get(ST);
  switch (MI.getOpcode()) {
  default: return TargetInstrInfo::expandPostRAPseudo(MI);
  case AMDGPU::S_MOV_B64_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_MOV_B64));
    break;

  case AMDGPU::S_MOV_B32_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_MOV_B32));
    break;

  case AMDGPU::S_XOR_B64_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_XOR_B64));
    break;

  case AMDGPU::S_XOR_B32_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_XOR_B32));
    break;
  case AMDGPU::S_OR_B64_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_OR_B64));
    break;
  case AMDGPU::S_OR_B32_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_OR_B32));
    break;

  case AMDGPU::S_ANDN2_B64_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_ANDN2_B64));
    break;

  case AMDGPU::S_ANDN2_B32_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_ANDN2_B32));
    break;

  case AMDGPU::S_AND_B64_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_AND_B64));
    break;

  case AMDGPU::S_AND_B32_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_AND_B32));
    break;

  case AMDGPU::S_AND_SAVEEXEC_B64_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_AND_SAVEEXEC_B64));
    break;

  case AMDGPU::S_AND_SAVEEXEC_B32_term:
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(get(AMDGPU::S_AND_SAVEEXEC_B32));
    break;

  case AMDGPU::SI_SPILL_S32_TO_VGPR:
    MI.setDesc(get(AMDGPU::V_WRITELANE_B32));
    break;

  case AMDGPU::SI_RESTORE_S32_FROM_VGPR:
    MI.setDesc(get(AMDGPU::V_READLANE_B32));
    break;
  case AMDGPU::AV_MOV_B32_IMM_PSEUDO: {
    Register Dst = MI.getOperand(0).getReg();
    bool IsAGPR = SIRegisterInfo::isAGPRClass(RI.getPhysRegBaseClass(Dst));
    MI.setDesc(
        get(IsAGPR ? AMDGPU::V_ACCVGPR_WRITE_B32_e64 : AMDGPU::V_MOV_B32_e32));
    break;
  }
  case AMDGPU::AV_MOV_B64_IMM_PSEUDO: {
    Register Dst = MI.getOperand(0).getReg();
    if (SIRegisterInfo::isAGPRClass(RI.getPhysRegBaseClass(Dst))) {
      int64_t Imm = MI.getOperand(1).getImm();

      Register DstLo = RI.getSubReg(Dst, AMDGPU::sub0);
      Register DstHi = RI.getSubReg(Dst, AMDGPU::sub1);
      BuildMI(MBB, MI, DL, get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), DstLo)
          .addImm(SignExtend64<32>(Imm))
          .addReg(Dst, RegState::Implicit | RegState::Define);
      BuildMI(MBB, MI, DL, get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), DstHi)
          .addImm(SignExtend64<32>(Imm >> 32))
          .addReg(Dst, RegState::Implicit | RegState::Define);
      MI.eraseFromParent();
      break;
    }

    [[fallthrough]];
  }
  case AMDGPU::V_MOV_B64_PSEUDO: {
    Register Dst = MI.getOperand(0).getReg();
    Register DstLo = RI.getSubReg(Dst, AMDGPU::sub0);
    Register DstHi = RI.getSubReg(Dst, AMDGPU::sub1);

    const MCInstrDesc &Mov64Desc = get(AMDGPU::V_MOV_B64_e32);
    const TargetRegisterClass *Mov64RC = getRegClass(Mov64Desc, /*OpNum=*/0);

    const MachineOperand &SrcOp = MI.getOperand(1);
    // FIXME: Will this work for 64-bit floating point immediates?
    assert(!SrcOp.isFPImm());
    if (ST.hasMovB64() && Mov64RC->contains(Dst)) {
      MI.setDesc(Mov64Desc);
      if (SrcOp.isReg() || isInlineConstant(MI, 1) ||
          isUInt<32>(SrcOp.getImm()) || ST.has64BitLiterals())
        break;
    }
    if (SrcOp.isImm()) {
      APInt Imm(64, SrcOp.getImm());
      APInt Lo(32, Imm.getLoBits(32).getZExtValue());
      APInt Hi(32, Imm.getHiBits(32).getZExtValue());
      const MCInstrDesc &PkMovDesc = get(AMDGPU::V_PK_MOV_B32);
      const TargetRegisterClass *PkMovRC = getRegClass(PkMovDesc, /*OpNum=*/0);

      if (ST.hasPkMovB32() && Lo == Hi && isInlineConstant(Lo) &&
          PkMovRC->contains(Dst)) {
        BuildMI(MBB, MI, DL, PkMovDesc, Dst)
            .addImm(SISrcMods::OP_SEL_1)
            .addImm(Lo.getSExtValue())
            .addImm(SISrcMods::OP_SEL_1)
            .addImm(Lo.getSExtValue())
            .addImm(0)  // op_sel_lo
            .addImm(0)  // op_sel_hi
            .addImm(0)  // neg_lo
            .addImm(0)  // neg_hi
            .addImm(0); // clamp
      } else {
        BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstLo)
          .addImm(Lo.getSExtValue())
          .addReg(Dst, RegState::Implicit | RegState::Define);
        BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstHi)
          .addImm(Hi.getSExtValue())
          .addReg(Dst, RegState::Implicit | RegState::Define);
      }
    } else {
      assert(SrcOp.isReg());
      if (ST.hasPkMovB32() &&
          !RI.isAGPR(MBB.getParent()->getRegInfo(), SrcOp.getReg())) {
        BuildMI(MBB, MI, DL, get(AMDGPU::V_PK_MOV_B32), Dst)
          .addImm(SISrcMods::OP_SEL_1) // src0_mod
          .addReg(SrcOp.getReg())
          .addImm(SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1) // src1_mod
          .addReg(SrcOp.getReg())
          .addImm(0)  // op_sel_lo
          .addImm(0)  // op_sel_hi
          .addImm(0)  // neg_lo
          .addImm(0)  // neg_hi
          .addImm(0); // clamp
      } else {
        BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstLo)
          .addReg(RI.getSubReg(SrcOp.getReg(), AMDGPU::sub0))
          .addReg(Dst, RegState::Implicit | RegState::Define);
        BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstHi)
          .addReg(RI.getSubReg(SrcOp.getReg(), AMDGPU::sub1))
          .addReg(Dst, RegState::Implicit | RegState::Define);
      }
    }
    MI.eraseFromParent();
    break;
  }
  case AMDGPU::V_MOV_B64_DPP_PSEUDO: {
    expandMovDPP64(MI);
    break;
  }
  case AMDGPU::S_MOV_B64_IMM_PSEUDO: {
    const MachineOperand &SrcOp = MI.getOperand(1);
    assert(!SrcOp.isFPImm());

    if (ST.has64BitLiterals()) {
      MI.setDesc(get(AMDGPU::S_MOV_B64));
      break;
    }

    APInt Imm(64, SrcOp.getImm());
    if (Imm.isIntN(32) || isInlineConstant(Imm)) {
      MI.setDesc(get(AMDGPU::S_MOV_B64));
      break;
    }

    Register Dst = MI.getOperand(0).getReg();
    Register DstLo = RI.getSubReg(Dst, AMDGPU::sub0);
    Register DstHi = RI.getSubReg(Dst, AMDGPU::sub1);

    APInt Lo(32, Imm.getLoBits(32).getZExtValue());
    APInt Hi(32, Imm.getHiBits(32).getZExtValue());
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B32), DstLo)
      .addImm(Lo.getSExtValue())
      .addReg(Dst, RegState::Implicit | RegState::Define);
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B32), DstHi)
      .addImm(Hi.getSExtValue())
      .addReg(Dst, RegState::Implicit | RegState::Define);
    MI.eraseFromParent();
    break;
  }
  case AMDGPU::V_SET_INACTIVE_B32: {
    // Lower V_SET_INACTIVE_B32 to V_CNDMASK_B32.
    Register DstReg = MI.getOperand(0).getReg();
    BuildMI(MBB, MI, DL, get(AMDGPU::V_CNDMASK_B32_e64), DstReg)
        .add(MI.getOperand(3))
        .add(MI.getOperand(4))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .add(MI.getOperand(5));
    MI.eraseFromParent();
    break;
  }
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V1:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V2:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V3:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V4:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V5:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V6:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V7:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V8:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V9:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V10:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V11:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V12:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V16:
  case AMDGPU::V_INDIRECT_REG_WRITE_MOVREL_B32_V32:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V1:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V2:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V3:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V4:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V5:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V6:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V7:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V8:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V9:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V10:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V11:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V12:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V16:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B32_V32:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V1:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V2:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V4:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V8:
  case AMDGPU::S_INDIRECT_REG_WRITE_MOVREL_B64_V16: {
    const TargetRegisterClass *EltRC = getOpRegClass(MI, 2);

    unsigned Opc;
    if (RI.hasVGPRs(EltRC)) {
      Opc = AMDGPU::V_MOVRELD_B32_e32;
    } else {
      Opc = RI.getRegSizeInBits(*EltRC) == 64 ? AMDGPU::S_MOVRELD_B64
                                              : AMDGPU::S_MOVRELD_B32;
    }

    const MCInstrDesc &OpDesc = get(Opc);
    Register VecReg = MI.getOperand(0).getReg();
    bool IsUndef = MI.getOperand(1).isUndef();
    unsigned SubReg = MI.getOperand(3).getImm();
    assert(VecReg == MI.getOperand(1).getReg());

    MachineInstrBuilder MIB =
        BuildMI(MBB, MI, DL, OpDesc)
            .addReg(RI.getSubReg(VecReg, SubReg), RegState::Undef)
            .add(MI.getOperand(2))
            .addReg(VecReg, RegState::ImplicitDefine)
            .addReg(VecReg, RegState::Implicit | getUndefRegState(IsUndef));

    const int ImpDefIdx =
        OpDesc.getNumOperands() + OpDesc.implicit_uses().size();
    const int ImpUseIdx = ImpDefIdx + 1;
    MIB->tieOperands(ImpDefIdx, ImpUseIdx);
    MI.eraseFromParent();
    break;
  }
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V1:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V2:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V3:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V4:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V5:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V6:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V7:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V8:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V9:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V10:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V11:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V12:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V16:
  case AMDGPU::V_INDIRECT_REG_WRITE_GPR_IDX_B32_V32: {
    assert(ST.useVGPRIndexMode());
    Register VecReg = MI.getOperand(0).getReg();
    bool IsUndef = MI.getOperand(1).isUndef();
    MachineOperand &Idx = MI.getOperand(3);
    Register SubReg = MI.getOperand(4).getImm();

    MachineInstr *SetOn = BuildMI(MBB, MI, DL, get(AMDGPU::S_SET_GPR_IDX_ON))
                              .add(Idx)
                              .addImm(AMDGPU::VGPRIndexMode::DST_ENABLE);
    SetOn->getOperand(3).setIsUndef();

    const MCInstrDesc &OpDesc = get(AMDGPU::V_MOV_B32_indirect_write);
    MachineInstrBuilder MIB =
        BuildMI(MBB, MI, DL, OpDesc)
            .addReg(RI.getSubReg(VecReg, SubReg), RegState::Undef)
            .add(MI.getOperand(2))
            .addReg(VecReg, RegState::ImplicitDefine)
            .addReg(VecReg, RegState::Implicit | getUndefRegState(IsUndef));

    const int ImpDefIdx =
        OpDesc.getNumOperands() + OpDesc.implicit_uses().size();
    const int ImpUseIdx = ImpDefIdx + 1;
    MIB->tieOperands(ImpDefIdx, ImpUseIdx);

    MachineInstr *SetOff = BuildMI(MBB, MI, DL, get(AMDGPU::S_SET_GPR_IDX_OFF));

    finalizeBundle(MBB, SetOn->getIterator(), std::next(SetOff->getIterator()));

    MI.eraseFromParent();
    break;
  }
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V1:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V2:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V3:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V4:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V5:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V6:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V7:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V8:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V9:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V10:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V11:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V12:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V16:
  case AMDGPU::V_INDIRECT_REG_READ_GPR_IDX_B32_V32: {
    assert(ST.useVGPRIndexMode());
    Register Dst = MI.getOperand(0).getReg();
    Register VecReg = MI.getOperand(1).getReg();
    bool IsUndef = MI.getOperand(1).isUndef();
    Register SubReg = MI.getOperand(3).getImm();

    MachineInstr *SetOn = BuildMI(MBB, MI, DL, get(AMDGPU::S_SET_GPR_IDX_ON))
                              .add(MI.getOperand(2))
                              .addImm(AMDGPU::VGPRIndexMode::SRC0_ENABLE);
    SetOn->getOperand(3).setIsUndef();

    BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_indirect_read))
        .addDef(Dst)
        .addReg(RI.getSubReg(VecReg, SubReg), RegState::Undef)
        .addReg(VecReg, RegState::Implicit | getUndefRegState(IsUndef));

    MachineInstr *SetOff = BuildMI(MBB, MI, DL, get(AMDGPU::S_SET_GPR_IDX_OFF));

    finalizeBundle(MBB, SetOn->getIterator(), std::next(SetOff->getIterator()));

    MI.eraseFromParent();
    break;
  }
  case AMDGPU::SI_PC_ADD_REL_OFFSET: {
    MachineFunction &MF = *MBB.getParent();
    Register Reg = MI.getOperand(0).getReg();
    Register RegLo = RI.getSubReg(Reg, AMDGPU::sub0);
    Register RegHi = RI.getSubReg(Reg, AMDGPU::sub1);
    MachineOperand OpLo = MI.getOperand(1);
    MachineOperand OpHi = MI.getOperand(2);

    // Create a bundle so these instructions won't be re-ordered by the
    // post-RA scheduler.
    MIBundleBuilder Bundler(MBB, MI);
    Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_GETPC_B64), Reg));

    // What we want here is an offset from the value returned by s_getpc (which
    // is the address of the s_add_u32 instruction) to the global variable, but
    // since the encoding of $symbol starts 4 bytes after the start of the
    // s_add_u32 instruction, we end up with an offset that is 4 bytes too
    // small. This requires us to add 4 to the global variable offset in order
    // to compute the correct address. Similarly for the s_addc_u32 instruction,
    // the encoding of $symbol starts 12 bytes after the start of the s_add_u32
    // instruction.

    int64_t Adjust = 0;
    if (ST.hasGetPCZeroExtension()) {
      // Fix up hardware that does not sign-extend the 48-bit PC value by
      // inserting: s_sext_i32_i16 reghi, reghi
      Bundler.append(
          BuildMI(MF, DL, get(AMDGPU::S_SEXT_I32_I16), RegHi).addReg(RegHi));
      Adjust += 4;
    }

    if (OpLo.isGlobal())
      OpLo.setOffset(OpLo.getOffset() + Adjust + 4);
    Bundler.append(
        BuildMI(MF, DL, get(AMDGPU::S_ADD_U32), RegLo).addReg(RegLo).add(OpLo));

    if (OpHi.isGlobal())
      OpHi.setOffset(OpHi.getOffset() + Adjust + 12);
    Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_ADDC_U32), RegHi)
                       .addReg(RegHi)
                       .add(OpHi));

    finalizeBundle(MBB, Bundler.begin());

    MI.eraseFromParent();
    break;
  }
  case AMDGPU::SI_PC_ADD_REL_OFFSET64: {
    MachineFunction &MF = *MBB.getParent();
    Register Reg = MI.getOperand(0).getReg();
    MachineOperand Op = MI.getOperand(1);

    // Create a bundle so these instructions won't be re-ordered by the
    // post-RA scheduler.
    MIBundleBuilder Bundler(MBB, MI);
    Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_GETPC_B64), Reg));
    if (Op.isGlobal())
      Op.setOffset(Op.getOffset() + 4);
    Bundler.append(
        BuildMI(MF, DL, get(AMDGPU::S_ADD_U64), Reg).addReg(Reg).add(Op));

    finalizeBundle(MBB, Bundler.begin());

    MI.eraseFromParent();
    break;
  }
  case AMDGPU::ENTER_STRICT_WWM: {
    // This only gets its own opcode so that SIPreAllocateWWMRegs can tell when
    // Whole Wave Mode is entered.
    MI.setDesc(get(LMC.OrSaveExecOpc));
    break;
  }
  case AMDGPU::ENTER_STRICT_WQM: {
    // This only gets its own opcode so that SIPreAllocateWWMRegs can tell when
    // STRICT_WQM is entered.
    BuildMI(MBB, MI, DL, get(LMC.MovOpc), MI.getOperand(0).getReg())
        .addReg(LMC.ExecReg);
    BuildMI(MBB, MI, DL, get(LMC.WQMOpc), LMC.ExecReg).addReg(LMC.ExecReg);

    MI.eraseFromParent();
    break;
  }
  case AMDGPU::EXIT_STRICT_WWM:
  case AMDGPU::EXIT_STRICT_WQM: {
    // This only gets its own opcode so that SIPreAllocateWWMRegs can tell when
    // WWM/STICT_WQM is exited.
    MI.setDesc(get(LMC.MovOpc));
    break;
  }
  case AMDGPU::SI_RETURN: {
    const MachineFunction *MF = MBB.getParent();
    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    const SIRegisterInfo *TRI = ST.getRegisterInfo();
    // Hiding the return address use with SI_RETURN may lead to extra kills in
    // the function and missing live-ins. We are fine in practice because callee
    // saved register handling ensures the register value is restored before
    // RET, but we need the undef flag here to appease the MachineVerifier
    // liveness checks.
    MachineInstrBuilder MIB =
        BuildMI(MBB, MI, DL, get(AMDGPU::S_SETPC_B64_return))
            .addReg(TRI->getReturnAddressReg(*MF), RegState::Undef);

    MIB.copyImplicitOps(MI);
    MI.eraseFromParent();
    break;
  }

  case AMDGPU::S_MUL_U64_U32_PSEUDO:
  case AMDGPU::S_MUL_I64_I32_PSEUDO:
    MI.setDesc(get(AMDGPU::S_MUL_U64));
    break;

  case AMDGPU::S_GETPC_B64_pseudo:
    MI.setDesc(get(AMDGPU::S_GETPC_B64));
    if (ST.hasGetPCZeroExtension()) {
      Register Dst = MI.getOperand(0).getReg();
      Register DstHi = RI.getSubReg(Dst, AMDGPU::sub1);
      // Fix up hardware that does not sign-extend the 48-bit PC value by
      // inserting: s_sext_i32_i16 dsthi, dsthi
      BuildMI(MBB, std::next(MI.getIterator()), DL, get(AMDGPU::S_SEXT_I32_I16),
              DstHi)
          .addReg(DstHi);
    }
    break;

  case AMDGPU::V_MAX_BF16_PSEUDO_e64: {
    assert(ST.hasBF16PackedInsts());
    MI.setDesc(get(AMDGPU::V_PK_MAX_NUM_BF16));
    MI.addOperand(MachineOperand::CreateImm(0)); // op_sel
    MI.addOperand(MachineOperand::CreateImm(0)); // neg_lo
    MI.addOperand(MachineOperand::CreateImm(0)); // neg_hi
    auto Op0 = getNamedOperand(MI, AMDGPU::OpName::src0_modifiers);
    Op0->setImm(Op0->getImm() | SISrcMods::OP_SEL_1);
    auto Op1 = getNamedOperand(MI, AMDGPU::OpName::src1_modifiers);
    Op1->setImm(Op1->getImm() | SISrcMods::OP_SEL_1);
    break;
  }

  case AMDGPU::GET_STACK_BASE:
    // The stack starts at offset 0 unless we need to reserve some space at the
    // bottom.
    if (ST.getFrameLowering()->mayReserveScratchForCWSR(*MBB.getParent())) {
      // When CWSR is used in dynamic VGPR mode, the trap handler needs to save
      // some of the VGPRs. The size of the required scratch space has already
      // been computed by prolog epilog insertion.
      const SIMachineFunctionInfo *MFI =
          MBB.getParent()->getInfo<SIMachineFunctionInfo>();
      unsigned VGPRSize = MFI->getScratchReservedForDynamicVGPRs();
      Register DestReg = MI.getOperand(0).getReg();
      BuildMI(MBB, MI, DL, get(AMDGPU::S_GETREG_B32), DestReg)
          .addImm(AMDGPU::Hwreg::HwregEncoding::encode(
              AMDGPU::Hwreg::ID_HW_ID2, AMDGPU::Hwreg::OFFSET_ME_ID, 2));
      // The MicroEngine ID is 0 for the graphics queue, and 1 or 2 for compute
      // (3 is unused, so we ignore it). Unfortunately, S_GETREG doesn't set
      // SCC, so we need to check for 0 manually.
      BuildMI(MBB, MI, DL, get(AMDGPU::S_CMP_LG_U32)).addImm(0).addReg(DestReg);
      // Change the implicif-def of SCC to an explicit use (but first remove
      // the dead flag if present).
      MI.getOperand(MI.getNumExplicitOperands()).setIsDead(false);
      MI.getOperand(MI.getNumExplicitOperands()).setIsUse();
      MI.setDesc(get(AMDGPU::S_CMOVK_I32));
      MI.addOperand(MachineOperand::CreateImm(VGPRSize));
    } else {
      MI.setDesc(get(AMDGPU::S_MOV_B32));
      MI.addOperand(MachineOperand::CreateImm(0));
      MI.removeOperand(
          MI.getNumExplicitOperands()); // Drop implicit def of SCC.
    }
    break;
  }

  return true;
}

void SIInstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I, Register DestReg,
                                unsigned SubIdx,
                                const MachineInstr &Orig) const {

  // Try shrinking the instruction to remat only the part needed for current
  // context.
  // TODO: Handle more cases.
  unsigned Opcode = Orig.getOpcode();
  switch (Opcode) {
  case AMDGPU::S_LOAD_DWORDX16_IMM:
  case AMDGPU::S_LOAD_DWORDX8_IMM: {
    if (SubIdx != 0)
      break;

    if (I == MBB.end())
      break;

    if (I->isBundled())
      break;

    // Look for a single use of the register that is also a subreg.
    Register RegToFind = Orig.getOperand(0).getReg();
    MachineOperand *UseMO = nullptr;
    for (auto &CandMO : I->operands()) {
      if (!CandMO.isReg() || CandMO.getReg() != RegToFind || CandMO.isDef())
        continue;
      if (UseMO) {
        UseMO = nullptr;
        break;
      }
      UseMO = &CandMO;
    }
    if (!UseMO || UseMO->getSubReg() == AMDGPU::NoSubRegister)
      break;

    unsigned Offset = RI.getSubRegIdxOffset(UseMO->getSubReg());
    unsigned SubregSize = RI.getSubRegIdxSize(UseMO->getSubReg());

    MachineFunction *MF = MBB.getParent();
    MachineRegisterInfo &MRI = MF->getRegInfo();
    assert(MRI.use_nodbg_empty(DestReg) && "DestReg should have no users yet.");

    unsigned NewOpcode = -1;
    if (SubregSize == 256)
      NewOpcode = AMDGPU::S_LOAD_DWORDX8_IMM;
    else if (SubregSize == 128)
      NewOpcode = AMDGPU::S_LOAD_DWORDX4_IMM;
    else
      break;

    const MCInstrDesc &TID = get(NewOpcode);
    const TargetRegisterClass *NewRC =
        RI.getAllocatableClass(getRegClass(TID, 0));
    MRI.setRegClass(DestReg, NewRC);

    UseMO->setReg(DestReg);
    UseMO->setSubReg(AMDGPU::NoSubRegister);

    // Use a smaller load with the desired size, possibly with updated offset.
    MachineInstr *MI = MF->CloneMachineInstr(&Orig);
    MI->setDesc(TID);
    MI->getOperand(0).setReg(DestReg);
    MI->getOperand(0).setSubReg(AMDGPU::NoSubRegister);
    if (Offset) {
      MachineOperand *OffsetMO = getNamedOperand(*MI, AMDGPU::OpName::offset);
      int64_t FinalOffset = OffsetMO->getImm() + Offset / 8;
      OffsetMO->setImm(FinalOffset);
    }
    SmallVector<MachineMemOperand *> NewMMOs;
    for (const MachineMemOperand *MemOp : Orig.memoperands())
      NewMMOs.push_back(MF->getMachineMemOperand(MemOp, MemOp->getPointerInfo(),
                                                 SubregSize / 8));
    MI->setMemRefs(*MF, NewMMOs);

    MBB.insert(I, MI);
    return;
  }

  default:
    break;
  }

  TargetInstrInfo::reMaterialize(MBB, I, DestReg, SubIdx, Orig);
}

std::pair<MachineInstr*, MachineInstr*>
SIInstrInfo::expandMovDPP64(MachineInstr &MI) const {
  assert (MI.getOpcode() == AMDGPU::V_MOV_B64_DPP_PSEUDO);

  if (ST.hasMovB64() && ST.hasFeature(AMDGPU::FeatureDPALU_DPP) &&
      AMDGPU::isLegalDPALU_DPPControl(
          ST, getNamedOperand(MI, AMDGPU::OpName::dpp_ctrl)->getImm())) {
    MI.setDesc(get(AMDGPU::V_MOV_B64_dpp));
    return std::pair(&MI, nullptr);
  }

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MBB.findDebugLoc(MI);
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  Register Dst = MI.getOperand(0).getReg();
  unsigned Part = 0;
  MachineInstr *Split[2];

  for (auto Sub : { AMDGPU::sub0, AMDGPU::sub1 }) {
    auto MovDPP = BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_dpp));
    if (Dst.isPhysical()) {
      MovDPP.addDef(RI.getSubReg(Dst, Sub));
    } else {
      assert(MRI.isSSA());
      auto Tmp = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
      MovDPP.addDef(Tmp);
    }

    for (unsigned I = 1; I <= 2; ++I) { // old and src operands.
      const MachineOperand &SrcOp = MI.getOperand(I);
      assert(!SrcOp.isFPImm());
      if (SrcOp.isImm()) {
        APInt Imm(64, SrcOp.getImm());
        Imm.ashrInPlace(Part * 32);
        MovDPP.addImm(Imm.getLoBits(32).getZExtValue());
      } else {
        assert(SrcOp.isReg());
        Register Src = SrcOp.getReg();
        if (Src.isPhysical())
          MovDPP.addReg(RI.getSubReg(Src, Sub));
        else
          MovDPP.addReg(Src, getUndefRegState(SrcOp.isUndef()), Sub);
      }
    }

    for (const MachineOperand &MO : llvm::drop_begin(MI.explicit_operands(), 3))
      MovDPP.addImm(MO.getImm());

    Split[Part] = MovDPP;
    ++Part;
  }

  if (Dst.isVirtual())
    BuildMI(MBB, MI, DL, get(AMDGPU::REG_SEQUENCE), Dst)
      .addReg(Split[0]->getOperand(0).getReg())
      .addImm(AMDGPU::sub0)
      .addReg(Split[1]->getOperand(0).getReg())
      .addImm(AMDGPU::sub1);

  MI.eraseFromParent();
  return std::pair(Split[0], Split[1]);
}

std::optional<DestSourcePair>
SIInstrInfo::isCopyInstrImpl(const MachineInstr &MI) const {
  if (MI.getOpcode() == AMDGPU::WWM_COPY)
    return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};

  return std::nullopt;
}

bool SIInstrInfo::swapSourceModifiers(MachineInstr &MI, MachineOperand &Src0,
                                      AMDGPU::OpName Src0OpName,
                                      MachineOperand &Src1,
                                      AMDGPU::OpName Src1OpName) const {
  MachineOperand *Src0Mods = getNamedOperand(MI, Src0OpName);
  if (!Src0Mods)
    return false;

  MachineOperand *Src1Mods = getNamedOperand(MI, Src1OpName);
  assert(Src1Mods &&
         "All commutable instructions have both src0 and src1 modifiers");

  int Src0ModsVal = Src0Mods->getImm();
  int Src1ModsVal = Src1Mods->getImm();

  Src1Mods->setImm(Src0ModsVal);
  Src0Mods->setImm(Src1ModsVal);
  return true;
}

static MachineInstr *swapRegAndNonRegOperand(MachineInstr &MI,
                                             MachineOperand &RegOp,
                                             MachineOperand &NonRegOp) {
  Register Reg = RegOp.getReg();
  unsigned SubReg = RegOp.getSubReg();
  bool IsKill = RegOp.isKill();
  bool IsDead = RegOp.isDead();
  bool IsUndef = RegOp.isUndef();
  bool IsDebug = RegOp.isDebug();

  if (NonRegOp.isImm())
    RegOp.ChangeToImmediate(NonRegOp.getImm());
  else if (NonRegOp.isFI())
    RegOp.ChangeToFrameIndex(NonRegOp.getIndex());
  else if (NonRegOp.isGlobal()) {
    RegOp.ChangeToGA(NonRegOp.getGlobal(), NonRegOp.getOffset(),
                     NonRegOp.getTargetFlags());
  } else
    return nullptr;

  // Make sure we don't reinterpret a subreg index in the target flags.
  RegOp.setTargetFlags(NonRegOp.getTargetFlags());

  NonRegOp.ChangeToRegister(Reg, false, false, IsKill, IsDead, IsUndef, IsDebug);
  NonRegOp.setSubReg(SubReg);

  return &MI;
}

static MachineInstr *swapImmOperands(MachineInstr &MI,
                                     MachineOperand &NonRegOp1,
                                     MachineOperand &NonRegOp2) {
  unsigned TargetFlags = NonRegOp1.getTargetFlags();
  int64_t NonRegVal = NonRegOp1.getImm();

  NonRegOp1.setImm(NonRegOp2.getImm());
  NonRegOp2.setImm(NonRegVal);
  NonRegOp1.setTargetFlags(NonRegOp2.getTargetFlags());
  NonRegOp2.setTargetFlags(TargetFlags);
  return &MI;
}

bool SIInstrInfo::isLegalToSwap(const MachineInstr &MI, unsigned OpIdx0,
                                unsigned OpIdx1) const {
  const MCInstrDesc &InstDesc = MI.getDesc();
  const MCOperandInfo &OpInfo0 = InstDesc.operands()[OpIdx0];
  const MCOperandInfo &OpInfo1 = InstDesc.operands()[OpIdx1];

  unsigned Opc = MI.getOpcode();
  int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);

  const MachineOperand &MO0 = MI.getOperand(OpIdx0);
  const MachineOperand &MO1 = MI.getOperand(OpIdx1);

  // Swap doesn't breach constant bus or literal limits
  // It may move literal to position other than src0, this is not allowed
  // pre-gfx10 However, most test cases need literals in Src0 for VOP
  // FIXME: After gfx9, literal can be in place other than Src0
  if (isVALU(MI)) {
    if ((int)OpIdx0 == Src0Idx && !MO0.isReg() &&
        !isInlineConstant(MO0, OpInfo1))
      return false;
    if ((int)OpIdx1 == Src0Idx && !MO1.isReg() &&
        !isInlineConstant(MO1, OpInfo0))
      return false;
  }

  if ((int)OpIdx1 != Src0Idx && MO0.isReg()) {
    if (OpInfo1.RegClass == -1)
      return OpInfo1.OperandType == MCOI::OPERAND_UNKNOWN;
    return isLegalRegOperand(MI, OpIdx1, MO0) &&
           (!MO1.isReg() || isLegalRegOperand(MI, OpIdx0, MO1));
  }
  if ((int)OpIdx0 != Src0Idx && MO1.isReg()) {
    if (OpInfo0.RegClass == -1)
      return OpInfo0.OperandType == MCOI::OPERAND_UNKNOWN;
    return (!MO0.isReg() || isLegalRegOperand(MI, OpIdx1, MO0)) &&
           isLegalRegOperand(MI, OpIdx0, MO1);
  }

  // No need to check 64-bit literals since swapping does not bring new
  // 64-bit literals into current instruction to fold to 32-bit

  return isImmOperandLegal(MI, OpIdx1, MO0);
}

MachineInstr *SIInstrInfo::commuteInstructionImpl(MachineInstr &MI, bool NewMI,
                                                  unsigned Src0Idx,
                                                  unsigned Src1Idx) const {
  assert(!NewMI && "this should never be used");

  unsigned Opc = MI.getOpcode();
  int CommutedOpcode = commuteOpcode(Opc);
  if (CommutedOpcode == -1)
    return nullptr;

  if (Src0Idx > Src1Idx)
    std::swap(Src0Idx, Src1Idx);

  assert(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0) ==
           static_cast<int>(Src0Idx) &&
         AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1) ==
           static_cast<int>(Src1Idx) &&
         "inconsistency with findCommutedOpIndices");

  if (!isLegalToSwap(MI, Src0Idx, Src1Idx))
    return nullptr;

  MachineInstr *CommutedMI = nullptr;
  MachineOperand &Src0 = MI.getOperand(Src0Idx);
  MachineOperand &Src1 = MI.getOperand(Src1Idx);
  if (Src0.isReg() && Src1.isReg()) {
    // Be sure to copy the source modifiers to the right place.
    CommutedMI =
        TargetInstrInfo::commuteInstructionImpl(MI, NewMI, Src0Idx, Src1Idx);
  } else if (Src0.isReg() && !Src1.isReg()) {
    CommutedMI = swapRegAndNonRegOperand(MI, Src0, Src1);
  } else if (!Src0.isReg() && Src1.isReg()) {
    CommutedMI = swapRegAndNonRegOperand(MI, Src1, Src0);
  } else if (Src0.isImm() && Src1.isImm()) {
    CommutedMI = swapImmOperands(MI, Src0, Src1);
  } else {
    // FIXME: Found two non registers to commute. This does happen.
    return nullptr;
  }

  if (CommutedMI) {
    swapSourceModifiers(MI, Src0, AMDGPU::OpName::src0_modifiers,
                        Src1, AMDGPU::OpName::src1_modifiers);

    swapSourceModifiers(MI, Src0, AMDGPU::OpName::src0_sel, Src1,
                        AMDGPU::OpName::src1_sel);

    CommutedMI->setDesc(get(CommutedOpcode));
  }

  return CommutedMI;
}

// This needs to be implemented because the source modifiers may be inserted
// between the true commutable operands, and the base
// TargetInstrInfo::commuteInstruction uses it.
bool SIInstrInfo::findCommutedOpIndices(const MachineInstr &MI,
                                        unsigned &SrcOpIdx0,
                                        unsigned &SrcOpIdx1) const {
  return findCommutedOpIndices(MI.getDesc(), SrcOpIdx0, SrcOpIdx1);
}

bool SIInstrInfo::findCommutedOpIndices(const MCInstrDesc &Desc,
                                        unsigned &SrcOpIdx0,
                                        unsigned &SrcOpIdx1) const {
  if (!Desc.isCommutable())
    return false;

  unsigned Opc = Desc.getOpcode();
  int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  if (Src0Idx == -1)
    return false;

  int Src1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1);
  if (Src1Idx == -1)
    return false;

  return fixCommutedOpIndices(SrcOpIdx0, SrcOpIdx1, Src0Idx, Src1Idx);
}

bool SIInstrInfo::isBranchOffsetInRange(unsigned BranchOp,
                                        int64_t BrOffset) const {
  // BranchRelaxation should never have to check s_setpc_b64 or s_add_pc_i64
  // because its dest block is unanalyzable.
  assert(isSOPP(BranchOp) || isSOPK(BranchOp));

  // Convert to dwords.
  BrOffset /= 4;

  // The branch instructions do PC += signext(SIMM16 * 4) + 4, so the offset is
  // from the next instruction.
  BrOffset -= 1;

  return isIntN(BranchOffsetBits, BrOffset);
}

MachineBasicBlock *
SIInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  return MI.getOperand(0).getMBB();
}

bool SIInstrInfo::hasDivergentBranch(const MachineBasicBlock *MBB) const {
  for (const MachineInstr &MI : MBB->terminators()) {
    if (MI.getOpcode() == AMDGPU::SI_IF || MI.getOpcode() == AMDGPU::SI_ELSE ||
        MI.getOpcode() == AMDGPU::SI_LOOP)
      return true;
  }
  return false;
}

void SIInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                                       MachineBasicBlock &DestBB,
                                       MachineBasicBlock &RestoreBB,
                                       const DebugLoc &DL, int64_t BrOffset,
                                       RegScavenger *RS) const {
  assert(MBB.empty() &&
         "new block should be inserted for expanding unconditional branch");
  assert(MBB.pred_size() == 1);
  assert(RestoreBB.empty() &&
         "restore block should be inserted for restoring clobbered registers");

  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  auto I = MBB.end();
  auto &MCCtx = MF->getContext();

  if (ST.useAddPC64Inst()) {
    MCSymbol *Offset =
        MCCtx.createTempSymbol("offset", /*AlwaysAddSuffix=*/true);
    auto AddPC = BuildMI(MBB, I, DL, get(AMDGPU::S_ADD_PC_I64))
                     .addSym(Offset, MO_FAR_BRANCH_OFFSET);
    MCSymbol *PostAddPCLabel =
        MCCtx.createTempSymbol("post_addpc", /*AlwaysAddSuffix=*/true);
    AddPC->setPostInstrSymbol(*MF, PostAddPCLabel);
    auto *OffsetExpr = MCBinaryExpr::createSub(
        MCSymbolRefExpr::create(DestBB.getSymbol(), MCCtx),
        MCSymbolRefExpr::create(PostAddPCLabel, MCCtx), MCCtx);
    Offset->setVariableValue(OffsetExpr);
    return;
  }

  assert(RS && "RegScavenger required for long branching");

  // FIXME: Virtual register workaround for RegScavenger not working with empty
  // blocks.
  Register PCReg = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);

  // Note: as this is used after hazard recognizer we need to apply some hazard
  // workarounds directly.
  const bool FlushSGPRWrites = (ST.isWave64() && ST.hasVALUMaskWriteHazard()) ||
                               ST.hasVALUReadSGPRHazard();
  auto ApplyHazardWorkarounds = [this, &MBB, &I, &DL, FlushSGPRWrites]() {
    if (FlushSGPRWrites)
      BuildMI(MBB, I, DL, get(AMDGPU::S_WAITCNT_DEPCTR))
          .addImm(AMDGPU::DepCtr::encodeFieldSaSdst(0, ST));
  };

  // We need to compute the offset relative to the instruction immediately after
  // s_getpc_b64. Insert pc arithmetic code before last terminator.
  MachineInstr *GetPC = BuildMI(MBB, I, DL, get(AMDGPU::S_GETPC_B64), PCReg);
  ApplyHazardWorkarounds();

  MCSymbol *PostGetPCLabel =
      MCCtx.createTempSymbol("post_getpc", /*AlwaysAddSuffix=*/true);
  GetPC->setPostInstrSymbol(*MF, PostGetPCLabel);

  MCSymbol *OffsetLo =
      MCCtx.createTempSymbol("offset_lo", /*AlwaysAddSuffix=*/true);
  MCSymbol *OffsetHi =
      MCCtx.createTempSymbol("offset_hi", /*AlwaysAddSuffix=*/true);
  BuildMI(MBB, I, DL, get(AMDGPU::S_ADD_U32))
      .addReg(PCReg, RegState::Define, AMDGPU::sub0)
      .addReg(PCReg, {}, AMDGPU::sub0)
      .addSym(OffsetLo, MO_FAR_BRANCH_OFFSET);
  BuildMI(MBB, I, DL, get(AMDGPU::S_ADDC_U32))
      .addReg(PCReg, RegState::Define, AMDGPU::sub1)
      .addReg(PCReg, {}, AMDGPU::sub1)
      .addSym(OffsetHi, MO_FAR_BRANCH_OFFSET);
  ApplyHazardWorkarounds();

  // Insert the indirect branch after the other terminator.
  BuildMI(&MBB, DL, get(AMDGPU::S_SETPC_B64))
    .addReg(PCReg);

  // If a spill is needed for the pc register pair, we need to insert a spill
  // restore block right before the destination block, and insert a short branch
  // into the old destination block's fallthrough predecessor.
  // e.g.:
  //
  // s_cbranch_scc0 skip_long_branch:
  //
  // long_branch_bb:
  //   spill s[8:9]
  //   s_getpc_b64 s[8:9]
  //   s_add_u32 s8, s8, restore_bb
  //   s_addc_u32 s9, s9, 0
  //   s_setpc_b64 s[8:9]
  //
  // skip_long_branch:
  //   foo;
  //
  // .....
  //
  // dest_bb_fallthrough_predecessor:
  // bar;
  // s_branch dest_bb
  //
  // restore_bb:
  //  restore s[8:9]
  //  fallthrough dest_bb
  ///
  // dest_bb:
  //   buzz;

  Register LongBranchReservedReg = MFI->getLongBranchReservedReg();
  Register Scav;

  // If we've previously reserved a register for long branches
  // avoid running the scavenger and just use those registers
  if (LongBranchReservedReg) {
    RS->enterBasicBlock(MBB);
    Scav = LongBranchReservedReg;
  } else {
    RS->enterBasicBlockEnd(MBB);
    Scav = RS->scavengeRegisterBackwards(
        AMDGPU::SReg_64RegClass, MachineBasicBlock::iterator(GetPC),
        /* RestoreAfter */ false, 0, /* AllowSpill */ false);
  }
  if (Scav) {
    RS->setRegUsed(Scav);
    MRI.replaceRegWith(PCReg, Scav);
    MRI.clearVirtRegs();
  } else {
    // As SGPR needs VGPR to be spilled, we reuse the slot of temporary VGPR for
    // SGPR spill.
    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    const SIRegisterInfo *TRI = ST.getRegisterInfo();
    TRI->spillEmergencySGPR(GetPC, RestoreBB, AMDGPU::SGPR0_SGPR1, RS);
    MRI.replaceRegWith(PCReg, AMDGPU::SGPR0_SGPR1);
    MRI.clearVirtRegs();
  }

  MCSymbol *DestLabel = Scav ? DestBB.getSymbol() : RestoreBB.getSymbol();
  // Now, the distance could be defined.
  auto *Offset = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(DestLabel, MCCtx),
      MCSymbolRefExpr::create(PostGetPCLabel, MCCtx), MCCtx);
  // Add offset assignments.
  auto *Mask = MCConstantExpr::create(0xFFFFFFFFULL, MCCtx);
  OffsetLo->setVariableValue(MCBinaryExpr::createAnd(Offset, Mask, MCCtx));
  auto *ShAmt = MCConstantExpr::create(32, MCCtx);
  OffsetHi->setVariableValue(MCBinaryExpr::createAShr(Offset, ShAmt, MCCtx));
}

unsigned SIInstrInfo::getBranchOpcode(SIInstrInfo::BranchPredicate Cond) {
  switch (Cond) {
  case SIInstrInfo::SCC_TRUE:
    return AMDGPU::S_CBRANCH_SCC1;
  case SIInstrInfo::SCC_FALSE:
    return AMDGPU::S_CBRANCH_SCC0;
  case SIInstrInfo::VCCNZ:
    return AMDGPU::S_CBRANCH_VCCNZ;
  case SIInstrInfo::VCCZ:
    return AMDGPU::S_CBRANCH_VCCZ;
  case SIInstrInfo::EXECNZ:
    return AMDGPU::S_CBRANCH_EXECNZ;
  case SIInstrInfo::EXECZ:
    return AMDGPU::S_CBRANCH_EXECZ;
  default:
    llvm_unreachable("invalid branch predicate");
  }
}

SIInstrInfo::BranchPredicate SIInstrInfo::getBranchPredicate(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::S_CBRANCH_SCC0:
    return SCC_FALSE;
  case AMDGPU::S_CBRANCH_SCC1:
    return SCC_TRUE;
  case AMDGPU::S_CBRANCH_VCCNZ:
    return VCCNZ;
  case AMDGPU::S_CBRANCH_VCCZ:
    return VCCZ;
  case AMDGPU::S_CBRANCH_EXECNZ:
    return EXECNZ;
  case AMDGPU::S_CBRANCH_EXECZ:
    return EXECZ;
  default:
    return INVALID_BR;
  }
}

bool SIInstrInfo::analyzeBranchImpl(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    MachineBasicBlock *&TBB,
                                    MachineBasicBlock *&FBB,
                                    SmallVectorImpl<MachineOperand> &Cond,
                                    bool AllowModify) const {
  if (I->getOpcode() == AMDGPU::S_BRANCH) {
    // Unconditional Branch
    TBB = I->getOperand(0).getMBB();
    return false;
  }

  BranchPredicate Pred = getBranchPredicate(I->getOpcode());
  if (Pred == INVALID_BR)
    return true;

  MachineBasicBlock *CondBB = I->getOperand(0).getMBB();
  Cond.push_back(MachineOperand::CreateImm(Pred));
  Cond.push_back(I->getOperand(1)); // Save the branch register.

  ++I;

  if (I == MBB.end()) {
    // Conditional branch followed by fall-through.
    TBB = CondBB;
    return false;
  }

  if (I->getOpcode() == AMDGPU::S_BRANCH) {
    TBB = CondBB;
    FBB = I->getOperand(0).getMBB();
    return false;
  }

  return true;
}

bool SIInstrInfo::analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                                MachineBasicBlock *&FBB,
                                SmallVectorImpl<MachineOperand> &Cond,
                                bool AllowModify) const {
  MachineBasicBlock::iterator I = MBB.getFirstTerminator();
  auto E = MBB.end();
  if (I == E)
    return false;

  // Skip over the instructions that are artificially terminators for special
  // exec management.
  while (I != E && !I->isBranch() && !I->isReturn()) {
    switch (I->getOpcode()) {
    case AMDGPU::S_MOV_B64_term:
    case AMDGPU::S_XOR_B64_term:
    case AMDGPU::S_OR_B64_term:
    case AMDGPU::S_ANDN2_B64_term:
    case AMDGPU::S_AND_B64_term:
    case AMDGPU::S_AND_SAVEEXEC_B64_term:
    case AMDGPU::S_MOV_B32_term:
    case AMDGPU::S_XOR_B32_term:
    case AMDGPU::S_OR_B32_term:
    case AMDGPU::S_ANDN2_B32_term:
    case AMDGPU::S_AND_B32_term:
    case AMDGPU::S_AND_SAVEEXEC_B32_term:
      break;
    case AMDGPU::SI_IF:
    case AMDGPU::SI_ELSE:
    case AMDGPU::SI_KILL_I1_TERMINATOR:
    case AMDGPU::SI_KILL_F32_COND_IMM_TERMINATOR:
      // FIXME: It's messy that these need to be considered here at all.
      return true;
    default:
      llvm_unreachable("unexpected non-branch terminator inst");
    }

    ++I;
  }

  if (I == E)
    return false;

  return analyzeBranchImpl(MBB, I, TBB, FBB, Cond, AllowModify);
}

unsigned SIInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                   int *BytesRemoved) const {
  unsigned Count = 0;
  unsigned RemovedSize = 0;
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB.terminators())) {
    // Skip over artificial terminators when removing instructions.
    if (MI.isBranch() || MI.isReturn()) {
      RemovedSize += getInstSizeInBytes(MI);
      MI.eraseFromParent();
      ++Count;
    }
  }

  if (BytesRemoved)
    *BytesRemoved = RemovedSize;

  return Count;
}

// Copy the flags onto the implicit condition register operand.
static void preserveCondRegFlags(MachineOperand &CondReg,
                                 const MachineOperand &OrigCond) {
  CondReg.setIsUndef(OrigCond.isUndef());
  CondReg.setIsKill(OrigCond.isKill());
}

unsigned SIInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *TBB,
                                   MachineBasicBlock *FBB,
                                   ArrayRef<MachineOperand> Cond,
                                   const DebugLoc &DL,
                                   int *BytesAdded) const {
  if (!FBB && Cond.empty()) {
    BuildMI(&MBB, DL, get(AMDGPU::S_BRANCH))
      .addMBB(TBB);
    if (BytesAdded)
      *BytesAdded = ST.hasOffset3fBug() ? 8 : 4;
    return 1;
  }

  assert(TBB && Cond[0].isImm());

  unsigned Opcode
    = getBranchOpcode(static_cast<BranchPredicate>(Cond[0].getImm()));

  if (!FBB) {
    MachineInstr *CondBr =
      BuildMI(&MBB, DL, get(Opcode))
      .addMBB(TBB);

    // Copy the flags onto the implicit condition register operand.
    preserveCondRegFlags(CondBr->getOperand(1), Cond[1]);
    fixImplicitOperands(*CondBr);

    if (BytesAdded)
      *BytesAdded = ST.hasOffset3fBug() ? 8 : 4;
    return 1;
  }

  assert(TBB && FBB);

  MachineInstr *CondBr =
    BuildMI(&MBB, DL, get(Opcode))
    .addMBB(TBB);
  fixImplicitOperands(*CondBr);
  BuildMI(&MBB, DL, get(AMDGPU::S_BRANCH))
    .addMBB(FBB);

  MachineOperand &CondReg = CondBr->getOperand(1);
  CondReg.setIsUndef(Cond[1].isUndef());
  CondReg.setIsKill(Cond[1].isKill());

  if (BytesAdded)
    *BytesAdded = ST.hasOffset3fBug() ? 16 : 8;

  return 2;
}

bool SIInstrInfo::reverseBranchCondition(
  SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond.size() != 2) {
    return true;
  }

  if (Cond[0].isImm()) {
    Cond[0].setImm(-Cond[0].getImm());
    return false;
  }

  return true;
}

bool SIInstrInfo::canInsertSelect(const MachineBasicBlock &MBB,
                                  ArrayRef<MachineOperand> Cond,
                                  Register DstReg, Register TrueReg,
                                  Register FalseReg, int &CondCycles,
                                  int &TrueCycles, int &FalseCycles) const {
  switch (Cond[0].getImm()) {
  case VCCNZ:
  case VCCZ: {
    const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
    const TargetRegisterClass *RC = MRI.getRegClass(TrueReg);
    if (MRI.getRegClass(FalseReg) != RC)
      return false;

    int NumInsts = AMDGPU::getRegBitWidth(*RC) / 32;
    CondCycles = TrueCycles = FalseCycles = NumInsts; // ???

    // Limit to equal cost for branch vs. N v_cndmask_b32s.
    return RI.hasVGPRs(RC) && NumInsts <= 6;
  }
  case SCC_TRUE:
  case SCC_FALSE: {
    // FIXME: We could insert for VGPRs if we could replace the original compare
    // with a vector one.
    const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
    const TargetRegisterClass *RC = MRI.getRegClass(TrueReg);
    if (MRI.getRegClass(FalseReg) != RC)
      return false;

    int NumInsts = AMDGPU::getRegBitWidth(*RC) / 32;

    // Multiples of 8 can do s_cselect_b64
    if (NumInsts % 2 == 0)
      NumInsts /= 2;

    CondCycles = TrueCycles = FalseCycles = NumInsts; // ???
    return RI.isSGPRClass(RC);
  }
  default:
    return false;
  }
}

void SIInstrInfo::insertSelect(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, const DebugLoc &DL,
                               Register DstReg, ArrayRef<MachineOperand> Cond,
                               Register TrueReg, Register FalseReg) const {
  BranchPredicate Pred = static_cast<BranchPredicate>(Cond[0].getImm());
  if (Pred == VCCZ || Pred == SCC_FALSE) {
    Pred = static_cast<BranchPredicate>(-Pred);
    std::swap(TrueReg, FalseReg);
  }

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *DstRC = MRI.getRegClass(DstReg);
  unsigned DstSize = RI.getRegSizeInBits(*DstRC);

  if (DstSize == 32) {
    MachineInstr *Select;
    if (Pred == SCC_TRUE) {
      Select = BuildMI(MBB, I, DL, get(AMDGPU::S_CSELECT_B32), DstReg)
        .addReg(TrueReg)
        .addReg(FalseReg);
    } else {
      // Instruction's operands are backwards from what is expected.
      Select = BuildMI(MBB, I, DL, get(AMDGPU::V_CNDMASK_B32_e32), DstReg)
        .addReg(FalseReg)
        .addReg(TrueReg);
    }

    preserveCondRegFlags(Select->getOperand(3), Cond[1]);
    return;
  }

  if (DstSize == 64 && Pred == SCC_TRUE) {
    MachineInstr *Select =
      BuildMI(MBB, I, DL, get(AMDGPU::S_CSELECT_B64), DstReg)
      .addReg(TrueReg)
      .addReg(FalseReg);

    preserveCondRegFlags(Select->getOperand(3), Cond[1]);
    return;
  }

  static const int16_t Sub0_15[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7,
    AMDGPU::sub8, AMDGPU::sub9, AMDGPU::sub10, AMDGPU::sub11,
    AMDGPU::sub12, AMDGPU::sub13, AMDGPU::sub14, AMDGPU::sub15,
  };

  static const int16_t Sub0_15_64[] = {
    AMDGPU::sub0_sub1, AMDGPU::sub2_sub3,
    AMDGPU::sub4_sub5, AMDGPU::sub6_sub7,
    AMDGPU::sub8_sub9, AMDGPU::sub10_sub11,
    AMDGPU::sub12_sub13, AMDGPU::sub14_sub15,
  };

  unsigned SelOp = AMDGPU::V_CNDMASK_B32_e32;
  const TargetRegisterClass *EltRC = &AMDGPU::VGPR_32RegClass;
  const int16_t *SubIndices = Sub0_15;
  int NElts = DstSize / 32;

  // 64-bit select is only available for SALU.
  // TODO: Split 96-bit into 64-bit and 32-bit, not 3x 32-bit.
  if (Pred == SCC_TRUE) {
    if (NElts % 2) {
      SelOp = AMDGPU::S_CSELECT_B32;
      EltRC = &AMDGPU::SGPR_32RegClass;
    } else {
      SelOp = AMDGPU::S_CSELECT_B64;
      EltRC = &AMDGPU::SGPR_64RegClass;
      SubIndices = Sub0_15_64;
      NElts /= 2;
    }
  }

  MachineInstrBuilder MIB = BuildMI(
    MBB, I, DL, get(AMDGPU::REG_SEQUENCE), DstReg);

  I = MIB->getIterator();

  SmallVector<Register, 8> Regs;
  for (int Idx = 0; Idx != NElts; ++Idx) {
    Register DstElt = MRI.createVirtualRegister(EltRC);
    Regs.push_back(DstElt);

    unsigned SubIdx = SubIndices[Idx];

    MachineInstr *Select;
    if (SelOp == AMDGPU::V_CNDMASK_B32_e32) {
      Select = BuildMI(MBB, I, DL, get(SelOp), DstElt)
                   .addReg(FalseReg, {}, SubIdx)
                   .addReg(TrueReg, {}, SubIdx);
    } else {
      Select = BuildMI(MBB, I, DL, get(SelOp), DstElt)
                   .addReg(TrueReg, {}, SubIdx)
                   .addReg(FalseReg, {}, SubIdx);
    }

    preserveCondRegFlags(Select->getOperand(3), Cond[1]);
    fixImplicitOperands(*Select);

    MIB.addReg(DstElt)
       .addImm(SubIdx);
  }
}

bool SIInstrInfo::isFoldableCopy(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::V_MOV_B16_t16_e32:
  case AMDGPU::V_MOV_B16_t16_e64:
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
  case AMDGPU::V_MOV_B64_PSEUDO:
  case AMDGPU::V_MOV_B64_e32:
  case AMDGPU::V_MOV_B64_e64:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::S_MOV_B64_IMM_PSEUDO:
  case AMDGPU::COPY:
  case AMDGPU::WWM_COPY:
  case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
  case AMDGPU::V_ACCVGPR_READ_B32_e64:
  case AMDGPU::V_ACCVGPR_MOV_B32:
  case AMDGPU::AV_MOV_B32_IMM_PSEUDO:
  case AMDGPU::AV_MOV_B64_IMM_PSEUDO:
    return true;
  default:
    return false;
  }
}

unsigned SIInstrInfo::getFoldableCopySrcIdx(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::V_MOV_B16_t16_e32:
  case AMDGPU::V_MOV_B16_t16_e64:
    return 2;
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
  case AMDGPU::V_MOV_B64_PSEUDO:
  case AMDGPU::V_MOV_B64_e32:
  case AMDGPU::V_MOV_B64_e64:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::S_MOV_B64_IMM_PSEUDO:
  case AMDGPU::COPY:
  case AMDGPU::WWM_COPY:
  case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
  case AMDGPU::V_ACCVGPR_READ_B32_e64:
  case AMDGPU::V_ACCVGPR_MOV_B32:
  case AMDGPU::AV_MOV_B32_IMM_PSEUDO:
  case AMDGPU::AV_MOV_B64_IMM_PSEUDO:
    return 1;
  default:
    llvm_unreachable("MI is not a foldable copy");
  }
}

static constexpr AMDGPU::OpName ModifierOpNames[] = {
    AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
    AMDGPU::OpName::src2_modifiers, AMDGPU::OpName::clamp,
    AMDGPU::OpName::omod,           AMDGPU::OpName::op_sel};

void SIInstrInfo::removeModOperands(MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  for (AMDGPU::OpName Name : reverse(ModifierOpNames)) {
    int Idx = AMDGPU::getNamedOperandIdx(Opc, Name);
    if (Idx >= 0)
      MI.removeOperand(Idx);
  }
}

void SIInstrInfo::mutateAndCleanupImplicit(MachineInstr &MI,
                                           const MCInstrDesc &NewDesc) const {
  MI.setDesc(NewDesc);

  // Remove any leftover implicit operands from mutating the instruction. e.g.
  // if we replace an s_and_b32 with a copy, we don't need the implicit scc def
  // anymore.
  const MCInstrDesc &Desc = MI.getDesc();
  unsigned NumOps = Desc.getNumOperands() + Desc.implicit_uses().size() +
                    Desc.implicit_defs().size();

  for (unsigned I = MI.getNumOperands() - 1; I >= NumOps; --I)
    MI.removeOperand(I);
}

std::optional<int64_t> SIInstrInfo::extractSubregFromImm(int64_t Imm,
                                                         unsigned SubRegIndex) {
  switch (SubRegIndex) {
  case AMDGPU::NoSubRegister:
    return Imm;
  case AMDGPU::sub0:
    return SignExtend64<32>(Imm);
  case AMDGPU::sub1:
    return SignExtend64<32>(Imm >> 32);
  case AMDGPU::lo16:
    return SignExtend64<16>(Imm);
  case AMDGPU::hi16:
    return SignExtend64<16>(Imm >> 16);
  case AMDGPU::sub1_lo16:
    return SignExtend64<16>(Imm >> 32);
  case AMDGPU::sub1_hi16:
    return SignExtend64<16>(Imm >> 48);
  default:
    return std::nullopt;
  }

  llvm_unreachable("covered subregister switch");
}

static unsigned getNewFMAAKInst(const GCNSubtarget &ST, unsigned Opc) {
  switch (Opc) {
  case AMDGPU::V_MAC_F16_e32:
  case AMDGPU::V_MAC_F16_e64:
  case AMDGPU::V_MAD_F16_e64:
    return AMDGPU::V_MADAK_F16;
  case AMDGPU::V_MAC_F32_e32:
  case AMDGPU::V_MAC_F32_e64:
  case AMDGPU::V_MAD_F32_e64:
    return AMDGPU::V_MADAK_F32;
  case AMDGPU::V_FMAC_F32_e32:
  case AMDGPU::V_FMAC_F32_e64:
  case AMDGPU::V_FMA_F32_e64:
    return AMDGPU::V_FMAAK_F32;
  case AMDGPU::V_FMAC_F16_e32:
  case AMDGPU::V_FMAC_F16_e64:
  case AMDGPU::V_FMAC_F16_t16_e64:
  case AMDGPU::V_FMAC_F16_fake16_e64:
  case AMDGPU::V_FMAC_F16_t16_e32:
  case AMDGPU::V_FMAC_F16_fake16_e32:
  case AMDGPU::V_FMA_F16_e64:
    return ST.hasTrue16BitInsts() ? ST.useRealTrue16Insts()
                                        ? AMDGPU::V_FMAAK_F16_t16
                                        : AMDGPU::V_FMAAK_F16_fake16
                                  : AMDGPU::V_FMAAK_F16;
  case AMDGPU::V_FMAC_F64_e32:
  case AMDGPU::V_FMAC_F64_e64:
  case AMDGPU::V_FMA_F64_e64:
    return AMDGPU::V_FMAAK_F64;
  default:
    llvm_unreachable("invalid instruction");
  }
}

static unsigned getNewFMAMKInst(const GCNSubtarget &ST, unsigned Opc) {
  switch (Opc) {
  case AMDGPU::V_MAC_F16_e32:
  case AMDGPU::V_MAC_F16_e64:
  case AMDGPU::V_MAD_F16_e64:
    return AMDGPU::V_MADMK_F16;
  case AMDGPU::V_MAC_F32_e32:
  case AMDGPU::V_MAC_F32_e64:
  case AMDGPU::V_MAD_F32_e64:
    return AMDGPU::V_MADMK_F32;
  case AMDGPU::V_FMAC_F32_e32:
  case AMDGPU::V_FMAC_F32_e64:
  case AMDGPU::V_FMA_F32_e64:
    return AMDGPU::V_FMAMK_F32;
  case AMDGPU::V_FMAC_F16_e32:
  case AMDGPU::V_FMAC_F16_e64:
  case AMDGPU::V_FMAC_F16_t16_e64:
  case AMDGPU::V_FMAC_F16_fake16_e64:
  case AMDGPU::V_FMAC_F16_t16_e32:
  case AMDGPU::V_FMAC_F16_fake16_e32:
  case AMDGPU::V_FMA_F16_e64:
    return ST.hasTrue16BitInsts() ? ST.useRealTrue16Insts()
                                        ? AMDGPU::V_FMAMK_F16_t16
                                        : AMDGPU::V_FMAMK_F16_fake16
                                  : AMDGPU::V_FMAMK_F16;
  case AMDGPU::V_FMAC_F64_e32:
  case AMDGPU::V_FMAC_F64_e64:
  case AMDGPU::V_FMA_F64_e64:
    return AMDGPU::V_FMAMK_F64;
  default:
    llvm_unreachable("invalid instruction");
  }
}

bool SIInstrInfo::foldImmediate(MachineInstr &UseMI, MachineInstr &DefMI,
                                Register Reg, MachineRegisterInfo *MRI) const {
  int64_t Imm;
  if (!getConstValDefinedInReg(DefMI, Reg, Imm))
    return false;

  const bool HasMultipleUses = !MRI->hasOneNonDBGUse(Reg);

  assert(!DefMI.getOperand(0).getSubReg() && "Expected SSA form");

  unsigned Opc = UseMI.getOpcode();
  if (Opc == AMDGPU::COPY) {
    assert(!UseMI.getOperand(0).getSubReg() && "Expected SSA form");

    Register DstReg = UseMI.getOperand(0).getReg();
    Register UseSubReg = UseMI.getOperand(1).getSubReg();

    const TargetRegisterClass *DstRC = RI.getRegClassForReg(*MRI, DstReg);

    if (HasMultipleUses) {
      // TODO: This should fold in more cases with multiple use, but we need to
      // more carefully consider what those uses are.
      unsigned ImmDefSize = RI.getRegSizeInBits(*MRI->getRegClass(Reg));

      // Avoid breaking up a 64-bit inline immediate into a subregister extract.
      if (UseSubReg != AMDGPU::NoSubRegister && ImmDefSize == 64)
        return false;

      // Most of the time folding a 32-bit inline constant is free (though this
      // might not be true if we can't later fold it into a real user).
      //
      // FIXME: This isInlineConstant check is imprecise if
      // getConstValDefinedInReg handled the tricky non-mov cases.
      if (ImmDefSize == 32 &&
          !isInlineConstant(Imm, AMDGPU::OPERAND_REG_IMM_INT32))
        return false;
    }

    bool Is16Bit = UseSubReg != AMDGPU::NoSubRegister &&
                   RI.getSubRegIdxSize(UseSubReg) == 16;

    if (Is16Bit) {
      if (RI.hasVGPRs(DstRC))
        return false; // Do not clobber vgpr_hi16

      if (DstReg.isVirtual() && UseSubReg != AMDGPU::lo16)
        return false;
    }

    MachineFunction *MF = UseMI.getMF();

    unsigned NewOpc = AMDGPU::INSTRUCTION_LIST_END;
    MCRegister MovDstPhysReg =
        DstReg.isPhysical() ? DstReg.asMCReg() : MCRegister();

    std::optional<int64_t> SubRegImm = extractSubregFromImm(Imm, UseSubReg);

    // TODO: Try to fold with AMDGPU::V_MOV_B16_t16_e64
    for (unsigned MovOp :
         {AMDGPU::S_MOV_B32, AMDGPU::V_MOV_B32_e32, AMDGPU::S_MOV_B64,
          AMDGPU::V_MOV_B64_PSEUDO, AMDGPU::V_ACCVGPR_WRITE_B32_e64}) {
      const MCInstrDesc &MovDesc = get(MovOp);

      const TargetRegisterClass *MovDstRC = getRegClass(MovDesc, 0);
      if (Is16Bit) {
        // We just need to find a correctly sized register class, so the
        // subregister index compatibility doesn't matter since we're statically
        // extracting the immediate value.
        MovDstRC = RI.getMatchingSuperRegClass(MovDstRC, DstRC, AMDGPU::lo16);
        if (!MovDstRC)
          continue;

        if (MovDstPhysReg) {
          // FIXME: We probably should not do this. If there is a live value in
          // the high half of the register, it will be corrupted.
          MovDstPhysReg =
              RI.getMatchingSuperReg(MovDstPhysReg, AMDGPU::lo16, MovDstRC);
          if (!MovDstPhysReg)
            continue;
        }
      }

      // Result class isn't the right size, try the next instruction.
      if (MovDstPhysReg) {
        if (!MovDstRC->contains(MovDstPhysReg))
          return false;
      } else if (!MRI->constrainRegClass(DstReg, MovDstRC)) {
        // TODO: This will be overly conservative in the case of 16-bit virtual
        // SGPRs. We could hack up the virtual register uses to use a compatible
        // 32-bit class.
        continue;
      }

      const MCOperandInfo &OpInfo = MovDesc.operands()[1];

      // Ensure the interpreted immediate value is a valid operand in the new
      // mov.
      //
      // FIXME: isImmOperandLegal should have form that doesn't require existing
      // MachineInstr or MachineOperand
      if (!RI.opCanUseLiteralConstant(OpInfo.OperandType) &&
          !isInlineConstant(*SubRegImm, OpInfo.OperandType))
        break;

      NewOpc = MovOp;
      break;
    }

    if (NewOpc == AMDGPU::INSTRUCTION_LIST_END)
      return false;

    if (Is16Bit) {
      UseMI.getOperand(0).setSubReg(AMDGPU::NoSubRegister);
      if (MovDstPhysReg)
        UseMI.getOperand(0).setReg(MovDstPhysReg);
      assert(UseMI.getOperand(1).getReg().isVirtual());
    }

    const MCInstrDesc &NewMCID = get(NewOpc);
    UseMI.setDesc(NewMCID);
    UseMI.getOperand(1).ChangeToImmediate(*SubRegImm);
    UseMI.addImplicitDefUseOperands(*MF);
    return true;
  }

  if (HasMultipleUses)
    return false;

  if (Opc == AMDGPU::V_MAD_F32_e64 || Opc == AMDGPU::V_MAC_F32_e64 ||
      Opc == AMDGPU::V_MAD_F16_e64 || Opc == AMDGPU::V_MAC_F16_e64 ||
      Opc == AMDGPU::V_FMA_F32_e64 || Opc == AMDGPU::V_FMAC_F32_e64 ||
      Opc == AMDGPU::V_FMA_F16_e64 || Opc == AMDGPU::V_FMAC_F16_e64 ||
      Opc == AMDGPU::V_FMAC_F16_t16_e64 ||
      Opc == AMDGPU::V_FMAC_F16_fake16_e64 || Opc == AMDGPU::V_FMA_F64_e64 ||
      Opc == AMDGPU::V_FMAC_F64_e64) {
    // Don't fold if we are using source or output modifiers. The new VOP2
    // instructions don't have them.
    if (hasAnyModifiersSet(UseMI))
      return false;

    // If this is a free constant, there's no reason to do this.
    // TODO: We could fold this here instead of letting SIFoldOperands do it
    // later.
    int Src0Idx = getNamedOperandIdx(UseMI.getOpcode(), AMDGPU::OpName::src0);

    // Any src operand can be used for the legality check.
    if (isInlineConstant(UseMI, Src0Idx, Imm))
      return false;

    MachineOperand *Src0 = &UseMI.getOperand(Src0Idx);

    MachineOperand *Src1 = getNamedOperand(UseMI, AMDGPU::OpName::src1);
    MachineOperand *Src2 = getNamedOperand(UseMI, AMDGPU::OpName::src2);

    auto CopyRegOperandToNarrowerRC =
        [MRI, this](MachineInstr &MI, unsigned OpNo,
                    const TargetRegisterClass *NewRC) -> void {
      if (!MI.getOperand(OpNo).isReg())
        return;
      Register Reg = MI.getOperand(OpNo).getReg();
      const TargetRegisterClass *RC = RI.getRegClassForReg(*MRI, Reg);
      if (RI.getCommonSubClass(RC, NewRC) != NewRC)
        return;
      Register Tmp = MRI->createVirtualRegister(NewRC);
      BuildMI(*MI.getParent(), MI.getIterator(), MI.getDebugLoc(),
              get(AMDGPU::COPY), Tmp)
          .addReg(Reg);
      MI.getOperand(OpNo).setReg(Tmp);
      MI.getOperand(OpNo).setIsKill();
    };

    // Multiplied part is the constant: Use v_madmk_{f16, f32}.
    if ((Src0->isReg() && Src0->getReg() == Reg) ||
        (Src1->isReg() && Src1->getReg() == Reg)) {
      MachineOperand *RegSrc =
          Src1->isReg() && Src1->getReg() == Reg ? Src0 : Src1;
      if (!RegSrc->isReg())
        return false;
      if (RI.isSGPRClass(MRI->getRegClass(RegSrc->getReg())) &&
          ST.getConstantBusLimit(Opc) < 2)
        return false;

      if (!Src2->isReg() || RI.isSGPRClass(MRI->getRegClass(Src2->getReg())))
        return false;

      // If src2 is also a literal constant then we have to choose which one to
      // fold. In general it is better to choose madak so that the other literal
      // can be materialized in an sgpr instead of a vgpr:
      //   s_mov_b32 s0, literal
      //   v_madak_f32 v0, s0, v0, literal
      // Instead of:
      //   v_mov_b32 v1, literal
      //   v_madmk_f32 v0, v0, literal, v1
      MachineInstr *Def = MRI->getUniqueVRegDef(Src2->getReg());
      if (Def && Def->isMoveImmediate() &&
          !isInlineConstant(Def->getOperand(1)))
        return false;

      unsigned NewOpc = getNewFMAMKInst(ST, Opc);
      if (pseudoToMCOpcode(NewOpc) == -1)
        return false;

      const std::optional<int64_t> SubRegImm = extractSubregFromImm(
          Imm, RegSrc == Src1 ? Src0->getSubReg() : Src1->getSubReg());

      // FIXME: This would be a lot easier if we could return a new instruction
      // instead of having to modify in place.

      Register SrcReg = RegSrc->getReg();
      unsigned SrcSubReg = RegSrc->getSubReg();
      Src0->setReg(SrcReg);
      Src0->setSubReg(SrcSubReg);
      Src0->setIsKill(RegSrc->isKill());

      if (Opc == AMDGPU::V_MAC_F32_e64 || Opc == AMDGPU::V_MAC_F16_e64 ||
          Opc == AMDGPU::V_FMAC_F32_e64 || Opc == AMDGPU::V_FMAC_F16_t16_e64 ||
          Opc == AMDGPU::V_FMAC_F16_fake16_e64 ||
          Opc == AMDGPU::V_FMAC_F16_e64 || Opc == AMDGPU::V_FMAC_F64_e64)
        UseMI.untieRegOperand(
            AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2));

      Src1->ChangeToImmediate(*SubRegImm);

      removeModOperands(UseMI);
      UseMI.setDesc(get(NewOpc));

      if (NewOpc == AMDGPU::V_FMAMK_F16_t16 ||
          NewOpc == AMDGPU::V_FMAMK_F16_fake16) {
        const TargetRegisterClass *NewRC = getRegClass(get(NewOpc), 0);
        Register Tmp = MRI->createVirtualRegister(NewRC);
        BuildMI(*UseMI.getParent(), std::next(UseMI.getIterator()),
                UseMI.getDebugLoc(), get(AMDGPU::COPY),
                UseMI.getOperand(0).getReg())
            .addReg(Tmp, RegState::Kill);
        UseMI.getOperand(0).setReg(Tmp);
        CopyRegOperandToNarrowerRC(UseMI, 1, NewRC);
        CopyRegOperandToNarrowerRC(UseMI, 3, NewRC);
      }

      bool DeleteDef = MRI->use_nodbg_empty(Reg);
      if (DeleteDef)
        DefMI.eraseFromParent();

      return true;
    }

    // Added part is the constant: Use v_madak_{f16, f32}.
    if (Src2->isReg() && Src2->getReg() == Reg) {
      if (ST.getConstantBusLimit(Opc) < 2) {
        // Not allowed to use constant bus for another operand.
        // We can however allow an inline immediate as src0.
        bool Src0Inlined = false;
        if (Src0->isReg()) {
          // Try to inline constant if possible.
          // If the Def moves immediate and the use is single
          // We are saving VGPR here.
          MachineInstr *Def = MRI->getUniqueVRegDef(Src0->getReg());
          if (Def && Def->isMoveImmediate() &&
              isInlineConstant(Def->getOperand(1)) &&
              MRI->hasOneNonDBGUse(Src0->getReg())) {
            Src0->ChangeToImmediate(Def->getOperand(1).getImm());
            Src0Inlined = true;
          } else if (ST.getConstantBusLimit(Opc) <= 1 &&
                     RI.isSGPRReg(*MRI, Src0->getReg())) {
            return false;
          }
          // VGPR is okay as Src0 - fallthrough
        }

        if (Src1->isReg() && !Src0Inlined) {
          // We have one slot for inlinable constant so far - try to fill it
          MachineInstr *Def = MRI->getUniqueVRegDef(Src1->getReg());
          if (Def && Def->isMoveImmediate() &&
              isInlineConstant(Def->getOperand(1)) &&
              MRI->hasOneNonDBGUse(Src1->getReg()) && commuteInstruction(UseMI))
            Src0->ChangeToImmediate(Def->getOperand(1).getImm());
          else if (RI.isSGPRReg(*MRI, Src1->getReg()))
            return false;
          // VGPR is okay as Src1 - fallthrough
        }
      }

      unsigned NewOpc = getNewFMAAKInst(ST, Opc);
      if (pseudoToMCOpcode(NewOpc) == -1)
        return false;

      // FIXME: This would be a lot easier if we could return a new instruction
      // instead of having to modify in place.

      if (Opc == AMDGPU::V_MAC_F32_e64 || Opc == AMDGPU::V_MAC_F16_e64 ||
          Opc == AMDGPU::V_FMAC_F32_e64 || Opc == AMDGPU::V_FMAC_F16_t16_e64 ||
          Opc == AMDGPU::V_FMAC_F16_fake16_e64 ||
          Opc == AMDGPU::V_FMAC_F16_e64 || Opc == AMDGPU::V_FMAC_F64_e64)
        UseMI.untieRegOperand(
            AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2));

      const std::optional<int64_t> SubRegImm =
          extractSubregFromImm(Imm, Src2->getSubReg());

      // ChangingToImmediate adds Src2 back to the instruction.
      Src2->ChangeToImmediate(*SubRegImm);

      // These come before src2.
      removeModOperands(UseMI);
      UseMI.setDesc(get(NewOpc));

      if (NewOpc == AMDGPU::V_FMAAK_F16_t16 ||
          NewOpc == AMDGPU::V_FMAAK_F16_fake16) {
        const TargetRegisterClass *NewRC = getRegClass(get(NewOpc), 0);
        Register Tmp = MRI->createVirtualRegister(NewRC);
        BuildMI(*UseMI.getParent(), std::next(UseMI.getIterator()),
                UseMI.getDebugLoc(), get(AMDGPU::COPY),
                UseMI.getOperand(0).getReg())
            .addReg(Tmp, RegState::Kill);
        UseMI.getOperand(0).setReg(Tmp);
        CopyRegOperandToNarrowerRC(UseMI, 1, NewRC);
        CopyRegOperandToNarrowerRC(UseMI, 2, NewRC);
      }

      // It might happen that UseMI was commuted
      // and we now have SGPR as SRC1. If so 2 inlined
      // constant and SGPR are illegal.
      legalizeOperands(UseMI);

      bool DeleteDef = MRI->use_nodbg_empty(Reg);
      if (DeleteDef)
        DefMI.eraseFromParent();

      return true;
    }
  }

  return false;
}

static bool
memOpsHaveSameBaseOperands(ArrayRef<const MachineOperand *> BaseOps1,
                           ArrayRef<const MachineOperand *> BaseOps2) {
  if (BaseOps1.size() != BaseOps2.size())
    return false;
  for (size_t I = 0, E = BaseOps1.size(); I < E; ++I) {
    if (!BaseOps1[I]->isIdenticalTo(*BaseOps2[I]))
      return false;
  }
  return true;
}

static bool offsetsDoNotOverlap(LocationSize WidthA, int OffsetA,
                                LocationSize WidthB, int OffsetB) {
  int LowOffset = OffsetA < OffsetB ? OffsetA : OffsetB;
  int HighOffset = OffsetA < OffsetB ? OffsetB : OffsetA;
  LocationSize LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
  return LowWidth.hasValue() &&
         LowOffset + (int)LowWidth.getValue() <= HighOffset;
}

bool SIInstrInfo::checkInstOffsetsDoNotOverlap(const MachineInstr &MIa,
                                               const MachineInstr &MIb) const {
  SmallVector<const MachineOperand *, 4> BaseOps0, BaseOps1;
  int64_t Offset0, Offset1;
  LocationSize Dummy0 = LocationSize::precise(0);
  LocationSize Dummy1 = LocationSize::precise(0);
  bool Offset0IsScalable, Offset1IsScalable;
  if (!getMemOperandsWithOffsetWidth(MIa, BaseOps0, Offset0, Offset0IsScalable,
                                     Dummy0, &RI) ||
      !getMemOperandsWithOffsetWidth(MIb, BaseOps1, Offset1, Offset1IsScalable,
                                     Dummy1, &RI))
    return false;

  if (!memOpsHaveSameBaseOperands(BaseOps0, BaseOps1))
    return false;

  if (!MIa.hasOneMemOperand() || !MIb.hasOneMemOperand()) {
    // FIXME: Handle ds_read2 / ds_write2.
    return false;
  }
  LocationSize Width0 = MIa.memoperands().front()->getSize();
  LocationSize Width1 = MIb.memoperands().front()->getSize();
  return offsetsDoNotOverlap(Width0, Offset0, Width1, Offset1);
}

bool SIInstrInfo::areMemAccessesTriviallyDisjoint(const MachineInstr &MIa,
                                                  const MachineInstr &MIb) const {
  assert(MIa.mayLoadOrStore() &&
         "MIa must load from or modify a memory location");
  assert(MIb.mayLoadOrStore() &&
         "MIb must load from or modify a memory location");

  if (MIa.hasUnmodeledSideEffects() || MIb.hasUnmodeledSideEffects())
    return false;

  // XXX - Can we relax this between address spaces?
  if (MIa.hasOrderedMemoryRef() || MIb.hasOrderedMemoryRef())
    return false;

  if (isLDSDMA(MIa) || isLDSDMA(MIb))
    return false;

  if (MIa.isBundle() || MIb.isBundle())
    return false;

  // TODO: Should we check the address space from the MachineMemOperand? That
  // would allow us to distinguish objects we know don't alias based on the
  // underlying address space, even if it was lowered to a different one,
  // e.g. private accesses lowered to use MUBUF instructions on a scratch
  // buffer.
  if (isDS(MIa)) {
    if (isDS(MIb))
      return checkInstOffsetsDoNotOverlap(MIa, MIb);

    return !isFLAT(MIb) || isSegmentSpecificFLAT(MIb);
  }

  if (isMUBUF(MIa) || isMTBUF(MIa)) {
    if (isMUBUF(MIb) || isMTBUF(MIb))
      return checkInstOffsetsDoNotOverlap(MIa, MIb);

    if (isFLAT(MIb))
      return isFLATScratch(MIb);

    return !isSMRD(MIb);
  }

  if (isSMRD(MIa)) {
    if (isSMRD(MIb))
      return checkInstOffsetsDoNotOverlap(MIa, MIb);

    if (isFLAT(MIb))
      return isFLATScratch(MIb);

    return !isMUBUF(MIb) && !isMTBUF(MIb);
  }

  if (isFLAT(MIa)) {
    if (isFLAT(MIb)) {
      if ((isFLATScratch(MIa) && isFLATGlobal(MIb)) ||
          (isFLATGlobal(MIa) && isFLATScratch(MIb)))
        return true;

      return checkInstOffsetsDoNotOverlap(MIa, MIb);
    }

    return false;
  }

  return false;
}

static bool getFoldableImm(Register Reg, const MachineRegisterInfo &MRI,
                           int64_t &Imm, MachineInstr **DefMI = nullptr) {
  if (Reg.isPhysical())
    return false;
  auto *Def = MRI.getUniqueVRegDef(Reg);
  if (Def && SIInstrInfo::isFoldableCopy(*Def) && Def->getOperand(1).isImm()) {
    Imm = Def->getOperand(1).getImm();
    if (DefMI)
      *DefMI = Def;
    return true;
  }
  return false;
}

static bool getFoldableImm(const MachineOperand *MO, int64_t &Imm,
                           MachineInstr **DefMI = nullptr) {
  if (!MO->isReg())
    return false;
  const MachineFunction *MF = MO->getParent()->getMF();
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  return getFoldableImm(MO->getReg(), MRI, Imm, DefMI);
}

static void updateLiveVariables(LiveVariables *LV, MachineInstr &MI,
                                MachineInstr &NewMI) {
  if (LV) {
    unsigned NumOps = MI.getNumOperands();
    for (unsigned I = 1; I < NumOps; ++I) {
      MachineOperand &Op = MI.getOperand(I);
      if (Op.isReg() && Op.isKill())
        LV->replaceKillInstruction(Op.getReg(), MI, NewMI);
    }
  }
}

static unsigned getNewFMAInst(const GCNSubtarget &ST, unsigned Opc) {
  switch (Opc) {
  case AMDGPU::V_MAC_F16_e32:
  case AMDGPU::V_MAC_F16_e64:
    return AMDGPU::V_MAD_F16_e64;
  case AMDGPU::V_MAC_F32_e32:
  case AMDGPU::V_MAC_F32_e64:
    return AMDGPU::V_MAD_F32_e64;
  case AMDGPU::V_MAC_LEGACY_F32_e32:
  case AMDGPU::V_MAC_LEGACY_F32_e64:
    return AMDGPU::V_MAD_LEGACY_F32_e64;
  case AMDGPU::V_FMAC_LEGACY_F32_e32:
  case AMDGPU::V_FMAC_LEGACY_F32_e64:
    return AMDGPU::V_FMA_LEGACY_F32_e64;
  case AMDGPU::V_FMAC_F16_e32:
  case AMDGPU::V_FMAC_F16_e64:
  case AMDGPU::V_FMAC_F16_t16_e64:
  case AMDGPU::V_FMAC_F16_fake16_e64:
    return ST.hasTrue16BitInsts() ? ST.useRealTrue16Insts()
                                        ? AMDGPU::V_FMA_F16_gfx9_t16_e64
                                        : AMDGPU::V_FMA_F16_gfx9_fake16_e64
                                  : AMDGPU::V_FMA_F16_gfx9_e64;
  case AMDGPU::V_FMAC_F32_e32:
  case AMDGPU::V_FMAC_F32_e64:
    return AMDGPU::V_FMA_F32_e64;
  case AMDGPU::V_FMAC_F64_e32:
  case AMDGPU::V_FMAC_F64_e64:
    return AMDGPU::V_FMA_F64_e64;
  default:
    llvm_unreachable("invalid instruction");
  }
}

/// Helper struct for the implementation of 3-address conversion to communicate
/// updates made to instruction operands.
struct SIInstrInfo::ThreeAddressUpdates {
  /// Other instruction whose def is no longer used by the converted
  /// instruction.
  MachineInstr *RemoveMIUse = nullptr;
};

MachineInstr *SIInstrInfo::convertToThreeAddress(MachineInstr &MI,
                                                 LiveVariables *LV,
                                                 LiveIntervals *LIS) const {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineInstr *CandidateMI = &MI;

  if (MI.isBundle()) {
    // This is a temporary placeholder for bundle handling that enables us to
    // exercise the relevant code paths in the two-address instruction pass.
    if (MI.getBundleSize() != 1)
      return nullptr;
    CandidateMI = MI.getNextNode();
  }

  ThreeAddressUpdates U;
  MachineInstr *NewMI = convertToThreeAddressImpl(*CandidateMI, U);
  if (!NewMI)
    return nullptr;

  if (MI.isBundle()) {
    CandidateMI->eraseFromBundle();

    for (MachineOperand &MO : MI.all_defs()) {
      if (MO.isTied())
        MI.untieRegOperand(MO.getOperandNo());
    }
  } else {
    updateLiveVariables(LV, MI, *NewMI);
    if (LIS) {
      LIS->ReplaceMachineInstrInMaps(MI, *NewMI);
      // SlotIndex of defs needs to be updated when converting to early-clobber
      MachineOperand &Def = NewMI->getOperand(0);
      if (Def.isEarlyClobber() && Def.isReg() &&
          LIS->hasInterval(Def.getReg())) {
        SlotIndex OldIndex = LIS->getInstructionIndex(*NewMI).getRegSlot(false);
        SlotIndex NewIndex = LIS->getInstructionIndex(*NewMI).getRegSlot(true);
        auto &LI = LIS->getInterval(Def.getReg());
        auto UpdateDefIndex = [&](LiveRange &LR) {
          auto *S = LR.find(OldIndex);
          if (S != LR.end() && S->start == OldIndex) {
            assert(S->valno && S->valno->def == OldIndex);
            S->start = NewIndex;
            S->valno->def = NewIndex;
          }
        };
        UpdateDefIndex(LI);
        for (auto &SR : LI.subranges())
          UpdateDefIndex(SR);
      }
    }
  }

  if (U.RemoveMIUse) {
    MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
    // The only user is the instruction which will be killed.
    Register DefReg = U.RemoveMIUse->getOperand(0).getReg();

    if (MRI.hasOneNonDBGUse(DefReg)) {
      // We cannot just remove the DefMI here, calling pass will crash.
      U.RemoveMIUse->setDesc(get(AMDGPU::IMPLICIT_DEF));
      U.RemoveMIUse->getOperand(0).setIsDead(true);
      for (unsigned I = U.RemoveMIUse->getNumOperands() - 1; I != 0; --I)
        U.RemoveMIUse->removeOperand(I);
      if (LV)
        LV->getVarInfo(DefReg).AliveBlocks.clear();
    }

    if (MI.isBundle()) {
      VirtRegInfo VRI = AnalyzeVirtRegInBundle(MI, DefReg);
      if (!VRI.Reads && !VRI.Writes) {
        for (MachineOperand &MO : MI.all_uses()) {
          if (MO.isReg() && MO.getReg() == DefReg) {
            assert(MO.getSubReg() == 0 &&
                   "tied sub-registers in bundles currently not supported");
            MI.removeOperand(MO.getOperandNo());
            break;
          }
        }

        if (LIS)
          LIS->shrinkToUses(&LIS->getInterval(DefReg));
      }
    } else if (LIS) {
      LiveInterval &DefLI = LIS->getInterval(DefReg);

      // We cannot delete the original instruction here, so hack out the use
      // in the original instruction with a dummy register so we can use
      // shrinkToUses to deal with any multi-use edge cases. Other targets do
      // not have the complexity of deleting a use to consider here.
      Register DummyReg = MRI.cloneVirtualRegister(DefReg);
      for (MachineOperand &MIOp : MI.uses()) {
        if (MIOp.isReg() && MIOp.getReg() == DefReg) {
          MIOp.setIsUndef(true);
          MIOp.setReg(DummyReg);
        }
      }

      if (MI.isBundle()) {
        VirtRegInfo VRI = AnalyzeVirtRegInBundle(MI, DefReg);
        if (!VRI.Reads && !VRI.Writes) {
          for (MachineOperand &MIOp : MI.uses()) {
            if (MIOp.isReg() && MIOp.getReg() == DefReg) {
              MIOp.setIsUndef(true);
              MIOp.setReg(DummyReg);
            }
          }
        }

        MI.addOperand(MachineOperand::CreateReg(DummyReg, false, false, false,
                                                false, /*isUndef=*/true));
      }

      LIS->shrinkToUses(&DefLI);
    }
  }

  return MI.isBundle() ? &MI : NewMI;
}

MachineInstr *
SIInstrInfo::convertToThreeAddressImpl(MachineInstr &MI,
                                       ThreeAddressUpdates &U) const {
  MachineBasicBlock &MBB = *MI.getParent();
  unsigned Opc = MI.getOpcode();

  // Handle MFMA.
  int NewMFMAOpc = AMDGPU::getMFMAEarlyClobberOp(Opc);
  if (NewMFMAOpc != -1) {
    MachineInstrBuilder MIB =
        BuildMI(MBB, MI, MI.getDebugLoc(), get(NewMFMAOpc));
    for (unsigned I = 0, E = MI.getNumExplicitOperands(); I != E; ++I)
      MIB.add(MI.getOperand(I));
    return MIB;
  }

  if (SIInstrInfo::isWMMA(MI)) {
    unsigned NewOpc = AMDGPU::mapWMMA2AddrTo3AddrOpcode(MI.getOpcode());
    MachineInstrBuilder MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
                                  .setMIFlags(MI.getFlags());
    for (unsigned I = 0, E = MI.getNumExplicitOperands(); I != E; ++I)
      MIB->addOperand(MI.getOperand(I));
    return MIB;
  }

  assert(Opc != AMDGPU::V_FMAC_F16_t16_e32 &&
         Opc != AMDGPU::V_FMAC_F16_fake16_e32 &&
         "V_FMAC_F16_t16/fake16_e32 is not supported and not expected to be "
         "present pre-RA");

  // Handle MAC/FMAC.
  bool IsF64 = Opc == AMDGPU::V_FMAC_F64_e32 || Opc == AMDGPU::V_FMAC_F64_e64;
  bool IsLegacy = Opc == AMDGPU::V_MAC_LEGACY_F32_e32 ||
                  Opc == AMDGPU::V_MAC_LEGACY_F32_e64 ||
                  Opc == AMDGPU::V_FMAC_LEGACY_F32_e32 ||
                  Opc == AMDGPU::V_FMAC_LEGACY_F32_e64;
  bool Src0Literal = false;

  switch (Opc) {
  default:
    return nullptr;
  case AMDGPU::V_MAC_F16_e64:
  case AMDGPU::V_FMAC_F16_e64:
  case AMDGPU::V_FMAC_F16_t16_e64:
  case AMDGPU::V_FMAC_F16_fake16_e64:
  case AMDGPU::V_MAC_F32_e64:
  case AMDGPU::V_MAC_LEGACY_F32_e64:
  case AMDGPU::V_FMAC_F32_e64:
  case AMDGPU::V_FMAC_LEGACY_F32_e64:
  case AMDGPU::V_FMAC_F64_e64:
    break;
  case AMDGPU::V_MAC_F16_e32:
  case AMDGPU::V_FMAC_F16_e32:
  case AMDGPU::V_MAC_F32_e32:
  case AMDGPU::V_MAC_LEGACY_F32_e32:
  case AMDGPU::V_FMAC_F32_e32:
  case AMDGPU::V_FMAC_LEGACY_F32_e32:
  case AMDGPU::V_FMAC_F64_e32: {
    int Src0Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                             AMDGPU::OpName::src0);
    const MachineOperand *Src0 = &MI.getOperand(Src0Idx);
    if (!Src0->isReg() && !Src0->isImm())
      return nullptr;

    if (Src0->isImm() && !isInlineConstant(MI, Src0Idx, *Src0))
      Src0Literal = true;

    break;
  }
  }

  MachineInstrBuilder MIB;
  const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::vdst);
  const MachineOperand *Src0 = getNamedOperand(MI, AMDGPU::OpName::src0);
  const MachineOperand *Src0Mods =
    getNamedOperand(MI, AMDGPU::OpName::src0_modifiers);
  const MachineOperand *Src1 = getNamedOperand(MI, AMDGPU::OpName::src1);
  const MachineOperand *Src1Mods =
    getNamedOperand(MI, AMDGPU::OpName::src1_modifiers);
  const MachineOperand *Src2 = getNamedOperand(MI, AMDGPU::OpName::src2);
  const MachineOperand *Src2Mods =
      getNamedOperand(MI, AMDGPU::OpName::src2_modifiers);
  const MachineOperand *Clamp = getNamedOperand(MI, AMDGPU::OpName::clamp);
  const MachineOperand *Omod = getNamedOperand(MI, AMDGPU::OpName::omod);
  const MachineOperand *OpSel = getNamedOperand(MI, AMDGPU::OpName::op_sel);

  if (!Src0Mods && !Src1Mods && !Src2Mods && !Clamp && !Omod && !IsLegacy &&
      (!IsF64 || ST.hasFmaakFmamkF64Insts()) &&
      // If we have an SGPR input, we will violate the constant bus restriction.
      (ST.getConstantBusLimit(Opc) > 1 || !Src0->isReg() ||
       !RI.isSGPRReg(MBB.getParent()->getRegInfo(), Src0->getReg()))) {
    MachineInstr *DefMI;

    int64_t Imm;
    if (!Src0Literal && getFoldableImm(Src2, Imm, &DefMI)) {
      unsigned NewOpc = getNewFMAAKInst(ST, Opc);
      if (pseudoToMCOpcode(NewOpc) != -1) {
        MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
                  .add(*Dst)
                  .add(*Src0)
                  .add(*Src1)
                  .addImm(Imm)
                  .setMIFlags(MI.getFlags());
        U.RemoveMIUse = DefMI;
        return MIB;
      }
    }
    unsigned NewOpc = getNewFMAMKInst(ST, Opc);
    if (!Src0Literal && getFoldableImm(Src1, Imm, &DefMI)) {
      if (pseudoToMCOpcode(NewOpc) != -1) {
        MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
                  .add(*Dst)
                  .add(*Src0)
                  .addImm(Imm)
                  .add(*Src2)
                  .setMIFlags(MI.getFlags());
        U.RemoveMIUse = DefMI;
        return MIB;
      }
    }
    if (Src0Literal || getFoldableImm(Src0, Imm, &DefMI)) {
      if (Src0Literal) {
        Imm = Src0->getImm();
        DefMI = nullptr;
      }
      if (pseudoToMCOpcode(NewOpc) != -1 &&
          isOperandLegal(
              MI, AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::src0),
              Src1)) {
        MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
                  .add(*Dst)
                  .add(*Src1)
                  .addImm(Imm)
                  .add(*Src2)
                  .setMIFlags(MI.getFlags());
        U.RemoveMIUse = DefMI;
        return MIB;
      }
    }
  }

  // VOP2 mac/fmac with a literal operand cannot be converted to VOP3 mad/fma
  // if VOP3 does not allow a literal operand.
  if (Src0Literal && !ST.hasVOP3Literal())
    return nullptr;

  unsigned NewOpc = getNewFMAInst(ST, Opc);

  if (pseudoToMCOpcode(NewOpc) == -1)
    return nullptr;

  MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
            .add(*Dst)
            .addImm(Src0Mods ? Src0Mods->getImm() : 0)
            .add(*Src0)
            .addImm(Src1Mods ? Src1Mods->getImm() : 0)
            .add(*Src1)
            .addImm(Src2Mods ? Src2Mods->getImm() : 0)
            .add(*Src2)
            .addImm(Clamp ? Clamp->getImm() : 0)
            .addImm(Omod ? Omod->getImm() : 0)
            .setMIFlags(MI.getFlags());
  if (AMDGPU::hasNamedOperand(NewOpc, AMDGPU::OpName::op_sel))
    MIB.addImm(OpSel ? OpSel->getImm() : 0);
  return MIB;
}

// It's not generally safe to move VALU instructions across these since it will
// start using the register as a base index rather than directly.
// XXX - Why isn't hasSideEffects sufficient for these?
static bool changesVGPRIndexingMode(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::S_SET_GPR_IDX_ON:
  case AMDGPU::S_SET_GPR_IDX_MODE:
  case AMDGPU::S_SET_GPR_IDX_OFF:
    return true;
  default:
    return false;
  }
}

bool SIInstrInfo::isSchedulingBoundary(const MachineInstr &MI,
                                       const MachineBasicBlock *MBB,
                                       const MachineFunction &MF) const {
  // Skipping the check for SP writes in the base implementation. The reason it
  // was added was apparently due to compile time concerns.
  //
  // TODO: Do we really want this barrier? It triggers unnecessary hazard nops
  // but is probably avoidable.

  // Copied from base implementation.
  // Terminators and labels can't be scheduled around.
  if (MI.isTerminator() || MI.isPosition())
    return true;

  // INLINEASM_BR can jump to another block
  if (MI.getOpcode() == TargetOpcode::INLINEASM_BR)
    return true;

  if (MI.getOpcode() == AMDGPU::SCHED_BARRIER && MI.getOperand(0).getImm() == 0)
    return true;

  // Target-independent instructions do not have an implicit-use of EXEC, even
  // when they operate on VGPRs. Treating EXEC modifications as scheduling
  // boundaries prevents incorrect movements of such instructions.
  return MI.modifiesRegister(AMDGPU::EXEC, &RI) ||
         MI.getOpcode() == AMDGPU::S_SETREG_IMM32_B32 ||
         MI.getOpcode() == AMDGPU::S_SETREG_B32 ||
         MI.getOpcode() == AMDGPU::S_SETPRIO ||
         MI.getOpcode() == AMDGPU::S_SETPRIO_INC_WG ||
         changesVGPRIndexingMode(MI);
}

bool SIInstrInfo::isAlwaysGDS(uint32_t Opcode) const {
  return Opcode == AMDGPU::DS_ORDERED_COUNT ||
         Opcode == AMDGPU::DS_ADD_GS_REG_RTN ||
         Opcode == AMDGPU::DS_SUB_GS_REG_RTN || isGWS(Opcode);
}

bool SIInstrInfo::mayAccessScratch(const MachineInstr &MI) const {
  // Instructions that access scratch use FLAT encoding or BUF encodings.
  if ((!isFLAT(MI) || isFLATGlobal(MI)) && !isBUF(MI))
    return false;

  // SCRATCH instructions always access scratch.
  if (isFLATScratch(MI))
    return true;

  // If FLAT_SCRATCH registers are not initialized, we can never access scratch
  // via the aperture.
  if (MI.getMF()->getFunction().hasFnAttribute("amdgpu-no-flat-scratch-init"))
    return false;

  // If there are no memory operands then conservatively assume the flat
  // operation may access scratch.
  if (MI.memoperands_empty())
    return true;

  // See if any memory operand specifies an address space that involves scratch.
  return any_of(MI.memoperands(), [](const MachineMemOperand *Memop) {
    unsigned AS = Memop->getAddrSpace();
    if (AS == AMDGPUAS::FLAT_ADDRESS) {
      const MDNode *MD = Memop->getAAInfo().NoAliasAddrSpace;
      return !MD || !AMDGPU::hasValueInRangeLikeMetadata(
                        *MD, AMDGPUAS::PRIVATE_ADDRESS);
    }
    return AS == AMDGPUAS::PRIVATE_ADDRESS;
  });
}

bool SIInstrInfo::mayAccessVMEMThroughFlat(const MachineInstr &MI) const {
  assert(isFLAT(MI));

  // All flat instructions use the VMEM counter except prefetch.
  if (!usesVM_CNT(MI))
    return false;

  // If there are no memory operands then conservatively assume the flat
  // operation may access VMEM.
  if (MI.memoperands_empty())
    return true;

  // See if any memory operand specifies an address space that involves VMEM.
  // Flat operations only supported FLAT, LOCAL (LDS), or address spaces
  // involving VMEM such as GLOBAL, CONSTANT, PRIVATE (SCRATCH), etc. The REGION
  // (GDS) address space is not supported by flat operations. Therefore, simply
  // return true unless only the LDS address space is found.
  for (const MachineMemOperand *Memop : MI.memoperands()) {
    unsigned AS = Memop->getAddrSpace();
    assert(AS != AMDGPUAS::REGION_ADDRESS);
    if (AS != AMDGPUAS::LOCAL_ADDRESS)
      return true;
  }

  return false;
}

bool SIInstrInfo::mayAccessLDSThroughFlat(const MachineInstr &MI) const {
  assert(isFLAT(MI));

  // Flat instruction such as SCRATCH and GLOBAL do not use the lgkm counter.
  if (!usesLGKM_CNT(MI))
    return false;

  // If in tgsplit mode then there can be no use of LDS.
  if (ST.isTgSplitEnabled())
    return false;

  // If there are no memory operands then conservatively assume the flat
  // operation may access LDS.
  if (MI.memoperands_empty())
    return true;

  // See if any memory operand specifies an address space that involves LDS.
  for (const MachineMemOperand *Memop : MI.memoperands()) {
    unsigned AS = Memop->getAddrSpace();
    if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS)
      return true;
  }

  return false;
}

bool SIInstrInfo::modifiesModeRegister(const MachineInstr &MI) {
  // Skip the full operand and register alias search modifiesRegister
  // does. There's only a handful of instructions that touch this, it's only an
  // implicit def, and doesn't alias any other registers.
  return is_contained(MI.getDesc().implicit_defs(), AMDGPU::MODE);
}

bool SIInstrInfo::hasUnwantedEffectsWhenEXECEmpty(const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();

  if (MI.mayStore() && isSMRD(MI))
    return true; // scalar store or atomic

  // This will terminate the function when other lanes may need to continue.
  if (MI.isReturn())
    return true;

  // These instructions cause shader I/O that may cause hardware lockups
  // when executed with an empty EXEC mask.
  //
  // Note: exp with VM = DONE = 0 is automatically skipped by hardware when
  //       EXEC = 0, but checking for that case here seems not worth it
  //       given the typical code patterns.
  if (Opcode == AMDGPU::S_SENDMSG || Opcode == AMDGPU::S_SENDMSGHALT ||
      isEXP(Opcode) || Opcode == AMDGPU::DS_ORDERED_COUNT ||
      Opcode == AMDGPU::S_TRAP || Opcode == AMDGPU::S_WAIT_EVENT)
    return true;

  if (MI.isCall() || MI.isInlineAsm())
    return true; // conservative assumption

  // Assume that barrier interactions are only intended with active lanes.
  if (isBarrier(Opcode))
    return true;

  // A mode change is a scalar operation that influences vector instructions.
  if (modifiesModeRegister(MI))
    return true;

  // These are like SALU instructions in terms of effects, so it's questionable
  // whether we should return true for those.
  //
  // However, executing them with EXEC = 0 causes them to operate on undefined
  // data, which we avoid by returning true here.
  if (Opcode == AMDGPU::V_READFIRSTLANE_B32 ||
      Opcode == AMDGPU::V_READLANE_B32 || Opcode == AMDGPU::V_WRITELANE_B32 ||
      Opcode == AMDGPU::SI_RESTORE_S32_FROM_VGPR ||
      Opcode == AMDGPU::SI_SPILL_S32_TO_VGPR)
    return true;

  return false;
}

bool SIInstrInfo::mayReadEXEC(const MachineRegisterInfo &MRI,
                              const MachineInstr &MI) const {
  if (MI.isMetaInstruction())
    return false;

  // This won't read exec if this is an SGPR->SGPR copy.
  if (MI.isCopyLike()) {
    if (!RI.isSGPRReg(MRI, MI.getOperand(0).getReg()))
      return true;

    // Make sure this isn't copying exec as a normal operand
    return MI.readsRegister(AMDGPU::EXEC, &RI);
  }

  // Make a conservative assumption about the callee.
  if (MI.isCall())
    return true;

  // Be conservative with any unhandled generic opcodes.
  if (!isTargetSpecificOpcode(MI.getOpcode()))
    return true;

  return !isSALU(MI) || MI.readsRegister(AMDGPU::EXEC, &RI);
}

bool SIInstrInfo::isInlineConstant(const APInt &Imm) const {
  switch (Imm.getBitWidth()) {
  case 1: // This likely will be a condition code mask.
    return true;

  case 32:
    return AMDGPU::isInlinableLiteral32(Imm.getSExtValue(),
                                        ST.hasInv2PiInlineImm());
  case 64:
    return AMDGPU::isInlinableLiteral64(Imm.getSExtValue(),
                                        ST.hasInv2PiInlineImm());
  case 16:
    return ST.has16BitInsts() &&
           AMDGPU::isInlinableLiteralI16(Imm.getSExtValue(),
                                         ST.hasInv2PiInlineImm());
  default:
    llvm_unreachable("invalid bitwidth");
  }
}

bool SIInstrInfo::isInlineConstant(const APFloat &Imm) const {
  APInt IntImm = Imm.bitcastToAPInt();
  int64_t IntImmVal = IntImm.getSExtValue();
  bool HasInv2Pi = ST.hasInv2PiInlineImm();
  switch (APFloat::SemanticsToEnum(Imm.getSemantics())) {
  default:
    llvm_unreachable("invalid fltSemantics");
  case APFloatBase::S_IEEEsingle:
  case APFloatBase::S_IEEEdouble:
    return isInlineConstant(IntImm);
  case APFloatBase::S_BFloat:
    return ST.has16BitInsts() &&
           AMDGPU::isInlinableLiteralBF16(IntImmVal, HasInv2Pi);
  case APFloatBase::S_IEEEhalf:
    return ST.has16BitInsts() &&
           AMDGPU::isInlinableLiteralFP16(IntImmVal, HasInv2Pi);
  }
}

bool SIInstrInfo::isInlineConstant(int64_t Imm, uint8_t OperandType) const {
  // MachineOperand provides no way to tell the true operand size, since it only
  // records a 64-bit value. We need to know the size to determine if a 32-bit
  // floating point immediate bit pattern is legal for an integer immediate. It
  // would be for any 32-bit integer operand, but would not be for a 64-bit one.
  switch (OperandType) {
  case AMDGPU::OPERAND_REG_IMM_INT32:
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_INT32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
  case AMDGPU::OPERAND_REG_IMM_V2FP32:
  case AMDGPU::OPERAND_REG_IMM_V2INT32:
  case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
  case AMDGPU::OPERAND_INLINE_SPLIT_BARRIER_INT32: {
    int32_t Trunc = static_cast<int32_t>(Imm);
    return AMDGPU::isInlinableLiteral32(Trunc, ST.hasInv2PiInlineImm());
  }
  case AMDGPU::OPERAND_REG_IMM_I64:
  case AMDGPU::OPERAND_REG_IMM_U64:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_INT64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP64:
    return AMDGPU::isInlinableLiteral64(Imm, ST.hasInv2PiInlineImm());
  case AMDGPU::OPERAND_REG_IMM_INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_INT16:
    // We would expect inline immediates to not be concerned with an integer/fp
    // distinction. However, in the case of 16-bit integer operations, the
    // "floating point" values appear to not work. It seems read the low 16-bits
    // of 32-bit immediates, which happens to always work for the integer
    // values.
    //
    // See llvm bugzilla 46302.
    //
    // TODO: Theoretically we could use op-sel to use the high bits of the
    // 32-bit FP values.
    return AMDGPU::isInlinableIntLiteral(Imm);
  case AMDGPU::OPERAND_REG_IMM_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
    return AMDGPU::isInlinableLiteralV2I16(Imm);
  case AMDGPU::OPERAND_REG_IMM_V2FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    return AMDGPU::isInlinableLiteralV2F16(Imm);
  case AMDGPU::OPERAND_REG_IMM_V2FP16_SPLAT:
    return AMDGPU::isPKFMACF16InlineConstant(Imm, ST.isGFX11Plus());
  case AMDGPU::OPERAND_REG_IMM_V2BF16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2BF16:
    return AMDGPU::isInlinableLiteralV2BF16(Imm);
  case AMDGPU::OPERAND_REG_IMM_NOINLINE_V2FP16:
    return false;
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16: {
    if (isInt<16>(Imm) || isUInt<16>(Imm)) {
      // A few special case instructions have 16-bit operands on subtargets
      // where 16-bit instructions are not legal.
      // TODO: Do the 32-bit immediates work? We shouldn't really need to handle
      // constants in these cases
      int16_t Trunc = static_cast<int16_t>(Imm);
      return ST.has16BitInsts() &&
             AMDGPU::isInlinableLiteralFP16(Trunc, ST.hasInv2PiInlineImm());
    }

    return false;
  }
  case AMDGPU::OPERAND_REG_IMM_BF16:
  case AMDGPU::OPERAND_REG_INLINE_C_BF16: {
    if (isInt<16>(Imm) || isUInt<16>(Imm)) {
      int16_t Trunc = static_cast<int16_t>(Imm);
      return ST.has16BitInsts() &&
             AMDGPU::isInlinableLiteralBF16(Trunc, ST.hasInv2PiInlineImm());
    }
    return false;
  }
  case AMDGPU::OPERAND_KIMM32:
  case AMDGPU::OPERAND_KIMM16:
  case AMDGPU::OPERAND_KIMM64:
    return false;
  case AMDGPU::OPERAND_INLINE_C_AV64_PSEUDO:
    return isLegalAV64PseudoImm(Imm);
  case AMDGPU::OPERAND_INPUT_MODS:
  case MCOI::OPERAND_IMMEDIATE:
    // Always embedded in the instruction for free.
    return true;
  case MCOI::OPERAND_UNKNOWN:
  case MCOI::OPERAND_REGISTER:
  case MCOI::OPERAND_PCREL:
  case MCOI::OPERAND_GENERIC_0:
  case MCOI::OPERAND_GENERIC_1:
  case MCOI::OPERAND_GENERIC_2:
  case MCOI::OPERAND_GENERIC_3:
  case MCOI::OPERAND_GENERIC_4:
  case MCOI::OPERAND_GENERIC_5:
    // Just ignore anything else.
    return true;
  default:
    llvm_unreachable("invalid operand type");
  }
}

static bool compareMachineOp(const MachineOperand &Op0,
                             const MachineOperand &Op1) {
  if (Op0.getType() != Op1.getType())
    return false;

  switch (Op0.getType()) {
  case MachineOperand::MO_Register:
    return Op0.getReg() == Op1.getReg();
  case MachineOperand::MO_Immediate:
    return Op0.getImm() == Op1.getImm();
  default:
    llvm_unreachable("Didn't expect to be comparing these operand types");
  }
}

bool SIInstrInfo::isLiteralOperandLegal(const MCInstrDesc &InstDesc,
                                        const MCOperandInfo &OpInfo) const {
  if (OpInfo.OperandType == MCOI::OPERAND_IMMEDIATE)
    return true;

  if (!RI.opCanUseLiteralConstant(OpInfo.OperandType))
    return false;

  if (!isVOP3(InstDesc) || !AMDGPU::isSISrcOperand(OpInfo))
    return true;

  return ST.hasVOP3Literal();
}

bool SIInstrInfo::isImmOperandLegal(const MCInstrDesc &InstDesc, unsigned OpNo,
                                    int64_t ImmVal) const {
  const MCOperandInfo &OpInfo = InstDesc.operands()[OpNo];
  if (isInlineConstant(ImmVal, OpInfo.OperandType)) {
    if (isMAI(InstDesc) && ST.hasMFMAInlineLiteralBug() &&
        OpNo == (unsigned)AMDGPU::getNamedOperandIdx(InstDesc.getOpcode(),
                                                     AMDGPU::OpName::src2))
      return false;
    return RI.opCanUseInlineConstant(OpInfo.OperandType);
  }

  return isLiteralOperandLegal(InstDesc, OpInfo);
}

bool SIInstrInfo::isImmOperandLegal(const MCInstrDesc &InstDesc, unsigned OpNo,
                                    const MachineOperand &MO) const {
  if (MO.isImm())
    return isImmOperandLegal(InstDesc, OpNo, MO.getImm());

  assert((MO.isTargetIndex() || MO.isFI() || MO.isGlobal()) &&
         "unexpected imm-like operand kind");
  const MCOperandInfo &OpInfo = InstDesc.operands()[OpNo];
  return isLiteralOperandLegal(InstDesc, OpInfo);
}

bool SIInstrInfo::isLegalAV64PseudoImm(uint64_t Imm) const {
  // 2 32-bit inline constants packed into one.
  return AMDGPU::isInlinableLiteral32(Lo_32(Imm), ST.hasInv2PiInlineImm()) &&
         AMDGPU::isInlinableLiteral32(Hi_32(Imm), ST.hasInv2PiInlineImm());
}

bool SIInstrInfo::hasVALU32BitEncoding(unsigned Opcode) const {
  // GFX90A does not have V_MUL_LEGACY_F32_e32.
  if (Opcode == AMDGPU::V_MUL_LEGACY_F32_e64 && ST.hasGFX90AInsts())
    return false;

  int Op32 = AMDGPU::getVOPe32(Opcode);
  if (Op32 == -1)
    return false;

  return pseudoToMCOpcode(Op32) != -1;
}

bool SIInstrInfo::hasModifiers(unsigned Opcode) const {
  // The src0_modifier operand is present on all instructions
  // that have modifiers.

  return AMDGPU::hasNamedOperand(Opcode, AMDGPU::OpName::src0_modifiers);
}

bool SIInstrInfo::hasModifiersSet(const MachineInstr &MI,
                                  AMDGPU::OpName OpName) const {
  const MachineOperand *Mods = getNamedOperand(MI, OpName);
  return Mods && Mods->getImm();
}

bool SIInstrInfo::hasAnyModifiersSet(const MachineInstr &MI) const {
  return any_of(ModifierOpNames,
                [&](AMDGPU::OpName Name) { return hasModifiersSet(MI, Name); });
}

bool SIInstrInfo::canShrink(const MachineInstr &MI,
                            const MachineRegisterInfo &MRI) const {
  const MachineOperand *Src2 = getNamedOperand(MI, AMDGPU::OpName::src2);
  // Can't shrink instruction with three operands.
  if (Src2) {
    switch (MI.getOpcode()) {
      default: return false;

      case AMDGPU::V_ADDC_U32_e64:
      case AMDGPU::V_SUBB_U32_e64:
      case AMDGPU::V_SUBBREV_U32_e64: {
        const MachineOperand *Src1
          = getNamedOperand(MI, AMDGPU::OpName::src1);
        if (!Src1->isReg() || !RI.isVGPR(MRI, Src1->getReg()))
          return false;
        // Additional verification is needed for sdst/src2.
        return true;
      }
      case AMDGPU::V_MAC_F16_e64:
      case AMDGPU::V_MAC_F32_e64:
      case AMDGPU::V_MAC_LEGACY_F32_e64:
      case AMDGPU::V_FMAC_F16_e64:
      case AMDGPU::V_FMAC_F16_t16_e64:
      case AMDGPU::V_FMAC_F16_fake16_e64:
      case AMDGPU::V_FMAC_F32_e64:
      case AMDGPU::V_FMAC_F64_e64:
      case AMDGPU::V_FMAC_LEGACY_F32_e64:
        if (!Src2->isReg() || !RI.isVGPR(MRI, Src2->getReg()) ||
            hasModifiersSet(MI, AMDGPU::OpName::src2_modifiers))
          return false;
        break;

      case AMDGPU::V_CNDMASK_B32_e64:
        break;
    }
  }

  const MachineOperand *Src1 = getNamedOperand(MI, AMDGPU::OpName::src1);
  if (Src1 && (!Src1->isReg() || !RI.isVGPR(MRI, Src1->getReg()) ||
               hasModifiersSet(MI, AMDGPU::OpName::src1_modifiers)))
    return false;

  // We don't need to check src0, all input types are legal, so just make sure
  // src0 isn't using any modifiers.
  if (hasModifiersSet(MI, AMDGPU::OpName::src0_modifiers))
    return false;

  // Can it be shrunk to a valid 32 bit opcode?
  if (!hasVALU32BitEncoding(MI.getOpcode()))
    return false;

  // Check output modifiers
  return !hasModifiersSet(MI, AMDGPU::OpName::omod) &&
         !hasModifiersSet(MI, AMDGPU::OpName::clamp) &&
         !hasModifiersSet(MI, AMDGPU::OpName::byte_sel) &&
         // TODO: Can we avoid checking bound_ctrl/fi here?
         // They are only used by permlane*_swap special case.
         !hasModifiersSet(MI, AMDGPU::OpName::bound_ctrl) &&
         !hasModifiersSet(MI, AMDGPU::OpName::fi);
}

// Set VCC operand with all flags from \p Orig, except for setting it as
// implicit.
static void copyFlagsToImplicitVCC(MachineInstr &MI,
                                   const MachineOperand &Orig) {

  for (MachineOperand &Use : MI.implicit_operands()) {
    if (Use.isUse() &&
        (Use.getReg() == AMDGPU::VCC || Use.getReg() == AMDGPU::VCC_LO)) {
      Use.setIsUndef(Orig.isUndef());
      Use.setIsKill(Orig.isKill());
      return;
    }
  }
}

MachineInstr *SIInstrInfo::buildShrunkInst(MachineInstr &MI,
                                           unsigned Op32) const {
  MachineBasicBlock *MBB = MI.getParent();

  const MCInstrDesc &Op32Desc = get(Op32);
  MachineInstrBuilder Inst32 =
    BuildMI(*MBB, MI, MI.getDebugLoc(), Op32Desc)
    .setMIFlags(MI.getFlags());

  // Add the dst operand if the 32-bit encoding also has an explicit $vdst.
  // For VOPC instructions, this is replaced by an implicit def of vcc.

  // We assume the defs of the shrunk opcode are in the same order, and the
  // shrunk opcode loses the last def (SGPR def, in the VOP3->VOPC case).
  for (int I = 0, E = Op32Desc.getNumDefs(); I != E; ++I)
    Inst32.add(MI.getOperand(I));

  const MachineOperand *Src2 = getNamedOperand(MI, AMDGPU::OpName::src2);

  int Idx = MI.getNumExplicitDefs();
  for (const MachineOperand &Use : MI.explicit_uses()) {
    int OpTy = MI.getDesc().operands()[Idx++].OperandType;
    if (OpTy == AMDGPU::OPERAND_INPUT_MODS || OpTy == MCOI::OPERAND_IMMEDIATE)
      continue;

    if (&Use == Src2) {
      if (AMDGPU::getNamedOperandIdx(Op32, AMDGPU::OpName::src2) == -1) {
        // In the case of V_CNDMASK_B32_e32, the explicit operand src2 is
        // replaced with an implicit read of vcc or vcc_lo. The implicit read
        // of vcc was already added during the initial BuildMI, but we
        // 1) may need to change vcc to vcc_lo to preserve the original register
        // 2) have to preserve the original flags.
        copyFlagsToImplicitVCC(*Inst32, *Src2);
        continue;
      }
    }

    Inst32.add(Use);
  }

  // FIXME: Losing implicit operands
  fixImplicitOperands(*Inst32);
  return Inst32;
}

bool SIInstrInfo::physRegUsesConstantBus(const MachineOperand &RegOp) const {
  // Null is free
  Register Reg = RegOp.getReg();
  if (Reg == AMDGPU::SGPR_NULL || Reg == AMDGPU::SGPR_NULL64)
    return false;

  // SGPRs use the constant bus

  // FIXME: implicit registers that are not part of the MCInstrDesc's implicit
  // physical register operands should also count, except for exec.
  if (RegOp.isImplicit())
    return Reg == AMDGPU::VCC || Reg == AMDGPU::VCC_LO || Reg == AMDGPU::M0;

  // SGPRs use the constant bus
  return AMDGPU::SReg_32RegClass.contains(Reg) ||
         AMDGPU::SReg_64RegClass.contains(Reg);
}

bool SIInstrInfo::regUsesConstantBus(const MachineOperand &RegOp,
                                     const MachineRegisterInfo &MRI) const {
  Register Reg = RegOp.getReg();
  return Reg.isVirtual() ? RI.isSGPRClass(MRI.getRegClass(Reg))
                         : physRegUsesConstantBus(RegOp);
}

bool SIInstrInfo::usesConstantBus(const MachineRegisterInfo &MRI,
                                  const MachineOperand &MO,
                                  const MCOperandInfo &OpInfo) const {
  // Literal constants use the constant bus.
  if (!MO.isReg())
    return !isInlineConstant(MO, OpInfo);

  Register Reg = MO.getReg();
  return Reg.isVirtual() ? RI.isSGPRClass(MRI.getRegClass(Reg))
                         : physRegUsesConstantBus(MO);
}

static Register findImplicitSGPRRead(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.implicit_operands()) {
    // We only care about reads.
    if (MO.isDef())
      continue;

    switch (MO.getReg()) {
    case AMDGPU::VCC:
    case AMDGPU::VCC_LO:
    case AMDGPU::VCC_HI:
    case AMDGPU::M0:
    case AMDGPU::FLAT_SCR:
      return MO.getReg();

    default:
      break;
    }
  }

  return Register();
}

static bool shouldReadExec(const MachineInstr &MI) {
  if (SIInstrInfo::isVALU(MI)) {
    switch (MI.getOpcode()) {
    case AMDGPU::V_READLANE_B32:
    case AMDGPU::SI_RESTORE_S32_FROM_VGPR:
    case AMDGPU::V_WRITELANE_B32:
    case AMDGPU::SI_SPILL_S32_TO_VGPR:
      return false;
    }

    return true;
  }

  if (MI.isPreISelOpcode() ||
      SIInstrInfo::isGenericOpcode(MI.getOpcode()) ||
      SIInstrInfo::isSALU(MI) ||
      SIInstrInfo::isSMRD(MI))
    return false;

  return true;
}

static bool isRegOrFI(const MachineOperand &MO) {
  return MO.isReg() || MO.isFI();
}

static bool isSubRegOf(const SIRegisterInfo &TRI,
                       const MachineOperand &SuperVec,
                       const MachineOperand &SubReg) {
  if (SubReg.getReg().isPhysical())
    return TRI.isSubRegister(SuperVec.getReg(), SubReg.getReg());

  return SubReg.getSubReg() != AMDGPU::NoSubRegister &&
         SubReg.getReg() == SuperVec.getReg();
}

// Verify the illegal copy from vector register to SGPR for generic opcode COPY
bool SIInstrInfo::verifyCopy(const MachineInstr &MI,
                             const MachineRegisterInfo &MRI,
                             StringRef &ErrInfo) const {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  // This is a check for copy from vector register to SGPR
  if (RI.isVectorRegister(MRI, SrcReg) && RI.isSGPRReg(MRI, DstReg)) {
    ErrInfo = "illegal copy from vector register to SGPR";
    return false;
  }
  return true;
}

bool SIInstrInfo::verifyInstruction(const MachineInstr &MI,
                                    StringRef &ErrInfo) const {
  uint32_t Opcode = MI.getOpcode();
  const MachineFunction *MF = MI.getMF();
  const MachineRegisterInfo &MRI = MF->getRegInfo();

  // FIXME: At this point the COPY verify is done only for non-ssa forms.
  // Find a better property to recognize the point where instruction selection
  // is just done.
  // We can only enforce this check after SIFixSGPRCopies pass so that the
  // illegal copies are legalized and thereafter we don't expect a pass
  // inserting similar copies.
  if (!MRI.isSSA() && MI.isCopy())
    return verifyCopy(MI, MRI, ErrInfo);

  if (SIInstrInfo::isGenericOpcode(Opcode))
    return true;

  int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
  int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);
  int Src3Idx = -1;
  if (Src0Idx == -1) {
    // VOPD V_DUAL_* instructions use different operand names.
    Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0X);
    Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::vsrc1X);
    Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0Y);
    Src3Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::vsrc1Y);
  }

  // Make sure the number of operands is correct.
  const MCInstrDesc &Desc = get(Opcode);
  if (!Desc.isVariadic() &&
      Desc.getNumOperands() != MI.getNumExplicitOperands()) {
    ErrInfo = "Instruction has wrong number of operands.";
    return false;
  }

  if (MI.isInlineAsm()) {
    // Verify register classes for inlineasm constraints.
    for (unsigned I = InlineAsm::MIOp_FirstOperand, E = MI.getNumOperands();
         I != E; ++I) {
      const TargetRegisterClass *RC = MI.getRegClassConstraint(I, this, &RI);
      if (!RC)
        continue;

      const MachineOperand &Op = MI.getOperand(I);
      if (!Op.isReg())
        continue;

      Register Reg = Op.getReg();
      if (!Reg.isVirtual() && !RC->contains(Reg)) {
        ErrInfo = "inlineasm operand has incorrect register class.";
        return false;
      }
    }

    return true;
  }

  if (isImage(MI) && MI.memoperands_empty() && MI.mayLoadOrStore()) {
    ErrInfo = "missing memory operand from image instruction.";
    return false;
  }

  // Make sure the register classes are correct.
  for (int i = 0, e = Desc.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (MO.isFPImm()) {
      ErrInfo = "FPImm Machine Operands are not supported. ISel should bitcast "
                "all fp values to integers.";
      return false;
    }

    const MCOperandInfo &OpInfo = Desc.operands()[i];
    int16_t RegClass = getOpRegClassID(OpInfo);

    switch (OpInfo.OperandType) {
    case MCOI::OPERAND_REGISTER:
      if (MI.getOperand(i).isImm() || MI.getOperand(i).isGlobal()) {
        ErrInfo = "Illegal immediate value for operand.";
        return false;
      }
      break;
    case AMDGPU::OPERAND_REG_IMM_INT32:
    case AMDGPU::OPERAND_REG_IMM_I64:
    case AMDGPU::OPERAND_REG_IMM_U64:
    case AMDGPU::OPERAND_REG_IMM_INT16:
    case AMDGPU::OPERAND_REG_IMM_FP32:
    case AMDGPU::OPERAND_REG_IMM_V2FP32:
    case AMDGPU::OPERAND_REG_IMM_BF16:
    case AMDGPU::OPERAND_REG_IMM_FP16:
    case AMDGPU::OPERAND_REG_IMM_FP64:
    case AMDGPU::OPERAND_REG_IMM_V2FP16:
    case AMDGPU::OPERAND_REG_IMM_V2FP16_SPLAT:
    case AMDGPU::OPERAND_REG_IMM_V2INT16:
    case AMDGPU::OPERAND_REG_IMM_V2INT32:
    case AMDGPU::OPERAND_REG_IMM_V2BF16:
      break;
    case AMDGPU::OPERAND_REG_IMM_NOINLINE_V2FP16:
      break;
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_INT16:
    case AMDGPU::OPERAND_REG_INLINE_C_INT32:
    case AMDGPU::OPERAND_REG_INLINE_C_INT64:
    case AMDGPU::OPERAND_REG_INLINE_C_FP32:
    case AMDGPU::OPERAND_REG_INLINE_C_FP64:
    case AMDGPU::OPERAND_REG_INLINE_C_BF16:
    case AMDGPU::OPERAND_REG_INLINE_C_FP16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2BF16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP64: {
      if (!MO.isReg() && (!MO.isImm() || !isInlineConstant(MI, i))) {
        ErrInfo = "Illegal immediate value for operand.";
        return false;
      }
      break;
    }
    case AMDGPU::OPERAND_INLINE_SPLIT_BARRIER_INT32:
      if (!MI.getOperand(i).isImm() || !isInlineConstant(MI, i)) {
        ErrInfo = "Expected inline constant for operand.";
        return false;
      }
      break;
    case AMDGPU::OPERAND_INPUT_MODS:
    case AMDGPU::OPERAND_SDWA_VOPC_DST:
    case AMDGPU::OPERAND_KIMM16:
      break;
    case MCOI::OPERAND_IMMEDIATE:
    case AMDGPU::OPERAND_KIMM32:
    case AMDGPU::OPERAND_KIMM64:
    case AMDGPU::OPERAND_INLINE_C_AV64_PSEUDO:
      // Check if this operand is an immediate.
      // FrameIndex operands will be replaced by immediates, so they are
      // allowed.
      if (!MI.getOperand(i).isImm() && !MI.getOperand(i).isFI()) {
        ErrInfo = "Expected immediate, but got non-immediate";
        return false;
      }
      break;
    case MCOI::OPERAND_UNKNOWN:
    case MCOI::OPERAND_MEMORY:
    case MCOI::OPERAND_PCREL:
      break;
    default:
      if (OpInfo.isGenericType())
        continue;
      break;
    }

    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (!Reg)
      continue;

    // FIXME: Ideally we would have separate instruction definitions with the
    // aligned register constraint.
    // FIXME: We do not verify inline asm operands, but custom inline asm
    // verification is broken anyway
    if (ST.needsAlignedVGPRs() && Opcode != AMDGPU::AV_MOV_B64_IMM_PSEUDO &&
        Opcode != AMDGPU::V_MOV_B64_PSEUDO && !isSpill(MI)) {
      const TargetRegisterClass *RC = RI.getRegClassForReg(MRI, Reg);
      if (RI.hasVectorRegisters(RC) && MO.getSubReg()) {
        if (const TargetRegisterClass *SubRC =
                RI.getSubRegisterClass(RC, MO.getSubReg())) {
          RC = RI.getCompatibleSubRegClass(RC, SubRC, MO.getSubReg());
          if (RC)
            RC = SubRC;
        }
      }

      // Check that this is the aligned version of the class.
      if (!RC || !RI.isProperlyAlignedRC(*RC)) {
        ErrInfo = "Subtarget requires even aligned vector registers";
        return false;
      }
    }

    if (RegClass != -1) {
      if (Reg.isVirtual())
        continue;

      const TargetRegisterClass *RC = RI.getRegClass(RegClass);
      if (!RC->contains(Reg)) {
        ErrInfo = "Operand has incorrect register class.";
        return false;
      }
    }
  }

  // Verify SDWA
  if (isSDWA(MI)) {
    if (!ST.hasSDWA()) {
      ErrInfo = "SDWA is not supported on this target";
      return false;
    }

    for (auto Op : {AMDGPU::OpName::src0_sel, AMDGPU::OpName::src1_sel,
                    AMDGPU::OpName::dst_sel}) {
      const MachineOperand *MO = getNamedOperand(MI, Op);
      if (!MO)
        continue;
      int64_t Imm = MO->getImm();
      if (Imm < 0 || Imm > AMDGPU::SDWA::SdwaSel::DWORD) {
        ErrInfo = "Invalid SDWA selection";
        return false;
      }
    }

    int DstIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::vdst);

    for (int OpIdx : {DstIdx, Src0Idx, Src1Idx, Src2Idx}) {
      if (OpIdx == -1)
        continue;
      const MachineOperand &MO = MI.getOperand(OpIdx);

      if (!ST.hasSDWAScalar()) {
        // Only VGPRS on VI
        if (!MO.isReg() || !RI.hasVGPRs(RI.getRegClassForReg(MRI, MO.getReg()))) {
          ErrInfo = "Only VGPRs allowed as operands in SDWA instructions on VI";
          return false;
        }
      } else {
        // No immediates on GFX9
        if (!MO.isReg()) {
          ErrInfo =
            "Only reg allowed as operands in SDWA instructions on GFX9+";
          return false;
        }
      }
    }

    if (!ST.hasSDWAOmod()) {
      // No omod allowed on VI
      const MachineOperand *OMod = getNamedOperand(MI, AMDGPU::OpName::omod);
      if (OMod != nullptr &&
        (!OMod->isImm() || OMod->getImm() != 0)) {
        ErrInfo = "OMod not allowed in SDWA instructions on VI";
        return false;
      }
    }

    if (Opcode == AMDGPU::V_CVT_F32_FP8_sdwa ||
        Opcode == AMDGPU::V_CVT_F32_BF8_sdwa ||
        Opcode == AMDGPU::V_CVT_PK_F32_FP8_sdwa ||
        Opcode == AMDGPU::V_CVT_PK_F32_BF8_sdwa) {
      const MachineOperand *Src0ModsMO =
          getNamedOperand(MI, AMDGPU::OpName::src0_modifiers);
      unsigned Mods = Src0ModsMO->getImm();
      if (Mods & SISrcMods::ABS || Mods & SISrcMods::NEG ||
          Mods & SISrcMods::SEXT) {
        ErrInfo = "sext, abs and neg are not allowed on this instruction";
        return false;
      }
    }

    uint32_t BasicOpcode = AMDGPU::getBasicFromSDWAOp(Opcode);
    if (isVOPC(BasicOpcode)) {
      if (!ST.hasSDWASdst() && DstIdx != -1) {
        // Only vcc allowed as dst on VI for VOPC
        const MachineOperand &Dst = MI.getOperand(DstIdx);
        if (!Dst.isReg() || Dst.getReg() != AMDGPU::VCC) {
          ErrInfo = "Only VCC allowed as dst in SDWA instructions on VI";
          return false;
        }
      } else if (!ST.hasSDWAOutModsVOPC()) {
        // No clamp allowed on GFX9 for VOPC
        const MachineOperand *Clamp = getNamedOperand(MI, AMDGPU::OpName::clamp);
        if (Clamp && (!Clamp->isImm() || Clamp->getImm() != 0)) {
          ErrInfo = "Clamp not allowed in VOPC SDWA instructions on VI";
          return false;
        }

        // No omod allowed on GFX9 for VOPC
        const MachineOperand *OMod = getNamedOperand(MI, AMDGPU::OpName::omod);
        if (OMod && (!OMod->isImm() || OMod->getImm() != 0)) {
          ErrInfo = "OMod not allowed in VOPC SDWA instructions on VI";
          return false;
        }
      }
    }

    const MachineOperand *DstUnused = getNamedOperand(MI, AMDGPU::OpName::dst_unused);
    if (DstUnused && DstUnused->isImm() &&
        DstUnused->getImm() == AMDGPU::SDWA::UNUSED_PRESERVE) {
      const MachineOperand &Dst = MI.getOperand(DstIdx);
      if (!Dst.isReg() || !Dst.isTied()) {
        ErrInfo = "Dst register should have tied register";
        return false;
      }

      const MachineOperand &TiedMO =
          MI.getOperand(MI.findTiedOperandIdx(DstIdx));
      if (!TiedMO.isReg() || !TiedMO.isImplicit() || !TiedMO.isUse()) {
        ErrInfo =
            "Dst register should be tied to implicit use of preserved register";
        return false;
      }
      if (TiedMO.getReg().isPhysical() && Dst.getReg() != TiedMO.getReg()) {
        ErrInfo = "Dst register should use same physical register as preserved";
        return false;
      }
    }
  }

  // Verify MIMG / VIMAGE / VSAMPLE
  if (isImage(Opcode) && !MI.mayStore()) {
    // Ensure that the return type used is large enough for all the options
    // being used TFE/LWE require an extra result register.
    const MachineOperand *DMask = getNamedOperand(MI, AMDGPU::OpName::dmask);
    if (DMask) {
      uint64_t DMaskImm = DMask->getImm();
      uint32_t RegCount = isGather4(Opcode) ? 4 : llvm::popcount(DMaskImm);
      const MachineOperand *TFE = getNamedOperand(MI, AMDGPU::OpName::tfe);
      const MachineOperand *LWE = getNamedOperand(MI, AMDGPU::OpName::lwe);
      const MachineOperand *D16 = getNamedOperand(MI, AMDGPU::OpName::d16);

      // Adjust for packed 16 bit values
      if (D16 && D16->getImm() && !ST.hasUnpackedD16VMem())
        RegCount = divideCeil(RegCount, 2);

      // Adjust if using LWE or TFE
      if ((LWE && LWE->getImm()) || (TFE && TFE->getImm()))
        RegCount += 1;

      const uint32_t DstIdx =
          AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::vdata);
      const MachineOperand &Dst = MI.getOperand(DstIdx);
      if (Dst.isReg()) {
        const TargetRegisterClass *DstRC = getOpRegClass(MI, DstIdx);
        uint32_t DstSize = RI.getRegSizeInBits(*DstRC) / 32;
        if (RegCount > DstSize) {
          ErrInfo = "Image instruction returns too many registers for dst "
                    "register class";
          return false;
        }
      }
    }
  }

  // Verify VOP*. Ignore multiple sgpr operands on writelane.
  if (isVALU(MI) && Desc.getOpcode() != AMDGPU::V_WRITELANE_B32) {
    unsigned ConstantBusCount = 0;
    bool UsesLiteral = false;
    const MachineOperand *LiteralVal = nullptr;

    int ImmIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::imm);
    if (ImmIdx != -1) {
      ++ConstantBusCount;
      UsesLiteral = true;
      LiteralVal = &MI.getOperand(ImmIdx);
    }

    SmallVector<Register, 2> SGPRsUsed;
    Register SGPRUsed;

    // Only look at the true operands. Only a real operand can use the constant
    // bus, and we don't want to check pseudo-operands like the source modifier
    // flags.
    for (int OpIdx : {Src0Idx, Src1Idx, Src2Idx, Src3Idx}) {
      if (OpIdx == -1)
        continue;
      const MachineOperand &MO = MI.getOperand(OpIdx);
      if (usesConstantBus(MRI, MO, MI.getDesc().operands()[OpIdx])) {
        if (MO.isReg()) {
          SGPRUsed = MO.getReg();
          if (!llvm::is_contained(SGPRsUsed, SGPRUsed)) {
            ++ConstantBusCount;
            SGPRsUsed.push_back(SGPRUsed);
          }
        } else if (!MO.isFI()) { // Treat FI like a register.
          if (!UsesLiteral) {
            ++ConstantBusCount;
            UsesLiteral = true;
            LiteralVal = &MO;
          } else if (!MO.isIdenticalTo(*LiteralVal)) {
            assert(isVOP2(MI) || isVOP3(MI));
            ErrInfo = "VOP2/VOP3 instruction uses more than one literal";
            return false;
          }
        }
      }
    }

    SGPRUsed = findImplicitSGPRRead(MI);
    if (SGPRUsed) {
      // Implicit uses may safely overlap true operands
      if (llvm::all_of(SGPRsUsed, [this, SGPRUsed](unsigned SGPR) {
            return !RI.regsOverlap(SGPRUsed, SGPR);
          })) {
        ++ConstantBusCount;
        SGPRsUsed.push_back(SGPRUsed);
      }
    }

    // v_writelane_b32 is an exception from constant bus restriction:
    // vsrc0 can be sgpr, const or m0 and lane select sgpr, m0 or inline-const
    if (ConstantBusCount > ST.getConstantBusLimit(Opcode) &&
        Opcode != AMDGPU::V_WRITELANE_B32) {
      ErrInfo = "VOP* instruction violates constant bus restriction";
      return false;
    }

    if (isVOP3(MI) && UsesLiteral && !ST.hasVOP3Literal()) {
      ErrInfo = "VOP3 instruction uses literal";
      return false;
    }
  }

  // Special case for writelane - this can break the multiple constant bus rule,
  // but still can't use more than one SGPR register
  if (Desc.getOpcode() == AMDGPU::V_WRITELANE_B32) {
    unsigned SGPRCount = 0;
    Register SGPRUsed;

    for (int OpIdx : {Src0Idx, Src1Idx}) {
      if (OpIdx == -1)
        break;

      const MachineOperand &MO = MI.getOperand(OpIdx);

      if (usesConstantBus(MRI, MO, MI.getDesc().operands()[OpIdx])) {
        if (MO.isReg() && MO.getReg() != AMDGPU::M0) {
          if (MO.getReg() != SGPRUsed)
            ++SGPRCount;
          SGPRUsed = MO.getReg();
        }
      }
      if (SGPRCount > ST.getConstantBusLimit(Opcode)) {
        ErrInfo = "WRITELANE instruction violates constant bus restriction";
        return false;
      }
    }
  }

  // Verify misc. restrictions on specific instructions.
  if (Desc.getOpcode() == AMDGPU::V_DIV_SCALE_F32_e64 ||
      Desc.getOpcode() == AMDGPU::V_DIV_SCALE_F64_e64) {
    const MachineOperand &Src0 = MI.getOperand(Src0Idx);
    const MachineOperand &Src1 = MI.getOperand(Src1Idx);
    const MachineOperand &Src2 = MI.getOperand(Src2Idx);
    if (Src0.isReg() && Src1.isReg() && Src2.isReg()) {
      if (!compareMachineOp(Src0, Src1) &&
          !compareMachineOp(Src0, Src2)) {
        ErrInfo = "v_div_scale_{f32|f64} require src0 = src1 or src2";
        return false;
      }
    }
    if ((getNamedOperand(MI, AMDGPU::OpName::src0_modifiers)->getImm() &
         SISrcMods::ABS) ||
        (getNamedOperand(MI, AMDGPU::OpName::src1_modifiers)->getImm() &
         SISrcMods::ABS) ||
        (getNamedOperand(MI, AMDGPU::OpName::src2_modifiers)->getImm() &
         SISrcMods::ABS)) {
      ErrInfo = "ABS not allowed in VOP3B instructions";
      return false;
    }
  }

  if (isSOP2(MI) || isSOPC(MI)) {
    const MachineOperand &Src0 = MI.getOperand(Src0Idx);
    const MachineOperand &Src1 = MI.getOperand(Src1Idx);

    if (!isRegOrFI(Src0) && !isRegOrFI(Src1) &&
        !isInlineConstant(Src0, Desc.operands()[Src0Idx]) &&
        !isInlineConstant(Src1, Desc.operands()[Src1Idx]) &&
        !Src0.isIdenticalTo(Src1)) {
      ErrInfo = "SOP2/SOPC instruction requires too many immediate constants";
      return false;
    }
  }

  if (isSOPK(MI)) {
    const auto *Op = getNamedOperand(MI, AMDGPU::OpName::simm16);
    if (Desc.isBranch()) {
      if (!Op->isMBB()) {
        ErrInfo = "invalid branch target for SOPK instruction";
        return false;
      }
    } else {
      uint64_t Imm = Op->getImm();
      if (sopkIsZext(Opcode)) {
        if (!isUInt<16>(Imm)) {
          ErrInfo = "invalid immediate for SOPK instruction";
          return false;
        }
      } else {
        if (!isInt<16>(Imm)) {
          ErrInfo = "invalid immediate for SOPK instruction";
          return false;
        }
      }
    }
  }

  if (Desc.getOpcode() == AMDGPU::V_MOVRELS_B32_e32 ||
      Desc.getOpcode() == AMDGPU::V_MOVRELS_B32_e64 ||
      Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e32 ||
      Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e64) {
    const bool IsDst = Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e32 ||
                       Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e64;

    const unsigned StaticNumOps =
        Desc.getNumOperands() + Desc.implicit_uses().size();
    const unsigned NumImplicitOps = IsDst ? 2 : 1;

    // Require additional implicit operands. This allows a fixup done by the
    // post RA scheduler where the main implicit operand is killed and
    // implicit-defs are added for sub-registers that remain live after this
    // instruction.
    if (MI.getNumOperands() < StaticNumOps + NumImplicitOps) {
      ErrInfo = "missing implicit register operands";
      return false;
    }

    const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::vdst);
    if (IsDst) {
      if (!Dst->isUse()) {
        ErrInfo = "v_movreld_b32 vdst should be a use operand";
        return false;
      }

      unsigned UseOpIdx;
      if (!MI.isRegTiedToUseOperand(StaticNumOps, &UseOpIdx) ||
          UseOpIdx != StaticNumOps + 1) {
        ErrInfo = "movrel implicit operands should be tied";
        return false;
      }
    }

    const MachineOperand &Src0 = MI.getOperand(Src0Idx);
    const MachineOperand &ImpUse
      = MI.getOperand(StaticNumOps + NumImplicitOps - 1);
    if (!ImpUse.isReg() || !ImpUse.isUse() ||
        !isSubRegOf(RI, ImpUse, IsDst ? *Dst : Src0)) {
      ErrInfo = "src0 should be subreg of implicit vector use";
      return false;
    }
  }

  // Make sure we aren't losing exec uses in the td files. This mostly requires
  // being careful when using let Uses to try to add other use registers.
  if (shouldReadExec(MI)) {
    if (!MI.hasRegisterImplicitUseOperand(AMDGPU::EXEC)) {
      ErrInfo = "VALU instruction does not implicitly read exec mask";
      return false;
    }
  }

  if (isSMRD(MI)) {
    if (MI.mayStore() &&
        ST.getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS) {
      // The register offset form of scalar stores may only use m0 as the
      // soffset register.
      const MachineOperand *Soff = getNamedOperand(MI, AMDGPU::OpName::soffset);
      if (Soff && Soff->getReg() != AMDGPU::M0) {
        ErrInfo = "scalar stores must use m0 as offset register";
        return false;
      }
    }
  }

  if (isFLAT(MI) && !ST.hasFlatInstOffsets()) {
    const MachineOperand *Offset = getNamedOperand(MI, AMDGPU::OpName::offset);
    if (Offset->getImm() != 0) {
      ErrInfo = "subtarget does not support offsets in flat instructions";
      return false;
    }
  }

  if (isDS(MI) && !ST.hasGDS()) {
    const MachineOperand *GDSOp = getNamedOperand(MI, AMDGPU::OpName::gds);
    if (GDSOp && GDSOp->getImm() != 0) {
      ErrInfo = "GDS is not supported on this subtarget";
      return false;
    }
  }

  if (isImage(MI)) {
    const MachineOperand *DimOp = getNamedOperand(MI, AMDGPU::OpName::dim);
    if (DimOp) {
      int VAddr0Idx = AMDGPU::getNamedOperandIdx(Opcode,
                                                 AMDGPU::OpName::vaddr0);
      AMDGPU::OpName RSrcOpName =
          isMIMG(MI) ? AMDGPU::OpName::srsrc : AMDGPU::OpName::rsrc;
      int RsrcIdx = AMDGPU::getNamedOperandIdx(Opcode, RSrcOpName);
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Opcode);
      const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode =
          AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
      const AMDGPU::MIMGDimInfo *Dim =
          AMDGPU::getMIMGDimInfoByEncoding(DimOp->getImm());

      if (!Dim) {
        ErrInfo = "dim is out of range";
        return false;
      }

      bool IsA16 = false;
      if (ST.hasR128A16()) {
        const MachineOperand *R128A16 = getNamedOperand(MI, AMDGPU::OpName::r128);
        IsA16 = R128A16->getImm() != 0;
      } else if (ST.hasA16()) {
        const MachineOperand *A16 = getNamedOperand(MI, AMDGPU::OpName::a16);
        IsA16 = A16->getImm() != 0;
      }

      bool IsNSA = RsrcIdx - VAddr0Idx > 1;

      unsigned AddrWords =
          AMDGPU::getAddrSizeMIMGOp(BaseOpcode, Dim, IsA16, ST.hasG16());

      unsigned VAddrWords;
      if (IsNSA) {
        VAddrWords = RsrcIdx - VAddr0Idx;
        if (ST.hasPartialNSAEncoding() &&
            AddrWords > ST.getNSAMaxSize(isVSAMPLE(MI))) {
          unsigned LastVAddrIdx = RsrcIdx - 1;
          VAddrWords += getOpSize(MI, LastVAddrIdx) / 4 - 1;
        }
      } else {
        VAddrWords = getOpSize(MI, VAddr0Idx) / 4;
        if (AddrWords > 12)
          AddrWords = 16;
      }

      if (VAddrWords != AddrWords) {
        LLVM_DEBUG(dbgs() << "bad vaddr size, expected " << AddrWords
                          << " but got " << VAddrWords << "\n");
        ErrInfo = "bad vaddr size";
        return false;
      }
    }
  }

  const MachineOperand *DppCt = getNamedOperand(MI, AMDGPU::OpName::dpp_ctrl);
  if (DppCt) {
    using namespace AMDGPU::DPP;

    unsigned DC = DppCt->getImm();
    if (DC == DppCtrl::DPP_UNUSED1 || DC == DppCtrl::DPP_UNUSED2 ||
        DC == DppCtrl::DPP_UNUSED3 || DC > DppCtrl::DPP_LAST ||
        (DC >= DppCtrl::DPP_UNUSED4_FIRST && DC <= DppCtrl::DPP_UNUSED4_LAST) ||
        (DC >= DppCtrl::DPP_UNUSED5_FIRST && DC <= DppCtrl::DPP_UNUSED5_LAST) ||
        (DC >= DppCtrl::DPP_UNUSED6_FIRST && DC <= DppCtrl::DPP_UNUSED6_LAST) ||
        (DC >= DppCtrl::DPP_UNUSED7_FIRST && DC <= DppCtrl::DPP_UNUSED7_LAST) ||
        (DC >= DppCtrl::DPP_UNUSED8_FIRST && DC <= DppCtrl::DPP_UNUSED8_LAST)) {
      ErrInfo = "Invalid dpp_ctrl value";
      return false;
    }
    if (DC >= DppCtrl::WAVE_SHL1 && DC <= DppCtrl::WAVE_ROR1 &&
        !ST.hasDPPWavefrontShifts()) {
      ErrInfo = "Invalid dpp_ctrl value: "
                "wavefront shifts are not supported on GFX10+";
      return false;
    }
    if (DC >= DppCtrl::BCAST15 && DC <= DppCtrl::BCAST31 &&
        !ST.hasDPPBroadcasts()) {
      ErrInfo = "Invalid dpp_ctrl value: "
                "broadcasts are not supported on GFX10+";
      return false;
    }
    if (DC >= DppCtrl::ROW_SHARE_FIRST && DC <= DppCtrl::ROW_XMASK_LAST &&
        ST.getGeneration() < AMDGPUSubtarget::GFX10) {
      if (DC >= DppCtrl::ROW_NEWBCAST_FIRST &&
          DC <= DppCtrl::ROW_NEWBCAST_LAST &&
          !ST.hasGFX90AInsts()) {
        ErrInfo = "Invalid dpp_ctrl value: "
                  "row_newbroadcast/row_share is not supported before "
                  "GFX90A/GFX10";
        return false;
      }
      if (DC > DppCtrl::ROW_NEWBCAST_LAST || !ST.hasGFX90AInsts()) {
        ErrInfo = "Invalid dpp_ctrl value: "
                  "row_share and row_xmask are not supported before GFX10";
        return false;
      }
    }

    if (Opcode != AMDGPU::V_MOV_B64_DPP_PSEUDO &&
        !AMDGPU::isLegalDPALU_DPPControl(ST, DC) &&
        AMDGPU::isDPALU_DPP(Desc, *this, ST)) {
      ErrInfo = "Invalid dpp_ctrl value: "
                "DP ALU dpp only support row_newbcast";
      return false;
    }
  }

  if ((MI.mayStore() || MI.mayLoad()) && !isVGPRSpill(MI)) {
    const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::vdst);
    AMDGPU::OpName DataName =
        isDS(Opcode) ? AMDGPU::OpName::data0 : AMDGPU::OpName::vdata;
    const MachineOperand *Data = getNamedOperand(MI, DataName);
    const MachineOperand *Data2 = getNamedOperand(MI, AMDGPU::OpName::data1);
    if (Data && !Data->isReg())
      Data = nullptr;

    if (ST.hasGFX90AInsts()) {
      if (Dst && Data && !Dst->isTied() && !Data->isTied() &&
          (RI.isAGPR(MRI, Dst->getReg()) != RI.isAGPR(MRI, Data->getReg()))) {
        ErrInfo = "Invalid register class: "
                  "vdata and vdst should be both VGPR or AGPR";
        return false;
      }
      if (Data && Data2 &&
          (RI.isAGPR(MRI, Data->getReg()) != RI.isAGPR(MRI, Data2->getReg()))) {
        ErrInfo = "Invalid register class: "
                  "both data operands should be VGPR or AGPR";
        return false;
      }
    } else {
      if ((Dst && RI.isAGPR(MRI, Dst->getReg())) ||
          (Data && RI.isAGPR(MRI, Data->getReg())) ||
          (Data2 && RI.isAGPR(MRI, Data2->getReg()))) {
        ErrInfo = "Invalid register class: "
                  "agpr loads and stores not supported on this GPU";
        return false;
      }
    }
  }

  if (ST.needsAlignedVGPRs()) {
    const auto isAlignedReg = [&MI, &MRI, this](AMDGPU::OpName OpName) -> bool {
      const MachineOperand *Op = getNamedOperand(MI, OpName);
      if (!Op)
        return true;
      Register Reg = Op->getReg();
      if (Reg.isPhysical())
        return !(RI.getHWRegIndex(Reg) & 1);
      const TargetRegisterClass &RC = *MRI.getRegClass(Reg);
      return RI.getRegSizeInBits(RC) > 32 && RI.isProperlyAlignedRC(RC) &&
             !(RI.getChannelFromSubReg(Op->getSubReg()) & 1);
    };

    if (Opcode == AMDGPU::DS_GWS_INIT || Opcode == AMDGPU::DS_GWS_SEMA_BR ||
        Opcode == AMDGPU::DS_GWS_BARRIER) {

      if (!isAlignedReg(AMDGPU::OpName::data0)) {
        ErrInfo = "Subtarget requires even aligned vector registers "
                  "for DS_GWS instructions";
        return false;
      }
    }

    if (isMIMG(MI)) {
      if (!isAlignedReg(AMDGPU::OpName::vaddr)) {
        ErrInfo = "Subtarget requires even aligned vector registers "
                  "for vaddr operand of image instructions";
        return false;
      }
    }
  }

  if (Opcode == AMDGPU::V_ACCVGPR_WRITE_B32_e64 && !ST.hasGFX90AInsts()) {
    const MachineOperand *Src = getNamedOperand(MI, AMDGPU::OpName::src0);
    if (Src->isReg() && RI.isSGPRReg(MRI, Src->getReg())) {
      ErrInfo = "Invalid register class: "
                "v_accvgpr_write with an SGPR is not supported on this GPU";
      return false;
    }
  }

  if (Desc.getOpcode() == AMDGPU::G_AMDGPU_WAVE_ADDRESS) {
    const MachineOperand &SrcOp = MI.getOperand(1);
    if (!SrcOp.isReg() || SrcOp.getReg().isVirtual()) {
      ErrInfo = "pseudo expects only physical SGPRs";
      return false;
    }
  }

  if (const MachineOperand *CPol = getNamedOperand(MI, AMDGPU::OpName::cpol)) {
    if (CPol->getImm() & AMDGPU::CPol::SCAL) {
      if (!ST.hasScaleOffset()) {
        ErrInfo = "Subtarget does not support offset scaling";
        return false;
      }
      if (!AMDGPU::supportsScaleOffset(*this, MI.getOpcode())) {
        ErrInfo = "Instruction does not support offset scaling";
        return false;
      }
    }
  }

  // See SIInstrInfo::isLegalGFX12PlusPackedMathFP32Operand for more
  // information.
  if (AMDGPU::isPackedFP32Inst(Opcode) && AMDGPU::isGFX12Plus(ST)) {
    for (unsigned I = 0; I < 3; ++I) {
      if (!isLegalGFX12PlusPackedMathFP32Operand(MRI, MI, I))
        return false;
    }
  }

  if (ST.hasFlatScratchHiInB64InstHazard() && isSALU(MI) &&
      MI.readsRegister(AMDGPU::SRC_FLAT_SCRATCH_BASE_HI, nullptr)) {
    const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::sdst);
    if ((Dst && RI.getRegClassForReg(MRI, Dst->getReg()) ==
                    &AMDGPU::SReg_64RegClass) ||
        Opcode == AMDGPU::S_BITCMP0_B64 || Opcode == AMDGPU::S_BITCMP1_B64) {
      ErrInfo = "Instruction cannot read flat_scratch_base_hi";
      return false;
    }
  }

  return true;
}

// It is more readable to list mapped opcodes on the same line.
// clang-format off

unsigned SIInstrInfo::getVALUOp(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default: return AMDGPU::INSTRUCTION_LIST_END;
  case AMDGPU::REG_SEQUENCE: return AMDGPU::REG_SEQUENCE;
  case AMDGPU::COPY: return AMDGPU::COPY;
  case AMDGPU::PHI: return AMDGPU::PHI;
  case AMDGPU::INSERT_SUBREG: return AMDGPU::INSERT_SUBREG;
  case AMDGPU::WQM: return AMDGPU::WQM;
  case AMDGPU::SOFT_WQM: return AMDGPU::SOFT_WQM;
  case AMDGPU::STRICT_WWM: return AMDGPU::STRICT_WWM;
  case AMDGPU::STRICT_WQM: return AMDGPU::STRICT_WQM;
  case AMDGPU::S_MOV_B32: {
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    return MI.getOperand(1).isReg() ||
           RI.isAGPR(MRI, MI.getOperand(0).getReg()) ?
           AMDGPU::COPY : AMDGPU::V_MOV_B32_e32;
  }
  case AMDGPU::S_ADD_I32:
    return ST.hasAddNoCarryInsts() ? AMDGPU::V_ADD_U32_e64 : AMDGPU::V_ADD_CO_U32_e32;
  case AMDGPU::S_ADDC_U32:
    return AMDGPU::V_ADDC_U32_e32;
  case AMDGPU::S_SUB_I32:
    return ST.hasAddNoCarryInsts() ? AMDGPU::V_SUB_U32_e64 : AMDGPU::V_SUB_CO_U32_e32;
    // FIXME: These are not consistently handled, and selected when the carry is
    // used.
  case AMDGPU::S_ADD_U32:
    return AMDGPU::V_ADD_CO_U32_e32;
  case AMDGPU::S_SUB_U32:
    return AMDGPU::V_SUB_CO_U32_e32;
  case AMDGPU::S_ADD_U64_PSEUDO:
    return AMDGPU::V_ADD_U64_PSEUDO;
  case AMDGPU::S_SUB_U64_PSEUDO:
    return AMDGPU::V_SUB_U64_PSEUDO;
  case AMDGPU::S_SUBB_U32: return AMDGPU::V_SUBB_U32_e32;
  case AMDGPU::S_MUL_I32: return AMDGPU::V_MUL_LO_U32_e64;
  case AMDGPU::S_MUL_HI_U32: return AMDGPU::V_MUL_HI_U32_e64;
  case AMDGPU::S_MUL_HI_I32: return AMDGPU::V_MUL_HI_I32_e64;
  case AMDGPU::S_AND_B32: return AMDGPU::V_AND_B32_e64;
  case AMDGPU::S_OR_B32: return AMDGPU::V_OR_B32_e64;
  case AMDGPU::S_XOR_B32: return AMDGPU::V_XOR_B32_e64;
  case AMDGPU::S_XNOR_B32:
    return ST.hasDLInsts() ? AMDGPU::V_XNOR_B32_e64 : AMDGPU::INSTRUCTION_LIST_END;
  case AMDGPU::S_MIN_I32: return AMDGPU::V_MIN_I32_e64;
  case AMDGPU::S_MIN_U32: return AMDGPU::V_MIN_U32_e64;
  case AMDGPU::S_MAX_I32: return AMDGPU::V_MAX_I32_e64;
  case AMDGPU::S_MAX_U32: return AMDGPU::V_MAX_U32_e64;
  case AMDGPU::S_ASHR_I32: return AMDGPU::V_ASHR_I32_e32;
  case AMDGPU::S_ASHR_I64: return AMDGPU::V_ASHR_I64_e64;
  case AMDGPU::S_LSHL_B32: return AMDGPU::V_LSHL_B32_e32;
  case AMDGPU::S_LSHL_B64: return AMDGPU::V_LSHL_B64_e64;
  case AMDGPU::S_LSHR_B32: return AMDGPU::V_LSHR_B32_e32;
  case AMDGPU::S_LSHR_B64: return AMDGPU::V_LSHR_B64_e64;
  case AMDGPU::S_SEXT_I32_I8: return AMDGPU::V_BFE_I32_e64;
  case AMDGPU::S_SEXT_I32_I16: return AMDGPU::V_BFE_I32_e64;
  case AMDGPU::S_BFE_U32: return AMDGPU::V_BFE_U32_e64;
  case AMDGPU::S_BFE_I32: return AMDGPU::V_BFE_I32_e64;
  case AMDGPU::S_BFM_B32: return AMDGPU::V_BFM_B32_e64;
  case AMDGPU::S_BREV_B32: return AMDGPU::V_BFREV_B32_e32;
  case AMDGPU::S_NOT_B32: return AMDGPU::V_NOT_B32_e32;
  case AMDGPU::S_NOT_B64: return AMDGPU::V_NOT_B32_e32;
  case AMDGPU::S_CMP_EQ_I32: return AMDGPU::V_CMP_EQ_I32_e64;
  case AMDGPU::S_CMP_LG_I32: return AMDGPU::V_CMP_NE_I32_e64;
  case AMDGPU::S_CMP_GT_I32: return AMDGPU::V_CMP_GT_I32_e64;
  case AMDGPU::S_CMP_GE_I32: return AMDGPU::V_CMP_GE_I32_e64;
  case AMDGPU::S_CMP_LT_I32: return AMDGPU::V_CMP_LT_I32_e64;
  case AMDGPU::S_CMP_LE_I32: return AMDGPU::V_CMP_LE_I32_e64;
  case AMDGPU::S_CMP_EQ_U32: return AMDGPU::V_CMP_EQ_U32_e64;
  case AMDGPU::S_CMP_LG_U32: return AMDGPU::V_CMP_NE_U32_e64;
  case AMDGPU::S_CMP_GT_U32: return AMDGPU::V_CMP_GT_U32_e64;
  case AMDGPU::S_CMP_GE_U32: return AMDGPU::V_CMP_GE_U32_e64;
  case AMDGPU::S_CMP_LT_U32: return AMDGPU::V_CMP_LT_U32_e64;
  case AMDGPU::S_CMP_LE_U32: return AMDGPU::V_CMP_LE_U32_e64;
  case AMDGPU::S_CMP_EQ_U64: return AMDGPU::V_CMP_EQ_U64_e64;
  case AMDGPU::S_CMP_LG_U64: return AMDGPU::V_CMP_NE_U64_e64;
  case AMDGPU::S_BCNT1_I32_B32: return AMDGPU::V_BCNT_U32_B32_e64;
  case AMDGPU::S_FF1_I32_B32: return AMDGPU::V_FFBL_B32_e32;
  case AMDGPU::S_FLBIT_I32_B32: return AMDGPU::V_FFBH_U32_e32;
  case AMDGPU::S_FLBIT_I32: return AMDGPU::V_FFBH_I32_e64;
  case AMDGPU::S_CBRANCH_SCC0: return AMDGPU::S_CBRANCH_VCCZ;
  case AMDGPU::S_CBRANCH_SCC1: return AMDGPU::S_CBRANCH_VCCNZ;
  case AMDGPU::S_CVT_F32_I32: return AMDGPU::V_CVT_F32_I32_e64;
  case AMDGPU::S_CVT_F32_U32: return AMDGPU::V_CVT_F32_U32_e64;
  case AMDGPU::S_CVT_I32_F32: return AMDGPU::V_CVT_I32_F32_e64;
  case AMDGPU::S_CVT_U32_F32: return AMDGPU::V_CVT_U32_F32_e64;
  case AMDGPU::S_CVT_F32_F16:
  case AMDGPU::S_CVT_HI_F32_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CVT_F32_F16_t16_e64
                                   : AMDGPU::V_CVT_F32_F16_fake16_e64;
  case AMDGPU::S_CVT_F16_F32:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CVT_F16_F32_t16_e64
                                   : AMDGPU::V_CVT_F16_F32_fake16_e64;
  case AMDGPU::S_CEIL_F32: return AMDGPU::V_CEIL_F32_e64;
  case AMDGPU::S_FLOOR_F32: return AMDGPU::V_FLOOR_F32_e64;
  case AMDGPU::S_TRUNC_F32: return AMDGPU::V_TRUNC_F32_e64;
  case AMDGPU::S_RNDNE_F32: return AMDGPU::V_RNDNE_F32_e64;
  case AMDGPU::S_CEIL_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CEIL_F16_t16_e64
                                   : AMDGPU::V_CEIL_F16_fake16_e64;
  case AMDGPU::S_FLOOR_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_FLOOR_F16_t16_e64
                                   : AMDGPU::V_FLOOR_F16_fake16_e64;
  case AMDGPU::S_TRUNC_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_TRUNC_F16_t16_e64
                                   : AMDGPU::V_TRUNC_F16_fake16_e64;
  case AMDGPU::S_RNDNE_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_RNDNE_F16_t16_e64
                                   : AMDGPU::V_RNDNE_F16_fake16_e64;
  case AMDGPU::S_ADD_F32: return AMDGPU::V_ADD_F32_e64;
  case AMDGPU::S_SUB_F32: return AMDGPU::V_SUB_F32_e64;
  case AMDGPU::S_MIN_F32: return AMDGPU::V_MIN_F32_e64;
  case AMDGPU::S_MAX_F32: return AMDGPU::V_MAX_F32_e64;
  case AMDGPU::S_MINIMUM_F32: return AMDGPU::V_MINIMUM_F32_e64;
  case AMDGPU::S_MAXIMUM_F32: return AMDGPU::V_MAXIMUM_F32_e64;
  case AMDGPU::S_MUL_F32: return AMDGPU::V_MUL_F32_e64;
  case AMDGPU::S_ADD_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_ADD_F16_t16_e64
                                   : AMDGPU::V_ADD_F16_fake16_e64;
  case AMDGPU::S_SUB_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_SUB_F16_t16_e64
                                   : AMDGPU::V_SUB_F16_fake16_e64;
  case AMDGPU::S_MIN_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_MIN_F16_t16_e64
                                   : AMDGPU::V_MIN_F16_fake16_e64;
  case AMDGPU::S_MAX_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_MAX_F16_t16_e64
                                   : AMDGPU::V_MAX_F16_fake16_e64;
  case AMDGPU::S_MINIMUM_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_MINIMUM_F16_t16_e64
                                   : AMDGPU::V_MINIMUM_F16_fake16_e64;
  case AMDGPU::S_MAXIMUM_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_MAXIMUM_F16_t16_e64
                                   : AMDGPU::V_MAXIMUM_F16_fake16_e64;
  case AMDGPU::S_MUL_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_MUL_F16_t16_e64
                                   : AMDGPU::V_MUL_F16_fake16_e64;
  case AMDGPU::S_CVT_PK_RTZ_F16_F32: return AMDGPU::V_CVT_PKRTZ_F16_F32_e64;
  case AMDGPU::S_FMAC_F32: return AMDGPU::V_FMAC_F32_e64;
  case AMDGPU::S_FMAC_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_FMAC_F16_t16_e64
                                   : AMDGPU::V_FMAC_F16_fake16_e64;
  case AMDGPU::S_FMAMK_F32: return AMDGPU::V_FMAMK_F32;
  case AMDGPU::S_FMAAK_F32: return AMDGPU::V_FMAAK_F32;
  case AMDGPU::S_CMP_LT_F32: return AMDGPU::V_CMP_LT_F32_e64;
  case AMDGPU::S_CMP_EQ_F32: return AMDGPU::V_CMP_EQ_F32_e64;
  case AMDGPU::S_CMP_LE_F32: return AMDGPU::V_CMP_LE_F32_e64;
  case AMDGPU::S_CMP_GT_F32: return AMDGPU::V_CMP_GT_F32_e64;
  case AMDGPU::S_CMP_LG_F32: return AMDGPU::V_CMP_LG_F32_e64;
  case AMDGPU::S_CMP_GE_F32: return AMDGPU::V_CMP_GE_F32_e64;
  case AMDGPU::S_CMP_O_F32: return AMDGPU::V_CMP_O_F32_e64;
  case AMDGPU::S_CMP_U_F32: return AMDGPU::V_CMP_U_F32_e64;
  case AMDGPU::S_CMP_NGE_F32: return AMDGPU::V_CMP_NGE_F32_e64;
  case AMDGPU::S_CMP_NLG_F32: return AMDGPU::V_CMP_NLG_F32_e64;
  case AMDGPU::S_CMP_NGT_F32: return AMDGPU::V_CMP_NGT_F32_e64;
  case AMDGPU::S_CMP_NLE_F32: return AMDGPU::V_CMP_NLE_F32_e64;
  case AMDGPU::S_CMP_NEQ_F32: return AMDGPU::V_CMP_NEQ_F32_e64;
  case AMDGPU::S_CMP_NLT_F32: return AMDGPU::V_CMP_NLT_F32_e64;
  case AMDGPU::S_CMP_LT_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_LT_F16_t16_e64
                                   : AMDGPU::V_CMP_LT_F16_fake16_e64;
  case AMDGPU::S_CMP_EQ_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_EQ_F16_t16_e64
                                   : AMDGPU::V_CMP_EQ_F16_fake16_e64;
  case AMDGPU::S_CMP_LE_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_LE_F16_t16_e64
                                   : AMDGPU::V_CMP_LE_F16_fake16_e64;
  case AMDGPU::S_CMP_GT_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_GT_F16_t16_e64
                                   : AMDGPU::V_CMP_GT_F16_fake16_e64;
  case AMDGPU::S_CMP_LG_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_LG_F16_t16_e64
                                   : AMDGPU::V_CMP_LG_F16_fake16_e64;
  case AMDGPU::S_CMP_GE_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_GE_F16_t16_e64
                                   : AMDGPU::V_CMP_GE_F16_fake16_e64;
  case AMDGPU::S_CMP_O_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_O_F16_t16_e64
                                   : AMDGPU::V_CMP_O_F16_fake16_e64;
  case AMDGPU::S_CMP_U_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_U_F16_t16_e64
                                   : AMDGPU::V_CMP_U_F16_fake16_e64;
  case AMDGPU::S_CMP_NGE_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_NGE_F16_t16_e64
                                   : AMDGPU::V_CMP_NGE_F16_fake16_e64;
  case AMDGPU::S_CMP_NLG_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_NLG_F16_t16_e64
                                   : AMDGPU::V_CMP_NLG_F16_fake16_e64;
  case AMDGPU::S_CMP_NGT_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_NGT_F16_t16_e64
                                   : AMDGPU::V_CMP_NGT_F16_fake16_e64;
  case AMDGPU::S_CMP_NLE_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_NLE_F16_t16_e64
                                   : AMDGPU::V_CMP_NLE_F16_fake16_e64;
  case AMDGPU::S_CMP_NEQ_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_NEQ_F16_t16_e64
                                   : AMDGPU::V_CMP_NEQ_F16_fake16_e64;
  case AMDGPU::S_CMP_NLT_F16:
    return ST.useRealTrue16Insts() ? AMDGPU::V_CMP_NLT_F16_t16_e64
                                   : AMDGPU::V_CMP_NLT_F16_fake16_e64;
  case AMDGPU::V_S_EXP_F32_e64: return AMDGPU::V_EXP_F32_e64;
  case AMDGPU::V_S_EXP_F16_e64:
    return ST.useRealTrue16Insts() ? AMDGPU::V_EXP_F16_t16_e64
                                   : AMDGPU::V_EXP_F16_fake16_e64;
  case AMDGPU::V_S_LOG_F32_e64: return AMDGPU::V_LOG_F32_e64;
  case AMDGPU::V_S_LOG_F16_e64:
    return ST.useRealTrue16Insts() ? AMDGPU::V_LOG_F16_t16_e64
                                   : AMDGPU::V_LOG_F16_fake16_e64;
  case AMDGPU::V_S_RCP_F32_e64: return AMDGPU::V_RCP_F32_e64;
  case AMDGPU::V_S_RCP_F16_e64:
    return ST.useRealTrue16Insts() ? AMDGPU::V_RCP_F16_t16_e64
                                   : AMDGPU::V_RCP_F16_fake16_e64;
  case AMDGPU::V_S_RSQ_F32_e64: return AMDGPU::V_RSQ_F32_e64;
  case AMDGPU::V_S_RSQ_F16_e64:
    return ST.useRealTrue16Insts() ? AMDGPU::V_RSQ_F16_t16_e64
                                   : AMDGPU::V_RSQ_F16_fake16_e64;
  case AMDGPU::V_S_SQRT_F32_e64: return AMDGPU::V_SQRT_F32_e64;
  case AMDGPU::V_S_SQRT_F16_e64:
    return ST.useRealTrue16Insts() ? AMDGPU::V_SQRT_F16_t16_e64
                                   : AMDGPU::V_SQRT_F16_fake16_e64;
  }
  llvm_unreachable(
      "Unexpected scalar opcode without corresponding vector one!");
}

// clang-format on

void SIInstrInfo::insertScratchExecCopy(MachineFunction &MF,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        const DebugLoc &DL, Register Reg,
                                        bool IsSCCLive,
                                        SlotIndexes *Indexes) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const AMDGPU::LaneMaskConstants &LMC = AMDGPU::LaneMaskConstants::get(ST);
  if (IsSCCLive) {
    // Insert two move instructions, one to save the original value of EXEC and
    // the other to turn on all bits in EXEC. This is required as we can't use
    // the single instruction S_OR_SAVEEXEC that clobbers SCC.
    auto StoreExecMI = BuildMI(MBB, MBBI, DL, TII->get(LMC.MovOpc), Reg)
                           .addReg(LMC.ExecReg, RegState::Kill);
    auto FlipExecMI =
        BuildMI(MBB, MBBI, DL, TII->get(LMC.MovOpc), LMC.ExecReg).addImm(-1);
    if (Indexes) {
      Indexes->insertMachineInstrInMaps(*StoreExecMI);
      Indexes->insertMachineInstrInMaps(*FlipExecMI);
    }
  } else {
    auto SaveExec =
        BuildMI(MBB, MBBI, DL, TII->get(LMC.OrSaveExecOpc), Reg).addImm(-1);
    SaveExec->getOperand(3).setIsDead(); // Mark SCC as dead.
    if (Indexes)
      Indexes->insertMachineInstrInMaps(*SaveExec);
  }
}

void SIInstrInfo::restoreExec(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              const DebugLoc &DL, Register Reg,
                              SlotIndexes *Indexes) const {
  const AMDGPU::LaneMaskConstants &LMC = AMDGPU::LaneMaskConstants::get(ST);
  auto ExecRestoreMI = BuildMI(MBB, MBBI, DL, get(LMC.MovOpc), LMC.ExecReg)
                           .addReg(Reg, RegState::Kill);
  if (Indexes)
    Indexes->insertMachineInstrInMaps(*ExecRestoreMI);
}

MachineInstr *
SIInstrInfo::getWholeWaveFunctionSetup(MachineFunction &MF) const {
  assert(MF.getInfo<SIMachineFunctionInfo>()->isWholeWaveFunction() &&
         "Not a whole wave func");
  MachineBasicBlock &MBB = *MF.begin();
  for (MachineInstr &MI : MBB)
    if (MI.getOpcode() == AMDGPU::SI_WHOLE_WAVE_FUNC_SETUP ||
        MI.getOpcode() == AMDGPU::G_AMDGPU_WHOLE_WAVE_FUNC_SETUP)
      return &MI;

  llvm_unreachable("Couldn't find SI_SETUP_WHOLE_WAVE_FUNC instruction");
}

const TargetRegisterClass *SIInstrInfo::getOpRegClass(const MachineInstr &MI,
                                                      unsigned OpNo) const {
  const MCInstrDesc &Desc = get(MI.getOpcode());
  if (MI.isVariadic() || OpNo >= Desc.getNumOperands() ||
      Desc.operands()[OpNo].RegClass == -1) {
    Register Reg = MI.getOperand(OpNo).getReg();

    if (Reg.isVirtual()) {
      const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
      return MRI.getRegClass(Reg);
    }
    return RI.getPhysRegBaseClass(Reg);
  }

  int16_t RegClass = getOpRegClassID(Desc.operands()[OpNo]);
  return RegClass < 0 ? nullptr : RI.getRegClass(RegClass);
}

void SIInstrInfo::legalizeOpWithMove(MachineInstr &MI, unsigned OpIdx) const {
  MachineBasicBlock::iterator I = MI;
  MachineBasicBlock *MBB = MI.getParent();
  MachineOperand &MO = MI.getOperand(OpIdx);
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  unsigned RCID = getOpRegClassID(get(MI.getOpcode()).operands()[OpIdx]);
  const TargetRegisterClass *RC = RI.getRegClass(RCID);
  unsigned Size = RI.getRegSizeInBits(*RC);
  unsigned Opcode = (Size == 64) ? AMDGPU::V_MOV_B64_PSEUDO
                    : Size == 16 ? AMDGPU::V_MOV_B16_t16_e64
                                 : AMDGPU::V_MOV_B32_e32;
  if (MO.isReg())
    Opcode = AMDGPU::COPY;
  else if (RI.isSGPRClass(RC))
    Opcode = (Size == 64) ? AMDGPU::S_MOV_B64 : AMDGPU::S_MOV_B32;

  const TargetRegisterClass *VRC = RI.getEquivalentVGPRClass(RC);
  Register Reg = MRI.createVirtualRegister(VRC);
  DebugLoc DL = MBB->findDebugLoc(I);
  BuildMI(*MI.getParent(), I, DL, get(Opcode), Reg).add(MO);
  MO.ChangeToRegister(Reg, false);
}

unsigned SIInstrInfo::buildExtractSubReg(
    MachineBasicBlock::iterator MI, MachineRegisterInfo &MRI,
    const MachineOperand &SuperReg, const TargetRegisterClass *SuperRC,
    unsigned SubIdx, const TargetRegisterClass *SubRC) const {
  if (!SuperReg.getReg().isVirtual())
    return RI.getSubReg(SuperReg.getReg(), SubIdx);

  MachineBasicBlock *MBB = MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();
  Register SubReg = MRI.createVirtualRegister(SubRC);

  unsigned NewSubIdx = RI.composeSubRegIndices(SuperReg.getSubReg(), SubIdx);
  BuildMI(*MBB, MI, DL, get(TargetOpcode::COPY), SubReg)
      .addReg(SuperReg.getReg(), {}, NewSubIdx);
  return SubReg;
}

MachineOperand SIInstrInfo::buildExtractSubRegOrImm(
    MachineBasicBlock::iterator MII, MachineRegisterInfo &MRI,
    const MachineOperand &Op, const TargetRegisterClass *SuperRC,
    unsigned SubIdx, const TargetRegisterClass *SubRC) const {
  if (Op.isImm()) {
    if (SubIdx == AMDGPU::sub0)
      return MachineOperand::CreateImm(static_cast<int32_t>(Op.getImm()));
    if (SubIdx == AMDGPU::sub1)
      return MachineOperand::CreateImm(static_cast<int32_t>(Op.getImm() >> 32));

    llvm_unreachable("Unhandled register index for immediate");
  }

  unsigned SubReg = buildExtractSubReg(MII, MRI, Op, SuperRC,
                                       SubIdx, SubRC);
  return MachineOperand::CreateReg(SubReg, false);
}

// Change the order of operands from (0, 1, 2) to (0, 2, 1)
void SIInstrInfo::swapOperands(MachineInstr &Inst) const {
  assert(Inst.getNumExplicitOperands() == 3);
  MachineOperand Op1 = Inst.getOperand(1);
  Inst.removeOperand(1);
  Inst.addOperand(Op1);
}

bool SIInstrInfo::isLegalRegOperand(const MachineRegisterInfo &MRI,
                                    const MCOperandInfo &OpInfo,
                                    const MachineOperand &MO) const {
  if (!MO.isReg())
    return false;

  Register Reg = MO.getReg();

  const TargetRegisterClass *DRC = RI.getRegClass(getOpRegClassID(OpInfo));
  if (Reg.isPhysical())
    return DRC->contains(Reg);

  const TargetRegisterClass *RC = MRI.getRegClass(Reg);

  if (MO.getSubReg()) {
    const MachineFunction *MF = MO.getParent()->getMF();
    const TargetRegisterClass *SuperRC = RI.getLargestLegalSuperClass(RC, *MF);
    if (!SuperRC)
      return false;
    return RI.getMatchingSuperRegClass(SuperRC, DRC, MO.getSubReg()) != nullptr;
  }

  return RI.getCommonSubClass(DRC, RC) != nullptr;
}

bool SIInstrInfo::isLegalRegOperand(const MachineInstr &MI, unsigned OpIdx,
                                    const MachineOperand &MO) const {
  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  const MCOperandInfo OpInfo = MI.getDesc().operands()[OpIdx];
  unsigned Opc = MI.getOpcode();

  // See SIInstrInfo::isLegalGFX12PlusPackedMathFP32Operand for more
  // information.
  if (AMDGPU::isPackedFP32Inst(MI.getOpcode()) && AMDGPU::isGFX12Plus(ST) &&
      MO.isReg() && RI.isSGPRReg(MRI, MO.getReg())) {
    constexpr AMDGPU::OpName OpNames[] = {
        AMDGPU::OpName::src0, AMDGPU::OpName::src1, AMDGPU::OpName::src2};

    for (auto [I, OpName] : enumerate(OpNames)) {
      int SrcIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OpNames[I]);
      if (static_cast<unsigned>(SrcIdx) == OpIdx &&
          !isLegalGFX12PlusPackedMathFP32Operand(MRI, MI, I, &MO))
        return false;
    }
  }

  if (!isLegalRegOperand(MRI, OpInfo, MO))
    return false;

  // check Accumulate GPR operand
  bool IsAGPR = RI.isAGPR(MRI, MO.getReg());
  if (IsAGPR && !ST.hasMAIInsts())
    return false;
  if (IsAGPR && (!ST.hasGFX90AInsts() || !MRI.reservedRegsFrozen()) &&
      (MI.mayLoad() || MI.mayStore() || isDS(Opc) || isMIMG(Opc)))
    return false;
  // Atomics should have both vdst and vdata either vgpr or agpr.
  const int VDstIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
  const int DataIdx = AMDGPU::getNamedOperandIdx(
      Opc, isDS(Opc) ? AMDGPU::OpName::data0 : AMDGPU::OpName::vdata);
  if ((int)OpIdx == VDstIdx && DataIdx != -1 &&
      MI.getOperand(DataIdx).isReg() &&
      RI.isAGPR(MRI, MI.getOperand(DataIdx).getReg()) != IsAGPR)
    return false;
  if ((int)OpIdx == DataIdx) {
    if (VDstIdx != -1 &&
        RI.isAGPR(MRI, MI.getOperand(VDstIdx).getReg()) != IsAGPR)
      return false;
    // DS instructions with 2 src operands also must have tied RC.
    const int Data1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data1);
    if (Data1Idx != -1 && MI.getOperand(Data1Idx).isReg() &&
        RI.isAGPR(MRI, MI.getOperand(Data1Idx).getReg()) != IsAGPR)
      return false;
  }

  // Check V_ACCVGPR_WRITE_B32_e64
  if (Opc == AMDGPU::V_ACCVGPR_WRITE_B32_e64 && !ST.hasGFX90AInsts() &&
      (int)OpIdx == AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0) &&
      RI.isSGPRReg(MRI, MO.getReg()))
    return false;

  if (ST.hasFlatScratchHiInB64InstHazard() &&
      MO.getReg() == AMDGPU::SRC_FLAT_SCRATCH_BASE_HI && isSALU(MI)) {
    if (const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::sdst)) {
      if (AMDGPU::getRegBitWidth(*RI.getRegClassForReg(MRI, Dst->getReg())) ==
          64)
        return false;
    }
    if (Opc == AMDGPU::S_BITCMP0_B64 || Opc == AMDGPU::S_BITCMP1_B64)
      return false;
  }

  return true;
}

bool SIInstrInfo::isLegalVSrcOperand(const MachineRegisterInfo &MRI,
                                     const MCOperandInfo &OpInfo,
                                     const MachineOperand &MO) const {
  if (MO.isReg())
    return isLegalRegOperand(MRI, OpInfo, MO);

  // Handle non-register types that are treated like immediates.
  assert(MO.isImm() || MO.isTargetIndex() || MO.isFI() || MO.isGlobal());
  return true;
}

bool SIInstrInfo::isLegalGFX12PlusPackedMathFP32Operand(
    const MachineRegisterInfo &MRI, const MachineInstr &MI, unsigned SrcN,
    const MachineOperand *MO) const {
  constexpr unsigned NumOps = 3;
  constexpr AMDGPU::OpName OpNames[NumOps * 2] = {
      AMDGPU::OpName::src0,           AMDGPU::OpName::src1,
      AMDGPU::OpName::src2,           AMDGPU::OpName::src0_modifiers,
      AMDGPU::OpName::src1_modifiers, AMDGPU::OpName::src2_modifiers};

  assert(SrcN < NumOps);

  if (!MO) {
    int SrcIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OpNames[SrcN]);
    if (SrcIdx == -1)
      return true;
    MO = &MI.getOperand(SrcIdx);
  }

  if (!MO->isReg() || !RI.isSGPRReg(MRI, MO->getReg()))
    return true;

  int ModsIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), OpNames[NumOps + SrcN]);
  if (ModsIdx == -1)
    return true;

  unsigned Mods = MI.getOperand(ModsIdx).getImm();
  bool OpSel = Mods & SISrcMods::OP_SEL_0;
  bool OpSelHi = Mods & SISrcMods::OP_SEL_1;

  return !OpSel && !OpSelHi;
}

bool SIInstrInfo::isOperandLegal(const MachineInstr &MI, unsigned OpIdx,
                                 const MachineOperand *MO) const {
  const MachineFunction &MF = *MI.getMF();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const MCInstrDesc &InstDesc = MI.getDesc();
  const MCOperandInfo &OpInfo = InstDesc.operands()[OpIdx];
  int64_t RegClass = getOpRegClassID(OpInfo);
  const TargetRegisterClass *DefinedRC =
      RegClass != -1 ? RI.getRegClass(RegClass) : nullptr;
  if (!MO)
    MO = &MI.getOperand(OpIdx);

  const bool IsInlineConst = !MO->isReg() && isInlineConstant(*MO, OpInfo);

  if (isVALU(MI) && !IsInlineConst && usesConstantBus(MRI, *MO, OpInfo)) {
    const MachineOperand *UsedLiteral = nullptr;

    int ConstantBusLimit = ST.getConstantBusLimit(MI.getOpcode());
    int LiteralLimit = !isVOP3(MI) || ST.hasVOP3Literal() ? 1 : 0;

    // TODO: Be more permissive with frame indexes.
    if (!MO->isReg() && !isInlineConstant(*MO, OpInfo)) {
      if (!LiteralLimit--)
        return false;

      UsedLiteral = MO;
    }

    SmallDenseSet<RegSubRegPair> SGPRsUsed;
    if (MO->isReg())
      SGPRsUsed.insert(RegSubRegPair(MO->getReg(), MO->getSubReg()));

    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      if (i == OpIdx)
        continue;
      const MachineOperand &Op = MI.getOperand(i);
      if (Op.isReg()) {
        if (Op.isUse()) {
          RegSubRegPair SGPR(Op.getReg(), Op.getSubReg());
          if (regUsesConstantBus(Op, MRI) && SGPRsUsed.insert(SGPR).second) {
            if (--ConstantBusLimit <= 0)
              return false;
          }
        }
      } else if (AMDGPU::isSISrcOperand(InstDesc.operands()[i]) &&
                 !isInlineConstant(Op, InstDesc.operands()[i])) {
        // The same literal may be used multiple times.
        if (!UsedLiteral)
          UsedLiteral = &Op;
        else if (UsedLiteral->isIdenticalTo(Op))
          continue;

        if (!LiteralLimit--)
          return false;
        if (--ConstantBusLimit <= 0)
          return false;
      }
    }
  } else if (!IsInlineConst && !MO->isReg() && isSALU(MI)) {
    // There can be at most one literal operand, but it can be repeated.
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      if (i == OpIdx)
        continue;
      const MachineOperand &Op = MI.getOperand(i);
      if (!Op.isReg() && !Op.isFI() && !Op.isRegMask() &&
          !isInlineConstant(Op, InstDesc.operands()[i]) &&
          !Op.isIdenticalTo(*MO))
        return false;

      // Do not fold a non-inlineable and non-register operand into an
      // instruction that already has a frame index. The frame index handling
      // code could not handle well when a frame index co-exists with another
      // non-register operand, unless that operand is an inlineable immediate.
      if (Op.isFI())
        return false;
    }
  } else if (IsInlineConst && ST.hasNoF16PseudoScalarTransInlineConstants() &&
             isF16PseudoScalarTrans(MI.getOpcode())) {
    return false;
  }

  if (MO->isReg()) {
    if (!DefinedRC)
      return OpInfo.OperandType == MCOI::OPERAND_UNKNOWN;
    return isLegalRegOperand(MI, OpIdx, *MO);
  }

  if (MO->isImm()) {
    uint64_t Imm = MO->getImm();
    bool Is64BitFPOp = OpInfo.OperandType == AMDGPU::OPERAND_REG_IMM_FP64;
    bool Is64BitSignedOp = OpInfo.OperandType == AMDGPU::OPERAND_REG_IMM_I64;
    bool Is64BitUnsignedOp = OpInfo.OperandType == AMDGPU::OPERAND_REG_IMM_U64;
    bool Is64BitOp = Is64BitFPOp || Is64BitSignedOp || Is64BitUnsignedOp ||
                     OpInfo.OperandType == AMDGPU::OPERAND_REG_IMM_V2INT32 ||
                     OpInfo.OperandType == AMDGPU::OPERAND_REG_IMM_V2FP32;
    if (Is64BitOp &&
        !AMDGPU::isInlinableLiteral64(Imm, ST.hasInv2PiInlineImm())) {
      if (!AMDGPU::isValid32BitLiteral(Imm, Is64BitFPOp) &&
          (!ST.has64BitLiterals() || InstDesc.getSize() != 4))
        return false;

      // For signed operands, we can use sign extended 32-bit literals when the
      // value fits in a signed 32-bit integer. For unsigned operands, we reject
      // negative values (when interpreted as 32-bit) since they would be
      // zero-extended, not sign-extended.
      // If 64-bit literals are supported and the literal will be encoded
      // as full 64 bit we still can use it.
      if (Is64BitSignedOp) {
        // Signed operand: 32-bit literal is valid if it fits in int32_t
        if (!isInt<32>(static_cast<int64_t>(Imm)) &&
            (!ST.has64BitLiterals() || AMDGPU::isValid32BitLiteral(Imm, false)))
          return false;
      } else if (Is64BitUnsignedOp) {
        // Unsigned operand: 32-bit literal is valid if it fits in uint32_t
        if (!isUInt<32>(Imm) &&
            (!ST.has64BitLiterals() || AMDGPU::isValid32BitLiteral(Imm, false)))
          return false;
      } else if (!Is64BitFPOp && (int32_t)Imm < 0 &&
                 (!ST.has64BitLiterals() ||
                  AMDGPU::isValid32BitLiteral(Imm, false))) {
        // Other 64-bit operands (V2INT32, V2FP32): be conservative
        return false;
      }
    }
  }

  // Handle non-register types that are treated like immediates.
  assert(MO->isImm() || MO->isTargetIndex() || MO->isFI() || MO->isGlobal());

  if (!DefinedRC) {
    // This operand expects an immediate.
    return true;
  }

  return isImmOperandLegal(MI, OpIdx, *MO);
}

bool SIInstrInfo::isNeverCoissue(MachineInstr &MI) const {
  bool IsGFX950Only = ST.hasGFX950Insts();
  bool IsGFX940Only = ST.hasGFX940Insts();

  if (!IsGFX950Only && !IsGFX940Only)
    return false;

  if (!isVALU(MI))
    return false;

  // V_COS, V_EXP, V_RCP, etc.
  if (isTRANS(MI))
    return true;

  // DOT2, DOT2C, DOT4, etc.
  if (isDOT(MI))
    return true;

  // MFMA, SMFMA
  if (isMFMA(MI))
    return true;

  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  case AMDGPU::V_CVT_PK_BF8_F32_e64:
  case AMDGPU::V_CVT_PK_FP8_F32_e64:
  case AMDGPU::V_MQSAD_PK_U16_U8_e64:
  case AMDGPU::V_MQSAD_U32_U8_e64:
  case AMDGPU::V_PK_ADD_F16:
  case AMDGPU::V_PK_ADD_F32:
  case AMDGPU::V_PK_ADD_I16:
  case AMDGPU::V_PK_ADD_U16:
  case AMDGPU::V_PK_ASHRREV_I16:
  case AMDGPU::V_PK_FMA_F16:
  case AMDGPU::V_PK_FMA_F32:
  case AMDGPU::V_PK_FMAC_F16_e32:
  case AMDGPU::V_PK_FMAC_F16_e64:
  case AMDGPU::V_PK_LSHLREV_B16:
  case AMDGPU::V_PK_LSHRREV_B16:
  case AMDGPU::V_PK_MAD_I16:
  case AMDGPU::V_PK_MAD_U16:
  case AMDGPU::V_PK_MAX_F16:
  case AMDGPU::V_PK_MAX_I16:
  case AMDGPU::V_PK_MAX_U16:
  case AMDGPU::V_PK_MIN_F16:
  case AMDGPU::V_PK_MIN_I16:
  case AMDGPU::V_PK_MIN_U16:
  case AMDGPU::V_PK_MOV_B32:
  case AMDGPU::V_PK_MUL_F16:
  case AMDGPU::V_PK_MUL_F32:
  case AMDGPU::V_PK_MUL_LO_U16:
  case AMDGPU::V_PK_SUB_I16:
  case AMDGPU::V_PK_SUB_U16:
  case AMDGPU::V_QSAD_PK_U16_U8_e64:
    return true;
  default:
    return false;
  }
}

void SIInstrInfo::legalizeOperandsVOP2(MachineRegisterInfo &MRI,
                                       MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  const MCInstrDesc &InstrDesc = get(Opc);

  int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  MachineOperand &Src0 = MI.getOperand(Src0Idx);

  int Src1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1);
  MachineOperand &Src1 = MI.getOperand(Src1Idx);

  // If there is an implicit SGPR use such as VCC use for v_addc_u32/v_subb_u32
  // we need to only have one constant bus use before GFX10.
  bool HasImplicitSGPR = findImplicitSGPRRead(MI);
  if (HasImplicitSGPR && ST.getConstantBusLimit(Opc) <= 1 && Src0.isReg() &&
      RI.isSGPRReg(MRI, Src0.getReg()))
    legalizeOpWithMove(MI, Src0Idx);

  // Special case: V_WRITELANE_B32 accepts only immediate or SGPR operands for
  // both the value to write (src0) and lane select (src1).  Fix up non-SGPR
  // src0/src1 with V_READFIRSTLANE.
  if (Opc == AMDGPU::V_WRITELANE_B32) {
    const DebugLoc &DL = MI.getDebugLoc();
    if (Src0.isReg() && RI.isVGPR(MRI, Src0.getReg())) {
      Register Reg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
      BuildMI(*MI.getParent(), MI, DL, get(AMDGPU::V_READFIRSTLANE_B32), Reg)
          .add(Src0);
      Src0.ChangeToRegister(Reg, false);
    }
    if (Src1.isReg() && RI.isVGPR(MRI, Src1.getReg())) {
      Register Reg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
      const DebugLoc &DL = MI.getDebugLoc();
      BuildMI(*MI.getParent(), MI, DL, get(AMDGPU::V_READFIRSTLANE_B32), Reg)
          .add(Src1);
      Src1.ChangeToRegister(Reg, false);
    }
    return;
  }

  // Special case: V_FMAC_F32 and V_FMAC_F16 have src2.
  if (Opc == AMDGPU::V_FMAC_F32_e32 || Opc == AMDGPU::V_FMAC_F16_e32) {
    int Src2Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2);
    if (!RI.isVGPR(MRI, MI.getOperand(Src2Idx).getReg()))
      legalizeOpWithMove(MI, Src2Idx);
  }

  // VOP2 src0 instructions support all operand types, so we don't need to check
  // their legality. If src1 is already legal, we don't need to do anything.
  if (isLegalRegOperand(MRI, InstrDesc.operands()[Src1Idx], Src1))
    return;

  // Special case: V_READLANE_B32 accepts only immediate or SGPR operands for
  // lane select. Fix up using V_READFIRSTLANE, since we assume that the lane
  // select is uniform.
  if (Opc == AMDGPU::V_READLANE_B32 && Src1.isReg() &&
      RI.isVGPR(MRI, Src1.getReg())) {
    Register Reg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
    const DebugLoc &DL = MI.getDebugLoc();
    BuildMI(*MI.getParent(), MI, DL, get(AMDGPU::V_READFIRSTLANE_B32), Reg)
        .add(Src1);
    Src1.ChangeToRegister(Reg, false);
    return;
  }

  // We do not use commuteInstruction here because it is too aggressive and will
  // commute if it is possible. We only want to commute here if it improves
  // legality. This can be called a fairly large number of times so don't waste
  // compile time pointlessly swapping and checking legality again.
  if (HasImplicitSGPR || !MI.isCommutable()) {
    legalizeOpWithMove(MI, Src1Idx);
    return;
  }

  // If src0 can be used as src1, commuting will make the operands legal.
  // Otherwise we have to give up and insert a move.
  //
  // TODO: Other immediate-like operand kinds could be commuted if there was a
  // MachineOperand::ChangeTo* for them.
  if ((!Src1.isImm() && !Src1.isReg()) ||
      !isLegalRegOperand(MRI, InstrDesc.operands()[Src1Idx], Src0)) {
    legalizeOpWithMove(MI, Src1Idx);
    return;
  }

  int CommutedOpc = commuteOpcode(MI);
  if (CommutedOpc == -1) {
    legalizeOpWithMove(MI, Src1Idx);
    return;
  }

  MI.setDesc(get(CommutedOpc));

  Register Src0Reg = Src0.getReg();
  unsigned Src0SubReg = Src0.getSubReg();
  bool Src0Kill = Src0.isKill();

  if (Src1.isImm())
    Src0.ChangeToImmediate(Src1.getImm());
  else if (Src1.isReg()) {
    Src0.ChangeToRegister(Src1.getReg(), false, false, Src1.isKill());
    Src0.setSubReg(Src1.getSubReg());
  } else
    llvm_unreachable("Should only have register or immediate operands");

  Src1.ChangeToRegister(Src0Reg, false, false, Src0Kill);
  Src1.setSubReg(Src0SubReg);
  fixImplicitOperands(MI);
}

// Legalize VOP3 operands. All operand types are supported for any operand
// but only one literal constant and only starting from GFX10.
void SIInstrInfo::legalizeOperandsVOP3(MachineRegisterInfo &MRI,
                                       MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  int VOP3Idx[3] = {
    AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0),
    AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1),
    AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2)
  };

  if (Opc == AMDGPU::V_PERMLANE16_B32_e64 ||
      Opc == AMDGPU::V_PERMLANEX16_B32_e64 ||
      Opc == AMDGPU::V_PERMLANE_BCAST_B32_e64 ||
      Opc == AMDGPU::V_PERMLANE_UP_B32_e64 ||
      Opc == AMDGPU::V_PERMLANE_DOWN_B32_e64 ||
      Opc == AMDGPU::V_PERMLANE_XOR_B32_e64 ||
      Opc == AMDGPU::V_PERMLANE_IDX_GEN_B32_e64) {
    // src1 and src2 must be scalar
    MachineOperand &Src1 = MI.getOperand(VOP3Idx[1]);
    const DebugLoc &DL = MI.getDebugLoc();
    if (Src1.isReg() && !RI.isSGPRClass(MRI.getRegClass(Src1.getReg()))) {
      Register Reg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
      BuildMI(*MI.getParent(), MI, DL, get(AMDGPU::V_READFIRSTLANE_B32), Reg)
        .add(Src1);
      Src1.ChangeToRegister(Reg, false);
    }
    if (VOP3Idx[2] != -1) {
      MachineOperand &Src2 = MI.getOperand(VOP3Idx[2]);
      if (Src2.isReg() && !RI.isSGPRClass(MRI.getRegClass(Src2.getReg()))) {
        Register Reg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
        BuildMI(*MI.getParent(), MI, DL, get(AMDGPU::V_READFIRSTLANE_B32), Reg)
            .add(Src2);
        Src2.ChangeToRegister(Reg, false);
      }
    }
  }

  // Find the one SGPR operand we are allowed to use.
  int ConstantBusLimit = ST.getConstantBusLimit(Opc);
  int LiteralLimit = ST.hasVOP3Literal() ? 1 : 0;
  SmallDenseSet<unsigned> SGPRsUsed;
  Register SGPRReg = findUsedSGPR(MI, VOP3Idx);
  if (SGPRReg) {
    SGPRsUsed.insert(SGPRReg);
    --ConstantBusLimit;
  }

  for (int Idx : VOP3Idx) {
    if (Idx == -1)
      break;
    MachineOperand &MO = MI.getOperand(Idx);

    if (!MO.isReg()) {
      if (isInlineConstant(MO, get(Opc).operands()[Idx]))
        continue;

      if (LiteralLimit > 0 && ConstantBusLimit > 0) {
        --LiteralLimit;
        --ConstantBusLimit;
        continue;
      }

      --LiteralLimit;
      --ConstantBusLimit;
      legalizeOpWithMove(MI, Idx);
      continue;
    }

    if (!RI.isSGPRClass(RI.getRegClassForReg(MRI, MO.getReg())))
      continue; // VGPRs are legal

    // We can use one SGPR in each VOP3 instruction prior to GFX10
    // and two starting from GFX10.
    if (SGPRsUsed.count(MO.getReg()))
      continue;
    if (ConstantBusLimit > 0) {
      SGPRsUsed.insert(MO.getReg());
      --ConstantBusLimit;
      continue;
    }

    // If we make it this far, then the operand is not legal and we must
    // legalize it.
    legalizeOpWithMove(MI, Idx);
  }

  // Special case: V_FMAC_F32 and V_FMAC_F16 have src2 tied to vdst.
  if ((Opc == AMDGPU::V_FMAC_F32_e64 || Opc == AMDGPU::V_FMAC_F16_e64) &&
      !RI.isVGPR(MRI, MI.getOperand(VOP3Idx[2]).getReg()))
    legalizeOpWithMove(MI, VOP3Idx[2]);

  // Fix the register class of packed FP32 instructions on gfx12+. See
  // SIInstrInfo::isLegalGFX12PlusPackedMathFP32Operand for more information.
  if (AMDGPU::isPackedFP32Inst(Opc) && AMDGPU::isGFX12Plus(ST)) {
    for (unsigned I = 0; I < 3; ++I) {
      if (!isLegalGFX12PlusPackedMathFP32Operand(MRI, MI, /*SrcN=*/I))
        legalizeOpWithMove(MI, VOP3Idx[I]);
    }
  }
}

Register SIInstrInfo::readlaneVGPRToSGPR(
    Register SrcReg, MachineInstr &UseMI, MachineRegisterInfo &MRI,
    const TargetRegisterClass *DstRC /*=nullptr*/) const {
  const TargetRegisterClass *VRC = MRI.getRegClass(SrcReg);
  const TargetRegisterClass *SRC = RI.getEquivalentSGPRClass(VRC);
  if (DstRC)
    SRC = RI.getCommonSubClass(SRC, DstRC);

  Register DstReg = MRI.createVirtualRegister(SRC);
  unsigned SubRegs = RI.getRegSizeInBits(*VRC) / 32;

  if (RI.hasAGPRs(VRC)) {
    VRC = RI.getEquivalentVGPRClass(VRC);
    Register NewSrcReg = MRI.createVirtualRegister(VRC);
    BuildMI(*UseMI.getParent(), UseMI, UseMI.getDebugLoc(),
            get(TargetOpcode::COPY), NewSrcReg)
        .addReg(SrcReg);
    SrcReg = NewSrcReg;
  }

  if (SubRegs == 1) {
    BuildMI(*UseMI.getParent(), UseMI, UseMI.getDebugLoc(),
            get(AMDGPU::V_READFIRSTLANE_B32), DstReg)
        .addReg(SrcReg);
    return DstReg;
  }

  SmallVector<Register, 8> SRegs;
  for (unsigned i = 0; i < SubRegs; ++i) {
    Register SGPR = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
    BuildMI(*UseMI.getParent(), UseMI, UseMI.getDebugLoc(),
            get(AMDGPU::V_READFIRSTLANE_B32), SGPR)
        .addReg(SrcReg, {}, RI.getSubRegFromChannel(i));
    SRegs.push_back(SGPR);
  }

  MachineInstrBuilder MIB =
      BuildMI(*UseMI.getParent(), UseMI, UseMI.getDebugLoc(),
              get(AMDGPU::REG_SEQUENCE), DstReg);
  for (unsigned i = 0; i < SubRegs; ++i) {
    MIB.addReg(SRegs[i]);
    MIB.addImm(RI.getSubRegFromChannel(i));
  }
  return DstReg;
}

void SIInstrInfo::legalizeOperandsSMRD(MachineRegisterInfo &MRI,
                                       MachineInstr &MI) const {

  // If the pointer is store in VGPRs, then we need to move them to
  // SGPRs using v_readfirstlane.  This is safe because we only select
  // loads with uniform pointers to SMRD instruction so we know the
  // pointer value is uniform.
  MachineOperand *SBase = getNamedOperand(MI, AMDGPU::OpName::sbase);
  if (SBase && !RI.isSGPRClass(MRI.getRegClass(SBase->getReg()))) {
    Register SGPR = readlaneVGPRToSGPR(SBase->getReg(), MI, MRI);
    SBase->setReg(SGPR);
  }
  MachineOperand *SOff = getNamedOperand(MI, AMDGPU::OpName::soffset);
  if (SOff && !RI.isSGPRReg(MRI, SOff->getReg())) {
    Register SGPR = readlaneVGPRToSGPR(SOff->getReg(), MI, MRI);
    SOff->setReg(SGPR);
  }
}

bool SIInstrInfo::moveFlatAddrToVGPR(MachineInstr &Inst) const {
  unsigned Opc = Inst.getOpcode();
  int OldSAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::saddr);
  if (OldSAddrIdx < 0)
    return false;

  assert(isSegmentSpecificFLAT(Inst) || (isFLAT(Inst) && ST.hasFlatGVSMode()));

  int NewOpc = AMDGPU::getGlobalVaddrOp(Opc);
  if (NewOpc < 0)
    NewOpc = AMDGPU::getFlatScratchInstSVfromSS(Opc);
  if (NewOpc < 0)
    return false;

  MachineRegisterInfo &MRI = Inst.getMF()->getRegInfo();
  MachineOperand &SAddr = Inst.getOperand(OldSAddrIdx);
  if (RI.isSGPRReg(MRI, SAddr.getReg()))
    return false;

  int NewVAddrIdx = AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vaddr);
  if (NewVAddrIdx < 0)
    return false;

  int OldVAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr);

  // Check vaddr, it shall be zero or absent.
  MachineInstr *VAddrDef = nullptr;
  if (OldVAddrIdx >= 0) {
    MachineOperand &VAddr = Inst.getOperand(OldVAddrIdx);
    VAddrDef = MRI.getUniqueVRegDef(VAddr.getReg());
    if (!VAddrDef || !VAddrDef->isMoveImmediate() ||
        !VAddrDef->getOperand(1).isImm() ||
        VAddrDef->getOperand(1).getImm() != 0)
      return false;
  }

  const MCInstrDesc &NewDesc = get(NewOpc);
  Inst.setDesc(NewDesc);

  // Callers expect iterator to be valid after this call, so modify the
  // instruction in place.
  if (OldVAddrIdx == NewVAddrIdx) {
    MachineOperand &NewVAddr = Inst.getOperand(NewVAddrIdx);
    // Clear use list from the old vaddr holding a zero register.
    MRI.removeRegOperandFromUseList(&NewVAddr);
    MRI.moveOperands(&NewVAddr, &SAddr, 1);
    Inst.removeOperand(OldSAddrIdx);
    // Update the use list with the pointer we have just moved from vaddr to
    // saddr position. Otherwise new vaddr will be missing from the use list.
    MRI.removeRegOperandFromUseList(&NewVAddr);
    MRI.addRegOperandToUseList(&NewVAddr);
  } else {
    assert(OldSAddrIdx == NewVAddrIdx);

    if (OldVAddrIdx >= 0) {
      int NewVDstIn = AMDGPU::getNamedOperandIdx(NewOpc,
                                                 AMDGPU::OpName::vdst_in);

      // removeOperand doesn't try to fixup tied operand indexes at it goes, so
      // it asserts. Untie the operands for now and retie them afterwards.
      if (NewVDstIn != -1) {
        int OldVDstIn = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst_in);
        Inst.untieRegOperand(OldVDstIn);
      }

      Inst.removeOperand(OldVAddrIdx);

      if (NewVDstIn != -1) {
