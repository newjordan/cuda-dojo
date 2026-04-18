# ==============================================================================
# CUDA Chess Engine - Makefile
# Default single-GPU target: NVIDIA GB10 / DGX Spark (sm_121)
# Override with CUDA_SINGLE_GPU_TARGET=<name> or CUDA_ARCH=<sm_xxx>.
# ==============================================================================

NVCC ?= /usr/local/cuda-13.0/bin/nvcc
CUDA_PATH ?= /usr/local/cuda-13.0
CUDA_SINGLE_GPU_TARGET ?= dgx-spark
CUDA_ARCH ?=
CUDA_EXTRA_NVCC_FLAGS ?=

ifeq ($(strip $(CUDA_ARCH)),)
  ifeq ($(CUDA_SINGLE_GPU_TARGET),dgx-spark)
    CUDA_ARCH := sm_121
  else ifeq ($(CUDA_SINGLE_GPU_TARGET),h200)
    CUDA_ARCH := sm_90a
  else ifeq ($(CUDA_SINGLE_GPU_TARGET),native)
    CUDA_ARCH := native
  else
    $(error Unsupported CUDA_SINGLE_GPU_TARGET '$(CUDA_SINGLE_GPU_TARGET)'; set CUDA_ARCH explicitly)
  endif
endif

ifeq ($(CUDA_ARCH),native)
  ARCH_FLAGS := -arch=native
else
  ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

# Optimization flags
OPT_FLAGS ?= -O3 --use_fast_math
LINK_FLAGS ?= -lcurand
NVCCFLAGS := $(ARCH_FLAGS) $(OPT_FLAGS) $(CUDA_EXTRA_NVCC_FLAGS)
PYTHON_BIN ?= $(shell command -v python3 2>/dev/null)
ORACLE_LIMIT ?= 32
ORACLE_DEPTH ?= 12
ORACLE_TOP_K ?= 16
ORACLE_PROBE_SIMS ?= 2000

# Targets
all: chess_mcts batch_eval knob_tuner gpu_forge gpu_fighter nn_eval nn_train fused_qsearch fused_eval

nn_eval: nn_eval.cu include/dojo_types.h
	$(NVCC) $(NVCCFLAGS) -o nn_eval nn_eval.cu -lcublas -lcurand

nn_train: nn_train.cu include/dojo_types.h
	$(NVCC) $(NVCCFLAGS) -o nn_train nn_train.cu -lcublas -lcurand

chess_mcts: chess_mcts.cu
	$(NVCC) $(NVCCFLAGS) -o chess_mcts chess_mcts.cu $(LINK_FLAGS)

gpu_fighter: gpu_fighter.cu
	$(NVCC) $(NVCCFLAGS) -o gpu_fighter gpu_fighter.cu

batch_eval: batch_eval.cu
	$(NVCC) $(NVCCFLAGS) -o batch_eval batch_eval.cu

knob_tuner: knob_tuner.cu
	$(NVCC) $(NVCCFLAGS) -o knob_tuner knob_tuner.cu $(LINK_FLAGS)

gpu_forge: gpu_forge.cu
	$(NVCC) $(NVCCFLAGS) -o gpu_forge gpu_forge.cu $(LINK_FLAGS)

fused_qsearch: fused_qsearch.cu include/dojo_types.h
	$(NVCC) $(NVCCFLAGS) -o fused_qsearch fused_qsearch.cu $(LINK_FLAGS)

fused_eval: fused_eval.cu include/dojo_types.h
	$(NVCC) $(NVCCFLAGS) -o fused_eval fused_eval.cu

clean:
	rm -f chess_mcts batch_eval knob_tuner gpu_forge gpu_fighter nn_eval nn_train fused_qsearch fused_eval

test: chess_mcts batch_eval
	@echo "=== Testing MCTS (starting position, 1000 sims) ==="
	echo "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" | ./chess_mcts --simulations 1000 --verbose
	@echo ""
	@echo "=== Testing MCTS (JSON output) ==="
	echo "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" | ./chess_mcts --simulations 1000 --json
	@echo ""
	@echo "=== Testing Batch Evaluator ==="
	printf "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nrnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\nr1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2\n" | ./batch_eval
	@echo ""
	@echo "=== All tests passed ==="

mcts_audit: chess_mcts mcts_audit.py
	$(PYTHON_BIN) ./mcts_audit.py

mcts_ablate: chess_mcts mcts_ablate.py
	$(PYTHON_BIN) ./mcts_ablate.py

root_shakeout: chess_mcts root_shakeout.py
	$(PYTHON_BIN) ./root_shakeout.py

root_region_calibrate: chess_mcts root_region_calibrate.py
	$(PYTHON_BIN) ./root_region_calibrate.py

oracle_sigma_smoke:
	node ./oracle_enrich.mjs \
	  --input ./self_play_data.jsonl \
	  --output ./artifacts/oracle_sigma_smoke/enriched.jsonl \
	  --report ./artifacts/oracle_sigma_smoke/report.json \
	  --limit $(ORACLE_LIMIT) \
	  --depth $(ORACLE_DEPTH) \
	  --move-key move \
	  --score-key engine_score_cp \
	  --with-lozza

oracle_sigma_probe: chess_mcts oracle_sigma_smoke
	$(PYTHON_BIN) ./oracle_sigma_probe.py \
	  --input ./artifacts/oracle_sigma_smoke/enriched.jsonl \
	  --output ./artifacts/oracle_sigma_smoke/probe_report.json \
	  --top-k $(ORACLE_TOP_K) \
	  --simulations $(ORACLE_PROBE_SIMS)

.PHONY: all clean test mcts_audit mcts_ablate root_shakeout root_region_calibrate oracle_sigma_smoke oracle_sigma_probe
