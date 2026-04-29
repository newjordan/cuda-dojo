# =============================================================================
# cuda/engine/Makefile
# =============================================================================
NVCC      ?= /usr/local/cuda-13.0/bin/nvcc
CUDA_PATH ?= /usr/local/cuda-13.0
CUDA_SINGLE_GPU_TARGET ?= dgx-spark
CUDA_ARCH ?=
CUDA_EXTRA_NVCC_FLAGS ?=
OPT_FLAGS  ?= -O3 --use_fast_math
CXX_STD    ?= -std=c++17
INCLUDES   ?= -I.
RDC_FLAGS  ?= -rdc=true
PYTHON_BIN ?= $(shell command -v python3 2>/dev/null)

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
  CUDA_ARCH_FLAGS := -arch=native
else
  CUDA_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

ARCH_FLAGS ?= $(CUDA_ARCH_FLAGS)
ENGINE_ARCH_FLAGS ?= $(ARCH_FLAGS)

.DEFAULT_GOAL := all

ENGINE_SRCS = movegen.cu make_unmake.cu attacks.cu eval.cu eval_service.cu scheduler.cu
ENGINE_OBJS = movegen.o make_unmake.o attacks.o eval.o eval_service.o scheduler.o
ENGINE_HDRS = engine_types.h movegen.cuh make_unmake.cuh attacks.cuh eval.cuh eval_service.cuh scheduler.cuh queues.cuh bitboard.cuh
LIB_NAME    = libengine.a

ENGINE_BIN_OBJS = engine_main.o uci_engine.o search.o time.o zobrist.o tt.o \
                  movegen.o make_unmake.o attacks.o eval.o eval_service.o scheduler.o

NVCCFLAGS_LIB = $(ENGINE_ARCH_FLAGS) $(OPT_FLAGS) $(INCLUDES) $(CXX_STD) $(RDC_FLAGS) $(CUDA_EXTRA_NVCC_FLAGS)
NVCCFLAGS_BIN = $(ENGINE_ARCH_FLAGS) $(OPT_FLAGS) $(INCLUDES) $(CXX_STD) $(RDC_FLAGS) $(CUDA_EXTRA_NVCC_FLAGS)
NVCCFLAGS_ENG = $(ENGINE_ARCH_FLAGS) $(OPT_FLAGS) $(CXX_STD) -Xcompiler -Wall $(RDC_FLAGS) $(CUDA_EXTRA_NVCC_FLAGS)

all: $(LIB_NAME) engine engine_test_perft tests/test_makeunmake_symmetry

movegen.o: movegen.cu $(ENGINE_HDRS)
	$(NVCC) $(NVCCFLAGS_LIB) -dc $< -o $@

make_unmake.o: make_unmake.cu $(ENGINE_HDRS)
	$(NVCC) $(NVCCFLAGS_LIB) -dc $< -o $@

attacks.o: attacks.cu $(ENGINE_HDRS)
	$(NVCC) $(NVCCFLAGS_LIB) -dc $< -o $@

eval.o: eval.cu $(ENGINE_HDRS)
	$(NVCC) $(NVCCFLAGS_LIB) -dc $< -o $@

eval_service.o: eval_service.cu $(ENGINE_HDRS)
	$(NVCC) $(NVCCFLAGS_LIB) -dc $< -o $@

scheduler.o: scheduler.cu $(ENGINE_HDRS)
	$(NVCC) $(NVCCFLAGS_LIB) -dc $< -o $@

$(LIB_NAME): $(ENGINE_OBJS)
	$(NVCC) $(ENGINE_ARCH_FLAGS) -lib $(ENGINE_OBJS) -o $(LIB_NAME)

engine_test_perft: tests/test_perft.cu $(LIB_NAME)
	$(NVCC) $(NVCCFLAGS_LIB) tests/test_perft.cu $(LIB_NAME) -o $@

tests/test_makeunmake_symmetry: tests/test_makeunmake_symmetry.cu $(LIB_NAME)
	$(NVCC) $(NVCCFLAGS_LIB) tests/test_makeunmake_symmetry.cu $(LIB_NAME) -o $@

uci_test_engine: uci.cu uci_stubs.cu uci.h
	$(NVCC) $(NVCCFLAGS_LIB) -o $@ uci.cu uci_stubs.cu

uci_test: engine uci_test.sh
	bash ./uci_test.sh

arena_gpu_smoke: engine arena_gpu_smoke.sh
	bash ./arena_gpu_smoke.sh

arena_smoke: engine arena_smoke.sh
	bash ./arena_smoke.sh

engine_verify_long: engine verify_engine_long.py
	$(PYTHON_BIN) ./verify_engine_long.py

zobrist.o: zobrist.cu zobrist.cuh engine_types.h
	$(NVCC) $(NVCCFLAGS_BIN) -dc -c $< -o $@

tt.o: tt.cu tt.cuh
	$(NVCC) $(NVCCFLAGS_BIN) -dc -c $< -o $@

search.o: search.cu search.cuh movegen.cuh make_unmake.cuh attacks.cuh eval.cuh zobrist.cuh tt.cuh time.cuh uci.h
	$(NVCC) $(NVCCFLAGS_BIN) -dc -c $< -o $@

time.o: time.cu time.cuh
	$(NVCC) $(NVCCFLAGS_BIN) -dc -c $< -o $@

engine_main.o: engine.cu uci.h search.cuh movegen.cuh make_unmake.cuh tt.cuh zobrist.cuh eval_service.cuh scheduler.cuh
	$(NVCC) $(NVCCFLAGS_BIN) -dc -c $< -o $@

uci_engine.o: uci.cu uci.h
	$(NVCC) $(NVCCFLAGS_BIN) -DUCI_NO_MAIN -dc -c $< -o $@

engine: $(ENGINE_BIN_OBJS)
	$(NVCC) $(NVCCFLAGS_BIN) $(ENGINE_BIN_OBJS) -o $@

tests/test_zobrist: tests/test_zobrist.cu zobrist.o engine_types.h zobrist.cuh
	$(NVCC) $(NVCCFLAGS_ENG) -rdc=true tests/test_zobrist.cu zobrist.o -o $@

tests/test_tt: tests/test_tt.cu tt.o tt.cuh
	$(NVCC) $(NVCCFLAGS_ENG) -rdc=true tests/test_tt.cu tt.o -o $@

test_zobrist: tests/test_zobrist
test_tt: tests/test_tt

engine_test: tests/test_zobrist tests/test_tt
	./tests/test_zobrist
	./tests/test_tt

# GPU referee shared library (FFI ABI for Rust / Python / any-language callers).
# Bundles refcuda.cu + the search subsystem (search/time/zobrist/tt) into a
# .so so Rust crates and Python ctypes wrappers can import the GPU chess
# referee. uci.cu is intentionally excluded — its UCI run-loop pulls in
# engine.cu's main(); refcuda.cu provides a stub uci::stop_requested.
REFCUDA_SO       = librefcuda.so
REFCUDA_SO_SRCS  = refcuda.cu search.cu time.cu zobrist.cu tt.cu
REFCUDA_SO_DEPS  = $(REFCUDA_SO_SRCS) $(LIB_NAME) coord3d.cuh
NVCCFLAGS_SO     = $(ENGINE_ARCH_FLAGS) $(OPT_FLAGS) $(INCLUDES) $(CXX_STD) \
                   $(RDC_FLAGS) -Xcompiler -fPIC --shared

$(REFCUDA_SO): $(REFCUDA_SO_DEPS)
	$(NVCC) $(NVCCFLAGS_SO) $(REFCUDA_SO_SRCS) $(LIB_NAME) -o $@

librefcuda: $(REFCUDA_SO)

# Build dojo-ref Rust crate (links librefcuda.so + cudart). Workspace lives
# at ../../rust/Cargo.toml. cargo manages incremental compilation.
RUST_DIR  ?= ../../rust
CARGO     ?= cargo

dojo_ref: $(REFCUDA_SO)
	$(CARGO) build --manifest-path $(RUST_DIR)/Cargo.toml --release

dojo_ref_test: dojo_ref
	$(CARGO) run --manifest-path $(RUST_DIR)/Cargo.toml --release \
	    --example smoke

dojo_ref_play: dojo_ref
	$(CARGO) run --manifest-path $(RUST_DIR)/Cargo.toml --release \
	    --example play_game

clean:
	rm -f uci_test_engine engine engine_main.o uci_engine.o search.o time.o \
	      zobrist.o tt.o tests/test_zobrist tests/test_tt $(ENGINE_OBJS) \
	      $(LIB_NAME) $(REFCUDA_SO) refcuda_smoke \
	      engine_test_perft tests/test_makeunmake_symmetry
	-$(CARGO) clean --manifest-path $(RUST_DIR)/Cargo.toml 2>/dev/null || true

.PHONY: all engine uci_test arena_gpu_smoke arena_smoke engine_verify_long \
        skeleton_test clean engine_test test_zobrist test_tt \
        librefcuda dojo_ref dojo_ref_test dojo_ref_play
