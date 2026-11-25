# CPU entry points. CUDA builds stay in cuda/ and cuda/engine/.
# `make` prints this help. It does not build or run the engine.

.DEFAULT_GOAL := help

.PHONY: help test host host-test

help:
	@echo "CPU checks (no CUDA):"
	@echo "  make test"
	@echo "CUDA machine:"
	@echo "  make -C cuda/engine engine CUDA_SINGLE_GPU_TARGET=native"
	@echo "  make -C cuda chess_mcts CUDA_SINGLE_GPU_TARGET=native"
	@echo "  python3 -m pytest -q -m gpu"

host:
	$(MAKE) -C cuda/engine host

host-test:
	$(MAKE) -C cuda/engine host-test

test: host
	python3 -m pytest -q -rs
