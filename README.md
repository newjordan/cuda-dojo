# CUDA Dojo

```text
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠴⠛⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⠖⠋⠁⠀⠀⠀⠈⠙⠲⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠛⢦⣄⠀⠀⠀⠀⠀⣠⡴⠛⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣠⣤⣄⠀⠀⠀⢸⠀⠀⠈⠛⢦⣤⡴⠛⠁⠀⠀⡇⠀⠀⠀⣠⣤⣤⡀⠀⠀⠀⠀
⠀⣠⡴⠞⠉⠀⣿⠈⠛⠶⣄⣸⡇⠀⠀⠀⠀⣿⠀⠀⠀⠀⢰⣇⣠⠶⠛⠁⣿⠀⠉⠳⢦⣄⡀
⣿⠁⠀⠀⠀⠀⢿⠀⠀⠀⠈⠙⣧⣄⠀⠀⠀⣿⠀⠀⠀⣠⣼⠋⠁⠀⠀⠀⡿⠀⠀⠀⠀⠈⣿
⢻⠀⠀⠀⠀⠀⣸⡀⠀⠀⠀⠀⣿⠈⠛⠶⣄⣿⣠⠶⠛⠁⣿⠀⠀⠀⠀⢀⣇⠀⠀⠀⠀⠀⡿
⢸⡀⣀⣤⠶⠛⠉⠙⠳⢤⣀⠀⣿⠀⠀⠀⠈⣿⠁⠀⠀⠀⣿⠀⣀⡤⠞⠋⠉⠛⠶⣤⣀⠀⡇
⠘⠿⣍⡀⠀⠀⠀⠀⠀⠀⢉⣻⣿⣄⡀⠀⠀⣿⠀⠀⢀⣠⣿⣟⡉⠀⠀⠀⠀⠀⠀⢀⣩⠿⠃
⠀⠀⠈⠙⠲⣤⣀⣠⡴⠞⠋⠁⠀⠈⠙⠳⢤⣿⣤⠞⠋⠁⠀⠈⠙⠳⢦⣄⣀⣤⠖⠋⠁⠀⠀
⠀⠀⢀⣠⠴⠛⠉⠙⠳⢦⣄⡀⠀⢀⣠⡴⠛⣿⠛⢦⣄⡀⠀⢀⣠⡴⠞⠋⠉⠛⠦⣄⡀⠀⠀
⢠⣾⣋⠁⠀⠀⠀⠀⠀⠀⣈⣽⣿⠋⠁⠀⠀⣿⠀⠀⠈⠙⣿⣯⣁⠀⠀⠀⠀⠀⠀⠈⣙⣷⡄
⢸⠀⠉⠛⠶⣤⣀⣠⡴⠚⠉⠀⣿⠀⠀⠀⢀⣿⡀⠀⠀⠀⣿⠀⠉⠓⢦⣄⣀⣤⠶⠛⠉⠀⡇
⣼⠀⠀⠀⠀⠀⢹⠁⠀⠀⠀⠀⣿⢀⣤⠖⠋⣿⠙⠲⣤⡀⣿⠀⠀⠀⠀⠈⡏⠀⠀⠀⠀⠀⣷
⣿⡀⠀⠀⠀⠀⣾⠀⠀⠀⢀⣠⡟⠋⠀⠀⠀⣿⠀⠀⠀⠙⢻⣄⡀⠀⠀⠀⣿⠀⠀⠀⠀⢀⣿
⠀⠙⠳⢦⣄⠀⣿⢀⣤⠶⠋⢹⡇⠀⠀⠀⠀⣿⠀⠀⠀⠀⠸⡏⠙⠶⣤⡀⣿⠀⣀⡴⠞⠋⠀
⠀⠀⠀⠀⠈⠙⠛⠋⠀⠀⠀⢸⠀⠀⢀⣤⠞⠛⠳⣤⡀⠀⠀⡇⠀⠀⠀⠉⠛⠋⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣤⠞⠋⠀⠀⠀⠀⠀⠙⠳⣤⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠶⣄⡀⠀⠀⠀⢀⣠⠴⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠲⣤⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

GPU-first chess research repo extracted from a private working academy so collaborators can help directly on architecture, search, training, and engine correctness.

The current emphasis is:
- fully playable GPU chess engine infrastructure
- GPU-native search experiments in `cuda/chess_mcts.cu` and `cuda/engine/`
- external-oracle sigma baselines with Stockfish and Lozza
- training-corpus construction and disagreement-driven probe loops

## Current Focus
- stabilize GPU search quality with measurement, not guesswork
- close the gap between our engine choices and external oracle baselines
- harden the modular UCI engine in `cuda/engine/`
- build GPU-native search and training loops that survive real regression batteries
- make opening and opening-into-middlegame allocation a first-class design variable

## Next Steps
- expand the sigma oracle battery beyond small smokes and rank worst disagreement bands
- use those disagreement rows as direct engine regression probes
- improve root search discrimination in `chess_mcts` on severe sigma positions
- tighten training-corpus weighting around oracle disagreement and pressure regions
- separate publishable architecture from local runtime exhaust so outside contributors can move quickly

## Ultimate Goals
- a genuinely strong GPU-first chess engine, not a CPU engine with CUDA bolted on
- a reusable dojo factory that turns every run into better search, better data, or better calibration
- opening, middlegame, and endgame allocation treated as measurable budget decisions
- architecture that scales from local GPU experimentation to larger accelerator targets cleanly
- a public collaboration surface where contributors can improve engine strength without needing the private Academy repo

## Main Areas
- `cuda/`: GPU engine, search kernels, audits, ablations, oracle loop
- `cuda/engine/`: modular UCI engine baseline
- `gpu_spine/`: Stockfish-backed book/oracle infrastructure
- `trainers/lozza/`: embedded Lozza baseline source
- `crawler_control/`: crawler-inspired control-plane experiments
- `schemas/`: JSON schemas for dojo artifacts

## Quick Start
1. Install Stockfish into `trainers/stockfish/stockfish_bin`:

```bash
bash ./install_stockfish.sh
```

2. Build the CUDA tools you want to work on:

```bash
make -C cuda chess_mcts
make -C cuda/engine engine
```

3. Run the oracle sigma smoke:

```bash
make -C cuda oracle_sigma_smoke
make -C cuda oracle_sigma_probe
```

## Notes
- This public repo is intentionally narrower than the private Academy repo.
- Runtime exhaust, generated fighters, and private Battle Academy integration are excluded here.
- Some scripts still assume you have CUDA, Python, and Stockfish available locally.
