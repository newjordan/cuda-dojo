# Strict Gate Port Manifest

Port date: 2026-05-03

Source lab:

- Path: `/home/frosty40/VC_V1/.docker_labs/dojo_conversion_lab`
- Branch observed: `codex/package-cuda-dojo-engine`
- Head observed: `bb865bb`
- State: dirty/generated lab tree, so this port is a frozen artifact import, not
  a clean upstream-source import.

Tracked gate files:

| File | SHA256 | Note |
|---|---:|---|
| `scripts/validate_cpu_gpu_accuracy.mjs` | `b7f9894baad6f934b1ac309da575e47345ad2df42d3eba0c9f2ada871762fe91` | Strict validator, modified to prefer tracked corpus and timeout `gpu_forge`. |
| `scripts/run_cpu_gpu_accuracy_matrix.mjs` | `a7a2c0b50cdc4af504d45c498dfc3b44f1ed9a2c6682e4d6d1665ff361907b7e` | Five-fighter matrix, modified to pass timeout. |
| `cuda/gpu_forge.cu` | `11c5bb7af1ab860e843269b49900a98c2ba8be0b7b4c616bf7800a9139585271` | GPU comparator copied byte-identical from source lab. |
| `cuda/generated/dojo_active_fighter_legacy.h` | `a078ce42b745396f046a8c3b23d310e3c07ae558e78e2f0dac22344079c64496` | Active fallback header copied from source lab. |
| `dojo_runtime.js` | `6a86b572efd915e8bcb78b40448e9d45bb6d3365a096486950c1545f2c09c5e2` | CPU fighter wrapper/runtime copied from source lab. |
| `dojo_chess.js` | `045b5128579b9d3419965d4d1737c39511a562e107f458c15335bc2e957c1a9b` | Legal move/corpus helper copied from source lab. |
| `fighter_accuracy/corpus/cuda_dojo_batch_latest.json` | `170583e396d94d53a0d3161b6b27497bf36304941d0baa827237f5c67f7ac7c0` | Tracked copy of prior batch corpus. |

Tracked fighter files:

| File | SHA256 |
|---|---:|
| `variants/razor_x.js` | `18737caef8aa8119c44f1e3560bf77dde4ba906bf94eb0e83f09b506f6258007` |
| `variants/razor_x.cuda_fighter_blob.json` | `c8074b3f50ed3dc24d7cf9b2fdb6bb1a6898116b7085c49a944226de40c3a232` |
| `variants/queensguard.js` | `fd2481ae9dfa7c5a51ea0ee71a9bfa4aaf8fa283dc243e6b28d170f0b59725a4` |
| `variants/queensguard.cuda_fighter_blob.json` | `c9b12315c92ed47b0b1c9de4ef5e1e83e9cd6c4beb978b50a2e4538fca5d6453` |
| `variants/firebird.js` | `ad2f32430f23add8df67c76004998f48bbe910c13c45fb396838ff3a6d341e90` |
| `variants/firebird_src_dojo_runtime1.cuda_fighter_blob.json` | `14d2c2090ed7313ad1ef25a41035e9a40b83b72ca224377af93290853bf80610` |
| `variants/fortress.js` | `722a9a4441a1692a5302b7a0eb2433040e8d53e789eed0128296026cbee86bdc` |
| `variants/fortress_src_dojo_runtime1.cuda_fighter_blob.json` | `8dfa3c352be8600f62c89e8d617d6ad8b37a3e7b67bb3dbebeb854f370320b24` |
| `variants/razorblade_ii.js` | `578fda494351b367a7361d5412c1e06f60187b70586d16dd7f24437eb370b3a9` |
| `variants/razorblade_ii_dojo.cuda_fighter_blob.json` | `20185ce843c0d3672bb68e0d842b33503b20b081b9f8b1cf9dab9746564d9a7c` |
| `variants/razor_traincar_backend.js` | `ee88356d6824b1c258c3be61e47e3ce3d5f46070088c007c86abbc9f5aca86bd` |
| `variants/morph_target_brain.js` | `feb41c9714ef78e5260f8255f4e2851676faadcaace7521c6e326d2cd983b533` |
| `frostd4d/variants/the_un.js` | `275966ca803c2c97162ccacacb60d1d2a35fda1351a8a9ad16c599eda587d875` |

Smoke condition:

- Command: `node scripts/run_cpu_gpu_accuracy_matrix.mjs --samples 4 --configs 4 --sims 4 --timeout-ms 30000 --min-accuracy 0.55 --min-coverage 0.75`
- Result: 0/5 pass.
- Receipt copy: `fighter_accuracy/baselines/tracked_gate_smoke_cpu_gpu_accuracy_matrix_2026-05-03.md`
- Notes: Firebird and Fortress hit the 30 second `gpu_forge` timeout at this smoke setting. That is recorded as a failure, not a pass.
