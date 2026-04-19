# Crawler Control

This directory is an experimental control-plane surface for structured calibration runs.

The goal is not to embed crawler internals into the chess engine. The current use is narrower:

- purpose-specific runner roles
- explicit seed manifests
- launch validation
- later, layered promotion and lane-backed state if the probes prove useful

## Files

- [purpose_profiles.json](./purpose_profiles.json): crawler roles and their intended job
- [build_seed_manifest.py](./build_seed_manifest.py): materialize concrete seed candidates
- [launch_probe.py](./launch_probe.py): dry-run canonical script runners

## Usage

```bash
cd crawler_control
python3 build_seed_manifest.py
python3 launch_probe.py
```

`launch_probe.py` only executes `runner_mode = script` candidates, and it uses
`--dry-run` so we can validate the launch surface without committing to long
training runs yet. This is a measurement/control utility, not part of the live
engine runtime.
