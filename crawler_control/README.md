# Crawler Control

This directory treats `~/sota_crawler` as a reusable research control plane.

The goal is not to copy one transformer block into chess. The goal is to reuse:

- purpose-specific runner roles
- explicit seed manifests
- launch validation
- later, layered promotion and lane-backed state

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
training runs yet.
