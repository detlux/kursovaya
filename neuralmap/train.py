"""Train a neural planner from a YAML config.

Usage:
    python train.py --config configs/dqn.yaml
    python train.py --config configs/iql.yaml

Produces:
    <out_dir>/training_log.csv
    <out_dir>/dqn_shared.pt          (for DQN)
    <out_dir>/iql_agent_{i}.pt       (for IQL)
"""
import argparse
import os
import sys

# allow running from project root: `python train.py ...`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import train_dqn_shared, train_iql


def _load_yaml(path: str) -> dict:
    """Minimal YAML loader (no PyYAML dependency)."""
    cfg: dict = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            key, _, val = line.partition(":")
            key = key.strip(); val = val.strip()
            if val == "":
                continue
            # type coercion
            try:
                cfg[key] = int(val)
            except ValueError:
                try:
                    cfg[key] = float(val)
                except ValueError:
                    cfg[key] = val
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = _load_yaml(a.config)
    out_dir = cfg.get("out_dir", "results/run")
    algo = cfg.get("algo", "dqn")
    print(f"== training {algo} ==")
    print(f"  grid {cfg['grid_width']}x{cfg['grid_height']}, "
          f"N={cfg['num_agents']}, episodes={cfg['episodes']}")
    if algo == "dqn":
        train_dqn_shared(cfg, out_dir)
    elif algo == "iql":
        train_iql(cfg, out_dir)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    print("done.")


if __name__ == "__main__":
    main()
