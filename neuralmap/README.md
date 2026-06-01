# neuralmap

A small platform for the controlled comparison of neural and classical
multi-agent planners. Three planners are implemented behind a common
interface:

- **DQN** (Deep Q-Network) with a Q-network shared across all agents;
- **IQL** (Independent Q-Learning) with one independent Q-network per agent;
- **Centralized A***, a symbolic reference baseline.

The environment is Cooperative Navigation on an 8x8 grid with 3 agents
and 3 landmarks.

## Installation

```
pip install -r requirements.txt
```

Python 3.10+ and PyTorch (CPU is sufficient) are required.

## Full pipeline

```
python compare.py
```

Steps performed by the script:

1. Train DQN (~5 min) -> `results/dqn/dqn_shared.pt`, `training_log.csv`
2. Train IQL (~14 min) -> `results/iql/iql_agent_{0,1,2}.pt`, `training_log.csv`
3. Evaluate each of the three planners on 50 episodes with matched seeds
4. Build figures: `training_dqn.{pdf,png}`, `training_iql.{pdf,png}`,
   `success_rate.{pdf,png}`, `decision_time.{pdf,png}`, `qheatmap.{pdf,png}`
5. Write the summary table `results/summary.csv`
6. Print success rates, mean decision times, and p-values for the
   pairwise Mann-Whitney U and two-proportion z-tests

## Selective runs

```
# only DQN training (IQL untouched)
python compare.py --only-dqn

# only IQL training (DQN untouched, reusing existing weights)
python compare.py --only-iql

# skip training entirely and re-evaluate with existing weights
python compare.py --skip-train
```

## Single-step entry points

```
# train one planner from a YAML config
python train.py --config configs/dqn.yaml
python train.py --config configs/iql.yaml

# evaluate one planner
python eval.py --algo astar --episodes 50 --out results/astar/eval.csv
python eval.py --algo dqn   --weights results/dqn/dqn_shared.pt \
               --episodes 50 --out results/dqn/eval.csv
python eval.py --algo iql   --weights-dir results/iql \
               --episodes 50 --out results/iql/eval.csv
```

## Layout

```
neuralmap/
  core/                       abstract interfaces and the simulation engine
    planner.py                Planner, NeuralPlanner
    environment.py            Environment, StepResult
    engine.py                 SimulationEngine
  env/
    grid_env.py               Cooperative Navigation grid environment
  planners/
    base.py                   Planner / NeuralPlanner abstract classes
    astar_planner.py          centralized A*
    dqn_planner.py            DQN (shared Q-network)
    iql_planner.py            IQL (independent Q-networks)
  training/
    replay_buffer.py          FIFO experience replay
    trainer.py                training loops for DQN and IQL
  evaluation/
    runner.py                 N-episode runner
    metrics.py                per-episode metric container
    stats.py                  Mann-Whitney U, Wilson CI, two-proportion z
  visualization/
    plot_training.py          reward and TD-loss curves from a CSV
    plot_results.py           success-rate bars and decision-time box plot
    plot_qheatmap.py          Q-value heat map
  configs/
    dqn.yaml                  DQN training hyperparameters
    iql.yaml                  IQL training hyperparameters
    astar.yaml                A* evaluation parameters
  train.py                    entry point: training
  eval.py                     entry point: single-planner evaluation
  compare.py                  end-to-end pipeline
  requirements.txt
```

## Reproducibility

Every run is determined by a YAML configuration plus the seed in the
`experiment.seed` field. Episode `i` uses `seed + i`. The simulator and
the evaluation-mode neural planners are deterministic for a fixed seed.

## Extending

Adding a new planner requires only a subclass of `Planner` with a `plan`
method; the runner picks it up through `compare.py` or `eval.py` once it
is registered in `evaluation/runner.py`. Adding a new environment
requires a class with `reset`, `step`, `observe`, and `is_terminal`
methods that mirrors the contract of `GridEnv`.
