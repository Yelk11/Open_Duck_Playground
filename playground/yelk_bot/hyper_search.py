"""Hyperparameter search using Ray Tune for Open Duck Mini V2 training.

This script runs multiple training trials in parallel using Ray Tune and reports
metrics back to Tune via a monkeypatched `progress_callback`.

Usage examples:

# quick test (no GPU):
python playground/open_duck_mini_v2/hyper_search.py --num_samples 2 --gpus_per_trial 0 --smaller

# run 4 trials using 1 GPU each:
python playground/open_duck_mini_v2/hyper_search.py --num_samples 4 --gpus_per_trial 1
"""

import os
import sys
import argparse
from types import SimpleNamespace

# allow importing local package
sys.path.append(os.getcwd())

import ray
from ray import tune

from playground.yelk_bot.runner import OpenDuckMiniV2Runner

# Avoid Ray overriding accelerator env var warning on num_gpus=0
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


def trainable(config):
    args = SimpleNamespace(
        output_dir=config.get("output_dir", "checkpoints"),
        num_timesteps=config.get("num_timesteps", 200000),
        env=config.get("env", "joystick"),
        task=config.get("task", "flat_terrain"),
        restore_checkpoint_path=config.get("restore_checkpoint_path", None),
    )

    runner = OpenDuckMiniV2Runner(args)

    # monkeypatch progress_callback to also report to Tune
    orig_cb = runner.progress_callback

    def cb(num_steps, metrics):
        try:
            orig_cb(num_steps, metrics)
        except Exception:
            pass
        # Sanitize metric keys for Tune (no slashes or spaces allowed as kwargs)
        flat = {k: float(v) for k, v in metrics.items()}
        flat["steps"] = int(num_steps)
        sanitized = {str(k).replace("/", "_").replace(" ", "_"): v for k, v in flat.items()}
        try:
            tune.report(**sanitized)
        except TypeError:
            # As a fallback, pass a single dict payload
            tune.report(metrics=sanitized)

    runner.progress_callback = cb

    # set simple overrides via environment variables (BaseRunner reads these)
    if "learning_rate" in config:
        os.environ["HP_LEARNING_RATE"] = str(config["learning_rate"])
    if "num_timesteps" in config:
        os.environ["HP_NUM_TIMESTEPS"] = str(config["num_timesteps"])

    runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--gpus_per_trial", type=float, default=1.0)
    parser.add_argument("--cpus_per_trial", type=int, default=2)
    parser.add_argument("--smaller", action="store_true", help="Use smaller timesteps for quick tests")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "num_timesteps": tune.choice([100000, 200000]) if args.smaller else tune.choice([200000, 500000, 1000000]),
        "env": tune.choice(["joystick", "standing"]),
    }

    # Use an explicit file:// URI for storage_path so pyarrow can resolve it.
    storage_uri = f"file://{os.path.abspath('ray_results')}"

    analysis = tune.run(
        trainable,
        config=search_space,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        num_samples=args.num_samples,
        name="open_duck_hyper",
        storage_path=storage_uri,
        reuse_actors=False,
    )

    best = analysis.get_best_trial(metric="eval/episode_reward", mode="max", scope="last")
    print("Best trial:", best)
    print("Best config:", best.config)
    print("Result dir:", analysis.get_best_logdir(metric="eval/episode_reward", mode="max"))
