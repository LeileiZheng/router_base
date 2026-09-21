import argparse
import os
import json
import yaml
from tasks.runner import BenchmarkRunner
from tasks.evaluator import BenchmarkEvaluator
from tasks import mmlu_pro, gsm_hard
# SRDD / CW are temporarily disabled in the sequential-only setup.
# from tasks import srdd, creative_writing


EVALUATION_MODES = frozenset({"validation", "test"})


def configure_policy_for_mode(config, task, mode):
    """Apply the dataset/checkpoint mode and its training safety policy."""
    config["dataset_name"] = task
    config["dataset_mode"] = mode
    config['paths']["checkpoint_path"] = f"checkpoint/sequential/{task}_{mode}"
    config['paths']["model_path"] = f"checkpoint/sequential/{task}_{mode}/policy_net_latest.pt"

    # This CLI exposes evaluation modes only.  They must never update the
    # policy, even if an old policy.json had training=true.
    if mode in EVALUATION_MODES:
        config.setdefault("training", {})["training"] = False
    return config


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    # parser.add_argument("--task", default='MMLU-Pro', choices=["MMLU-Pro", "gsm-hard", "SRDD", "CW"])
    parser.add_argument("--task", default='MMLU-Pro', choices=["MMLU-Pro", "gsm-hard"])
    parser.add_argument("--mode", default='validation', choices=["validation", "test"])
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--index", type=int, default=-1)
    # Old default limited every run to one sample.
    # parser.add_argument("--data_limit", type=int, default=1)
    # Process the full dataset by default; pass --data_limit N to limit a run.
    parser.add_argument("--data_limit", type=int, default=None)
    parser.add_argument("--personas", type=str, default="puppeteer/personas/personas_sequential.jsonl")

    args = parser.parse_args()

    # load global config
    with open("puppeteer/config/global.yaml", "r") as f:
        global_config = yaml.safe_load(f)

    runner = BenchmarkRunner(args.personas, global_config)
    evaluator = BenchmarkEvaluator()

    results_dir = os.path.join(os.getcwd(), "results", f"{args.task}_{args.mode}")
    os.makedirs(results_dir, exist_ok=True)

    # change policy.json
    config_path = "puppeteer/config/policy.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    configure_policy_for_mode(config, args.task, args.mode)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    task_map = {
        "MMLU-Pro": mmlu_pro.run,
        "gsm-hard": gsm_hard.run,
        # "SRDD": srdd.run,
        # "CW": creative_writing.run,
    }

    if args.task in task_map:
        task_map[args.task](runner, evaluator, results_dir, args.mode, args.data_limit)
    else:
        print(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()
