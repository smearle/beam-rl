"""
Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

from beam_env import BeamEnv

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=1000,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=10000000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=0.1,
    help="Reward at which we stop training.")
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")
parser.add_argument(
    "--n-cpu",
    type=int,
    default=7,
)
parser.add_argument(
    "--evaluate",
    action="store_true",
    help="Load trained model for evaluation.")
parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Location of a saved model for re-loading",
)

#def train_ppo(config, reporter):
#    agent = PPOTrainer(config)
##   agent.restore("/path/checkpoint_41/checkpoint-41")  # continue training
#    # training curriculum, start with phase 0
##   phase = 0
##   agent.workers.foreach_worker(
##           lambda ev: ev.foreach_env(
##               lambda env: env.set_phase(phase)))
##   episodes = 0
#    i = 0
#    while True:
#        result = agent.train()
#        if reporter is None:
#            continue
#        else:
#            reporter(**result)
#        if i % 10 == 0: #save every 10th training iteration
#            checkpoint_path = agent.save()
#            print("checkpoint saved at f{checkpoint_path)}")
#        i+=1
#        #you can also change the curriculum here

from beam_env import NeuralAgent

def evaluate(args, config):
    config["env_config"]["render"] = True
    env = config["env"](config["env_config"])
    agent = PPOTrainer(config)
    agent.restore(args.model_path)
    agent.train()
#   nn_agent = NeuralAgent(env, agent)
#   nn_agent.play_episode()

def train(args, config):
    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        # use fixed learning rate instead of grid search (needs tune)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPOTrainer(config=ppo_config, env=BeamEnv)
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps or \
                    result["episode_reward_mean"] >= args.stop_reward:
                break
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run(args.run, keep_checkpoints_num=10, checkpoint_freq=10, config=config, stop=stop)

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=args.local_mode)

    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    # If we don't see any GPUs via the above, we'll manually set num_gpus to 1 if cuda is available
    if num_gpus == 0 and torch.cuda.is_available():
        num_gpus = 1

    config = {
        "env": BeamEnv,  # or "corridor" if registered above
        "env_config": {
            "n_beams": 16,
            "n_paths": 4,
            "render": False,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": num_gpus,
        "model": {
            "use_lstm": True,
#           "custom_model": "my_model",
#           "vf_share_layers": True,
        },
        "num_workers": args.n_cpu,  # parallelism
        "framework": args.framework,
    }

    stop = {
#       "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
#       "episode_reward_mean": args.stop_reward,
    }

    if args.evaluate:
        evaluate(args, config)
    else:
        train(args, config)
