import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import subprocess
import time
import gymnasium as gym
from reinforcement_learning import train_rl_model
from src.hybrid_agent import CupheadAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def register_env():
    gym.envs.registration.register(
        id='CupheadEnv-v0',
        entry_point='src.environment:CupheadEnv',
        max_episode_steps=2000,
    )


def launch_cuphead():
    cuphead_path = (r"C:\Users\gokua\Downloads\Cuphead.v1.3.4.Incl.ALL.DLC\Cuphead.v1.3.4.Incl.ALL.DLC\Cuphead.v1.3.4"
                    r".Incl.ALL.DLC\Cuphead.exe")
    subprocess.Popen(cuphead_path)
    time.sleep(10)


if __name__ == "__main__":
    logging.info("Launching Cuphead...")
    launch_cuphead()
    register_env()

    # Initialize environment
    env = gym.make('CupheadEnv-v0')

    logging.info("Starting training of Cuphead AI...")
    model = train_rl_model(env)
    logging.info("Reinforcement learning model trained.")

    logging.info("Starting Cuphead AI...")
    agent = CupheadAI()
    agent.play()
    logging.info("Playing.")
