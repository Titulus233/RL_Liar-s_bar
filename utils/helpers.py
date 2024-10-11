# utils/helpers.py

import json
import argparse
import os
import numpy as np
from env.liars_deck_env import LiarsDeckEnv

def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Liar's Deck RL Agent Configuration")

    # General settings
    parser.add_argument('--config', type=str, default='config/config.json',
                        help='Path to the configuration file.')

    # Environment settings
    parser.add_argument('--num_players', type=int, help='Number of players in the environment.')

    # Agent settings
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the agent.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--gamma', type=float, help='Discount factor.')

    # Training settings
    parser.add_argument('--total_timesteps', type=int, help='Total timesteps for training.')
    parser.add_argument('--checkpoint_freq', type=int, help='Frequency of saving checkpoints.')
    parser.add_argument('--eval_freq', type=int, help='Frequency of evaluation during training.')
    parser.add_argument('--model_save_path', type=str, help='Path to save the trained model.')
    parser.add_argument('--log_path', type=str, help='Path to save logs.')

    # Evaluation settings
    parser.add_argument('--episodes', type=int, help='Number of episodes for evaluation.')
    parser.add_argument('--model_path', type=str, help='Path to the trained model for evaluation.')

    args = parser.parse_args()
    return vars(args)  # Return as a dictionary

def merge_configs(default_config, args):
    """
    Merge command-line arguments with the default configuration.
    Command-line arguments override the config file.
    """
    for key, value in args.items():
        if value is not None:
            if key in default_config:
                default_config[key] = value
            else:
                # Handle nested keys
                for section in default_config:
                    if key in default_config[section]:
                        default_config[section][key] = value
                        break
    return default_config

def evaluate_agent_performance(env, model, episodes=100):
    """
    Evaluate the trained agent over a number of episodes.
    Returns the win rate.
    """
    wins = 0
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                if reward == 100:
                    wins += 1
    win_rate = wins / episodes
    return win_rate
