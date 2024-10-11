# scripts/evaluate.py

import os
from agents.dqn_agent import create_agent
from utils.helpers import load_config, parse_args, merge_configs, evaluate_agent_performance
from env.liars_deck_env import LiarsDeckEnv

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration file
    config = load_config(args['config'])

    # Merge command-line arguments with configuration
    config = merge_configs(config, args)

    # Extract configurations
    env_config = config.get('environment', {})
    evaluation_config = config.get('evaluation', {})
    agent_config = config.get('agent', {})

    # Create the environment
    env = LiarsDeckEnv(num_players=env_config.get('num_players', 2))

    # Load the trained model
    model_path = evaluation_config.get('model_path', './models/dqn_liars_deck_final.zip')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    model = create_agent(env, agent_config, model_path=model_path)

    # Evaluate the agent
    episodes = evaluation_config.get('episodes', 100)
    win_rate = evaluate_agent_performance(env, model, episodes=episodes)
    print(f"Agent won {win_rate * 100:.2f}% of {episodes} episodes.")

if __name__ == "__main__":
    main()
