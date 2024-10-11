# scripts/train.py

import os
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from agents.dqn_agent import create_agent
from env.liars_deck_env import LiarsDeckEnv
from utils.helpers import load_config, parse_args, merge_configs

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration file
    config = load_config(args['config'])

    # Merge command-line arguments with configuration
    config = merge_configs(config, args)

    # Extract configurations
    env_config = config.get('environment', {})
    agent_config = config.get('agent', {})
    training_config = config.get('training', {})

    # Create the environment
    env = LiarsDeckEnv(num_players=env_config.get('num_players', 2))

    # Create the agent
    model = create_agent(env, agent_config)

    # Ensure the models and logs directories exist
    os.makedirs(os.path.dirname(training_config.get('model_save_path', './models/')), exist_ok=True)
    os.makedirs(training_config.get('log_path', './logs/'), exist_ok=True)

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.get('checkpoint_freq', 1000),
        save_path=os.path.join(training_config.get('model_save_path', './models/'), 'checkpoints/'),
        name_prefix='dqn_liars_deck'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(training_config.get('model_save_path', './models/'), 'best_model/'),
        log_path=os.path.join(training_config.get('log_path', './logs/'), 'eval/'),
        eval_freq=training_config.get('eval_freq', 5000),
        deterministic=True,
        render=env_config.get('render', False)
    )

    # Train the agent
    model.learn(
        total_timesteps=training_config.get('total_timesteps', 100000),
        callback=[checkpoint_callback, eval_callback]
    )

    # Save the final model
    final_model_path = training_config.get('model_save_path', './models/dqn_liars_deck_final.zip')
    model.save(final_model_path)
    print(f"Training completed and model saved to {final_model_path}.")

if __name__ == "__main__":
    main()
