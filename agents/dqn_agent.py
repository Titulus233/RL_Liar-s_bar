# agents/dqn_agent.py

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from env.liars_deck_env import LiarsDeckEnv

def create_agent(env, agent_config, model_path=None):
    """
    Create or load a DQN agent based on the provided configuration.
    """
    if model_path:
        model = DQN.load(model_path, env=env)
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=agent_config.get("learning_rate", 1e-3),
            buffer_size=agent_config.get("buffer_size", 10000),
            learning_starts=agent_config.get("learning_starts", 1000),
            batch_size=agent_config.get("batch_size", 32),
            gamma=agent_config.get("gamma", 0.99),
            train_freq=agent_config.get("train_freq", 4),
            target_update_interval=agent_config.get("target_update_interval", 1000),
            exploration_fraction=agent_config.get("exploration_fraction", 0.1),
            exploration_final_eps=agent_config.get("exploration_final_eps", 0.02),
            tensorboard_log=agent_config.get("tensorboard_log", "./dqn_tensorboard/")
        )
    return model
