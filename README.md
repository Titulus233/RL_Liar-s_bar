# Liar’s Deck Reinforcement Learning Agent

**Liar’s Deck** is a bluffing and social deduction game where the goal is to get rid of your cards while outsmarting and deceiving other players. This project implements a reinforcement learning (RL) agent to find the optimal strategy for **Liar’s Deck** using the `gymnasium` library and `Stable Baselines3`.

## Project Structure
liars_deck_rl/
├── config/
│   └── config.json
├── env/
│   ├── __init__.py
│   └── liars_deck_env.py
├── agents/
│   ├── __init__.py
│   └── dqn_agent.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── requirements.txt
└── README.md

