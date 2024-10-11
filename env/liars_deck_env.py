# env/liars_deck_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class LiarsDeckEnv(gym.Env):
    """
    Custom Environment for Liar's Deck game.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, num_players=2, seed=None):
        super(LiarsDeckEnv, self).__init__()
        
        self.num_players = num_players
        self.card_types = ['King', 'Queen', 'Ace']
        self.deck_composition = {
            'King': 6,
            'Queen': 6,
            'Ace': 6,
            'Joker': 2
        }
        self.max_cards_per_declaration = 3
        self.joker = 'Joker'
        self.seed = seed
        
        # Action space: Declare a card type and the number of cards (1 to 3)
        # For simplicity, action is encoded as (card_type * 3 + number_of_cards -1)
        # card_type: 0-King, 1-Queen, 2-Ace
        # number_of_cards: 1, 2, 3
        self.action_space = spaces.Discrete(len(self.card_types) * self.max_cards_per_declaration)
        
        # Observation space
        # player_hand: counts of each card type including Jokers
        # current_declaration: (card_type, number_of_cards)
        # previous_player_action: same as current_declaration
        # bullets_remaining: Discrete(6)
        self.observation_space = spaces.Dict({
            'player_hand': spaces.Box(low=0, high=6, shape=(4,), dtype=np.int8),
            'current_declaration': spaces.Box(low=0, high=3, shape=(2,), dtype=np.int8),
            'previous_player_action': spaces.Box(low=0, high=3, shape=(2,), dtype=np.int8),
            'bullets_remaining': spaces.Discrete(6)
        })
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize deck
        self.full_deck = []
        for card, count in self.deck_composition.items():
            self.full_deck += [card] * count
        
        # Shuffle deck
        random.shuffle(self.full_deck)
        
        # Initialize player hands
        self.player_hands = []
        for _ in range(self.num_players):
            hand = self._draw_cards(5)
            self.player_hands.append(hand)
        
        self.current_player = 0
        self.previous_player_action = np.array([0, 0], dtype=np.int8)  # No previous action
        self.current_declaration = np.array([0, 1], dtype=np.int8)  # Default declaration
        self.bullets_remaining = 6
        self.done = False
        self.winner = None
        self.info = {}
        
        return self._get_obs()
    
    def _draw_cards(self, num):
        drawn = []
        for _ in range(num):
            if len(self.full_deck) == 0:
                break
            drawn.append(self.full_deck.pop())
        return drawn
    
    def step(self, action):
        """
        Execute one action in the environment.
        Action: Declare a card type and number of cards.
        """
        if self.done:
            return self._get_obs(), 0, self.done, self.info
        
        # Decode action
        card_type_idx = action // self.max_cards_per_declaration
        num_cards = (action % self.max_cards_per_declaration) + 1
        declared_card = self.card_types[card_type_idx]
        
        # Current declaration
        self.current_declaration = np.array([card_type_idx, num_cards], dtype=np.int8)
        
        # Get current player's hand
        player_hand = self.player_hands[self.current_player]
        
        # Play actual cards (with potential bluffing)
        actual_cards = []
        for _ in range(num_cards):
            if len(player_hand) == 0:
                break
            # Simple strategy: try to play declared_card if possible
            if declared_card in player_hand:
                player_hand.remove(declared_card)
                actual_cards.append(declared_card)
            elif self.joker in player_hand:
                player_hand.remove(self.joker)
                actual_cards.append(self.joker)
            else:
                # Bluff by playing a random card
                card = random.choice(player_hand)
                player_hand.remove(card)
                actual_cards.append(card)
        
        # Check if the player is bluffing
        is_bluff = False
        for card in actual_cards:
            if card != declared_card and card != self.joker:
                is_bluff = True
                break
        
        # Next player decides to call bluff or not
        next_player = (self.current_player + 1) % self.num_players
        call_bluff = self._decide_call_bluff(next_player, declared_card, num_cards)
        
        reward = 0
        
        if call_bluff:
            if is_bluff:
                # Previous player gets caught bluffing
                # Consequences: Russian Roulette
                survival = self._russian_roulette()
                if survival:
                    reward = -1  # Penalty for bluffing
                else:
                    reward = 100  # Reward for catching bluff
                    self.done = True
                    self.winner = next_player
            else:
                # Accuser was wrong
                # Accuser faces consequences
                survival = self._russian_roulette()
                if survival:
                    reward = -1  # Penalty for wrong call
                else:
                    reward = 100  # Reward if accuser dies
                    self.done = True
                    self.winner = self.current_player
        else:
            # No bluff called, continue game
            reward = 0
        
        # Check if current player has no cards left
        if len(player_hand) == 0:
            self.done = True
            self.winner = self.current_player
            reward = 100  # Reward for winning
        
        # Update previous action
        self.previous_player_action = self.current_declaration.copy()
        
        # Rotate to next player
        self.current_player = next_player
        
        return self._get_obs(), reward, self.done, self.info
    
    def _decide_call_bluff(self, player, declared_card, num_cards):
        """
        Simple heuristic for deciding whether to call a bluff.
        This can be enhanced to incorporate more sophisticated logic or be part of the agent's strategy.
        For now, we simulate a random decision.
        """
        # Placeholder: 30% chance to call a bluff
        return random.random() < 0.3
    
    def _russian_roulette(self):
        """
        Simulate Russian Roulette.
        There is 1 bullet in 6 chambers.
        """
        if self.bullets_remaining <= 0:
            # All bullets have been used; automatic survival
            return True
        chamber = random.randint(1, 6)
        self.bullets_remaining -= 1
        return chamber != 1  # Survives if chamber is not 1
    
    def _get_obs(self):
        # Encode player hand as counts of each card type
        card_counts = [0, 0, 0, 0]  # King, Queen, Ace, Joker
        for card in self.player_hands[self.current_player]:
            if card == 'King':
                card_counts[0] += 1
            elif card == 'Queen':
                card_counts[1] += 1
            elif card == 'Ace':
                card_counts[2] += 1
            elif card == 'Joker':
                card_counts[3] += 1
        
        return {
            'player_hand': np.array(card_counts, dtype=np.int8),
            'current_declaration': self.current_declaration,
            'previous_player_action': self.previous_player_action,
            'bullets_remaining': self.bullets_remaining
        }
    
    def render(self, mode='human'):
        print(f"Player {self.current_player}'s turn.")
        print(f"Hand: {self.player_hands[self.current_player]}")
        print(f"Declared: {self.card_types[self.current_declaration[0]]} x {self.current_declaration[1]}")
        print(f"Previous Action: {self.card_types[self.previous_player_action[0]]} x {self.previous_player_action[1]}")
        print(f"Bullets Remaining: {self.bullets_remaining}")
        print("-" * 30)
    
    def close(self):
        pass
