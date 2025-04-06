import gymnasium as gym
import numpy as np
from game import Game

class CardGameEnv(gym.Env):
    def __init__(self):
        super(CardGameEnv, self).__init__()
        
        # Define action space: 0 or 1 (binary)
        self.action_space = gym.spaces.Discrete(60)  # 114 actions
        
        # Define observation space: 540-dimensional vector
        # Concatenation of 5 x 108
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(270,), dtype=np.float32
        )
        
        # Your game logic object (replace with your actual game class/instance)
        self.game = Game()  # Replace with your game implementation
        self.current_player_turn = 0  # Assuming single-player for now
        self.step_count = 0
        self.max_steps = 1000  # 1000 decisions per game
        
    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        self.step_count = 0
        self.game.reset()  # Replace with your game's reset method
        
        # Get initial state from your game
        game_state = self.game.get_game_state()  # Replace with your state retrieval method
        state = self._get_observation(game_state)
        
        # Return initial state and info dict (Gymnasium requirement)
        return state, {}
    
    def step(self, action):
        # Take an action (0 or 1) and update the game
        self.step_count += 1

        # Pass action to your game (replace with your game's action method)
        self.game.take_action_beta(action)

        game_state = self.game.get_game_state()  # Update state after action
        
        # Get next state
        next_state = self._get_observation(game_state)
        
        '''
        # Calculate reward (customize based on your game)
        if self.step_count < self.max_steps:
            reward = 0.0  # No reward until the end
        else:
            reward = self._compute_reward()  # Define end-game reward
        '''
        reward = self._compute_reward()

        # Check if game is done
        done = self.step_count >= self.max_steps or self.game.is_game_over
        
        # Info dict (optional, can include debug info)
        info = {}
        
        return next_state, reward, done, False, info  # False is 'truncated' flag
    
    def _get_observation(self, game_state):
        # Extract vectors from your game state
        player_hand_vector = game_state["player_encodings_hand"][self.current_player_turn]  # Cards in current players hand
        top_card_vector = game_state["top_card_discard_pile"]  # Top card in discard pile
        dead_cards_vector = game_state["linear_encoding_discard_pile_not_top_X"][self.current_player_turn]  # Dead cards in discard pile
        opp_known_cards_vector = game_state["linear_encoding_players_known_cards"][1] # Opponent known cards
        cards_in_down_pile_vector = game_state["linear_encoding_all_down_cards"] 
        action_mask_vector = game_state["action_mask"]


        # Reduce the action space from 108 * 5, to 56 * 5
        player_hand_vector = [player_hand_vector[i] + player_hand_vector[i+54] for i in range(int(len(player_hand_vector)/2))]
        top_card_vector = [top_card_vector[i] + top_card_vector[i+54] for i in range(int(len(top_card_vector)/2))]
        dead_cards_vector = [dead_cards_vector[i] + dead_cards_vector[i+54] for i in range(int(len(dead_cards_vector)/2))]
        opp_known_cards_vector = [opp_known_cards_vector[i] + opp_known_cards_vector[i+54] for i in range(int(len(opp_known_cards_vector)/2))]
        cards_in_down_pile_vector = [cards_in_down_pile_vector[i] + cards_in_down_pile_vector[i+54] for i in range(int(len(cards_in_down_pile_vector)/2))]
        action_mask_vector = action_mask_vector[0:6] + [min(action_mask_vector[i + 6] + action_mask_vector[i + 60], 1) for i in range(54)]

        # Convert to numpy arrays
        player_hand_vector = np.array(player_hand_vector, dtype=np.float32)
        top_card_vector = np.array(top_card_vector, dtype=np.float32)
        dead_cards_vector = np.array(dead_cards_vector, dtype=np.float32)
        opp_known_cards_vector = np.array(opp_known_cards_vector, dtype=np.float32)
        cards_in_down_pile_vector = np.array(cards_in_down_pile_vector, dtype=np.float32)
        action_mask_vector = np.array(action_mask_vector, dtype=np.float32)

        # Concatenate into a single 270-dimensional vector
        state = np.concatenate([
            player_hand_vector,
            top_card_vector,
            dead_cards_vector,
            opp_known_cards_vector,
            cards_in_down_pile_vector
        ])
        return (state.astype(np.float32), action_mask_vector)
    
    def _compute_reward(self):
        # Only return the rewards at the end of the game 0 until then
        #print(self.game.get_rewards(), "rewards")
        return self.game.get_rewards()[0]

    
    def render(self, mode="human"):
        # Optional: Visualize the game state (e.g., print to console)
        print(f"Step {self.step_count}: Hand: {self.game.get_game_state()['player_encodings_hand'][self.current_player_turn]}")


env = CardGameEnv()
state, info = env.reset()
print("Initial state shape:", state[0].shape)  # Should be (540,)

'''
done = False
while not done:
    action = env.action_space.sample()  # Random action (0 or 1)
    next_state, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")'
'''