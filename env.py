import gymnasium as gym
import numpy as np
from game import Game

class CardGameEnv(gym.Env):
    def __init__(self):
        super(CardGameEnv, self).__init__()
        
        # Define action space: 0 or 1 (binary)
        self.action_space = gym.spaces.Discrete(2)  # 2 actions: 0 or 1
        
        # Define observation space: 229-dimensional vector
        # Concatenation of 108 (hand) + 108 (top card) + 13 (down ranks)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(229,), dtype=np.float32
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
        self.game.take_action(action, self.current_player_turn)
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
        player_hand_vector = game_state["player_encodings_hand"][self.current_player_turn]  # 108
        top_card_vector = game_state["top_card_discard_pile"]  # 108
        down_pile_ranks_vector = game_state["encoding_player_down_ranks"]  # 13
        
        # Convert to numpy arrays
        player_hand_vector = np.array(player_hand_vector, dtype=np.float32)
        top_card_vector = np.array(top_card_vector, dtype=np.float32)
        down_pile_ranks_vector = np.array(down_pile_ranks_vector, dtype=np.float32)

        # Concatenate into a single 229-dimensional vector
        state = np.concatenate([
            player_hand_vector,
            top_card_vector,
            down_pile_ranks_vector
        ])
        
        return state.astype(np.float32)
    
    def _compute_reward(self):
        # Only return the rewards at the end of the game 0 until then
        print(self.game.get_rewards(), "rewards")
        return self.game.get_rewards()[0]

    
    def render(self, mode="human"):
        # Optional: Visualize the game state (e.g., print to console)
        print(f"Step {self.step_count}: Hand: {self.game.get_game_state()['player_encodings_hand'][self.current_player_turn]}")


env = CardGameEnv()
state, info = env.reset()
print("Initial state shape:", state.shape)  # Should be (229,)

done = False
while not done:
    action = env.action_space.sample()  # Random action (0 or 1)
    next_state, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")