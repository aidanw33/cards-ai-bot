import gymnasium as gym
import numpy as np
from game import Game

class CardGameEnv(gym.Env):
    def __init__(self):
        super(CardGameEnv, self).__init__()
        
        # Define action space: 0 or 1 (binary)
        self.action_space = gym.spaces.Discrete(20)  # 20 actions
        
        # Define observation space: 540-dimensional vector
        # Concatenation of 5 x 108
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(42,), dtype=np.float32
        )
        
        # Your game logic object (replace with your actual game class/instance)
        self.game = Game()  # Replace with your game implementation
        self.current_player_turn = 0  # Assuming single-player for now
        self.step_count = 0
        self.max_steps = 300  # 1000 decisions per game
        
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
        info = {"Winner": self.game.game_winner}
        
        return next_state, reward, done, False, info  # False is 'truncated' flag
    
    def _get_observation(self, game_state):
        # Extract vectors from your game state
        player_hand_vector = game_state["player_encodings_hand"][self.current_player_turn]  # Cards in current players hand
        top_card_vector = game_state["top_card_discard_pile"]  # Top card in discard pile
        dead_cards_vector = game_state["linear_encoding_discard_pile_not_top_X"][self.current_player_turn]  # Dead cards in discard pile
        opp_known_cards_vector = game_state["linear_encoding_players_known_cards"][1] # Opponent known cards
        cards_in_down_pile_vector = game_state["linear_encoding_all_down_cards"] 
        action_mask_vector = game_state["action_mask"] # Action mask is 0, 1 (Buy or discard), 2 - 5 (Amount of cards to buy), 6 - 114, card to discard
        
        player_hand_vector        = self._shape_observation_vector_ranks(player_hand_vector, True)
        top_card_vector           = self._shape_observation_vector_ranks(top_card_vector, False)
        cards_in_down_pile_vector = self._shape_observation_vector_ranks(cards_in_down_pile_vector, False)

        # Convert the action mask to keep the first 6 elements, but then translate 6-114, to just the rank representation of the cards
        action_mask_first_elements = action_mask_vector[0:6]
        action_mask_last_elements = self._shape_observation_vector_ranks(action_mask_vector[6:], False)

        # Convert to numpy arrays
        player_hand_vector = np.array(player_hand_vector, dtype=np.float32)
        top_card_vector = np.array(top_card_vector, dtype=np.float32)
        cards_in_down_pile_vector = np.array(cards_in_down_pile_vector, dtype=np.float32)
        action_mask_vector = np.append(action_mask_first_elements, action_mask_last_elements)

        # Concatenate into a single 270-dimensional vector
        state = np.concatenate([
            player_hand_vector,
            top_card_vector,
            cards_in_down_pile_vector
        ])
        return (state.astype(np.float32), action_mask_vector)

    # Returns a vector which only signifies rank and ignores suit
    # If count, values can be 0(does not appear), 1/3 (appears once), 2/3 (appears twice), 1 (appears 3+ times)
    # If not count, values can be 0 (does not appear), or 1 (does appear)
    def _shape_observation_vector_ranks(self, vector, count) :

        res = [0] * 14
        for i, val in enumerate(vector) :
            if val :
                i = i % 54
                if i == 0 or i == 1 :
                    if count :
                        res[0] = min(res[0] + (1/3), 1)
                    else :
                        res[0] = 1
                    continue

                if i < 2 or i > 53:
                    raise ValueError("Input must be between 2 and 53 inclusive")
                i = ((i - 2) // 4) + 1
                if count :
                    res[i] = min(res[i] + (1/3), 1)
                else :
                    res[i] = 1
        return res
    
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
action = env.action_space.sample()  # Random action (0 or 1)
next_state, reward, done, truncated, info = env.step(action)
'''
done = False
while not done:
    action = env.action_space.sample()  # Random action (0 or 1)
    next_state, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")'
'''