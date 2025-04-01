import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from env2 import CardGameEnv  # Assuming this is in env.py

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim=540, output_dim=114):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Input: 540-dimensional state
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)  # Output: 114 actions
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize environment and DQN components
env = CardGameEnv()
q_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=0.001)  # Reduced learning rate
replay_buffer = deque(maxlen=10000)
epsilon = 1.0
gamma = 0.99
batch_size = 64
num_episodes = 1000  # Reduced for practical training time
max_steps = 1000  # Matches env.max_steps

rewards_per_episode = []

# Set up live plotting
plt.ion()
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data, marker='o', linestyle='-')
ax.set_xlabel("Episode")
ax.set_ylabel("Average Reward (per 100 episodes)")
ax.set_title("Training Performance")
ax.grid()

# Training loop
for episode in range(num_episodes):
    observation, _ = env.reset()
    state, action_mask = observation  # Unpack the tuple
    total_reward = 0
    
    for step in range(max_steps):
        # Choose action with epsilon-greedy, respecting action mask
        if np.random.rand() < epsilon:
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                # Mask invalid actions by setting their Q-values to -inf
                masked_q_values = q_values.clone()
                masked_q_values[action_mask == 0] = float('-inf')
                action = torch.argmax(masked_q_values).item()
        
        next_observation, reward, done, truncated, info = env.step(action)
        next_state, next_action_mask = next_observation  # Unpack next observation
        total_reward += reward
        
        # Store transition in replay buffer
        replay_buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
        state = next_state
        action_mask = next_action_mask
        
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            next_action_masks = torch.FloatTensor(next_action_masks)
            
            # Compute Q-values and targets
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(next_states)
                # Mask invalid actions in next state
                next_q_values[next_action_masks == 0] = float('-inf')
                max_next_q_values = next_q_values.max(1)[0]
                targets = rewards + gamma * max_next_q_values * (1 - dones)
            
            loss = nn.MSELoss()(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done or truncated:
            break
    
    rewards_per_episode.append(total_reward)
    epsilon = max(0.01, epsilon * 0.995)  # Decay epsilon
    
    if episode % 100 == 0:
        target_net.load_state_dict(q_net.state_dict())  # Update target network
        
        # Compute moving average and update plot
        window_size = 100
        avg_reward = np.mean(rewards_per_episode[-window_size:]) if rewards_per_episode else 0
        x_data.append(episode)
        y_data.append(avg_reward)
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
    
    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

plt.ioff()
plt.show()