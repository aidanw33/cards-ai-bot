import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from env import CardGameEnv

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(229, 256)  # Input: 229-dimensional state
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)    # Output: 2 actions (0 or 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize environment and DQN components
env = CardGameEnv()

q_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=0.001)
replay_buffer = deque(maxlen=10000)
epsilon = 1.0
gamma = 0.99
batch_size = 64
num_episodes = 10000 

rewards_per_episode = []

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(30):  # Max 30 steps
        # Choose action with epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()
        
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + gamma * next_q_values * (1 - dones)
            
            loss = nn.MSELoss()(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done or truncated:
            break
    
    rewards_per_episode.append(total_reward)  # Store episode reward
    epsilon = max(0.01, epsilon * 0.995)  # Decay epsilon
    
    if episode % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())  # Update target net
    
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Compute moving average of rewards
window_size = 10
average_rewards = [np.mean(rewards_per_episode[i:i + window_size]) 
                   for i in range(0, len(rewards_per_episode), window_size)]

# Plot the average reward per 10 episodes
plt.plot(range(0, num_episodes, window_size), average_rewards, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Average Reward (per 10 episodes)")
plt.title("Training Performance")
plt.grid()
plt.show()
