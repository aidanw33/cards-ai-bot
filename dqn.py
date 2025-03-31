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
optimizer = optim.Adam(q_net.parameters(), lr=0.01)
replay_buffer = deque(maxlen=10000)
epsilon = 1.0
gamma = 0.99
batch_size = 64
num_episodes = 100000 

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
    
    if episode % 100 == 0:
        target_net.load_state_dict(q_net.state_dict())  # Update target net
        
        # Compute moving average and update plot
        window_size = 100
        avg_reward = np.mean(rewards_per_episode[-window_size:])
        x_data.append(episode)
        y_data.append(avg_reward)
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
    
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

plt.ioff()
plt.show()