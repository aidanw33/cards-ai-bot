import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from env2 import CardGameEnv
import sys

original_stdout = sys.stdout
log_file = open("training_log.txt", "w")
sys.stdout = log_file

class QNetwork(nn.Module):
    def __init__(self, input_dim=540, output_dim=114):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Larger capacity
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

env = CardGameEnv()
q_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=0.0001)  # Lower LR
replay_buffer = deque(maxlen=20000)  # Larger buffer
epsilon = 1.0
gamma = 0.95  # Higher gamma
batch_size = 64
num_episodes = 50000
max_steps = 200
rewards_per_episode = []

plt.ion()
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data, marker='o', linestyle='-')
ax.set_xlabel("Episode")
ax.set_ylabel("Average Reward (per 100 episodes)")
ax.grid()

total_steps = 0
for episode in range(num_episodes):
    observation, _ = env.reset()
    state, action_mask = observation
    total_reward = 0
    
    for step in range(max_steps):
        total_steps += 1
        if np.random.rand() < epsilon:
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                masked_q_values = q_values.clone()
                masked_q_values[action_mask == 0] = float('-inf')
                action = torch.argmax(masked_q_values).item()
        
        next_observation, reward, done, truncated, info = env.step(action)
        next_state, next_action_mask = next_observation
        total_reward += reward
        replay_buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
        state = next_state
        action_mask = next_action_mask
        
        if len(replay_buffer) >= batch_size:
            # Oversample reward transitions
            reward_transitions = [t for t in replay_buffer if t[2] != 0]
            if reward_transitions:
                batch = random.sample(reward_transitions, min(len(reward_transitions), batch_size//2)) + \
                        random.sample(replay_buffer, batch_size - min(len(reward_transitions), batch_size//2))
            else:
                batch = random.sample(replay_buffer, batch_size)
            
            states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(*batch)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            next_action_masks = torch.FloatTensor(next_action_masks)
            
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(next_states)
                next_q_values[next_action_masks == 0] = float('-inf')
                max_next_q_values = next_q_values.max(1)[0]
                targets = rewards + gamma * max_next_q_values * (1 - dones)
            
            loss = nn.MSELoss()(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()
        
        if done or truncated:
            break
    
    rewards_per_episode.append(total_reward)
    epsilon = max(0.01, 1.0 - episode / 25000)  # Linear decay
    #epsilon = max(0.01, 0.995 ** episode)  # Slower, smoother decay

    if episode % 50 == 0 or total_steps % 1000 == 0:  # More frequent updates
        target_net.load_state_dict(q_net.state_dict())
        avg_reward = np.mean(rewards_per_episode[-100:]) if rewards_per_episode else 0
        x_data.append(episode)
        y_data.append(avg_reward)
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
    
    sys.stdout = original_stdout
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {step+1}, Epsilon: {epsilon:.3f}")
    sys.stdout = log_file
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {step+1}, Epsilon: {epsilon:.3f}")


# After training is complete (e.g., after the episode loop)
torch.save(q_net.state_dict(), "trained_q_network.pth")
print("Model state dictionary saved to 'trained_q_network.pth'")

log_file.close()
sys.stdout = original_stdout
plt.ioff()
plt.show()