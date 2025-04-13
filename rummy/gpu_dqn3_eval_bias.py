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

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
original_stdout = sys.stdout
log_file = open("training_log_dqn_eval.txt", "w")
sys.stdout = log_file

class QNetwork(nn.Module):
    def __init__(self, input_dim=42, output_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.init_bias()

    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def init_bias(self):
        with torch.no_grad():
            # Add small positive bias to action 1
            self.fc3.bias.data[1] += 0.1
            self.fc3.bias.data[0] -= 0.1

# Initialize environment, models, and optimizer
env = CardGameEnv()
q_net = QNetwork().to(device)
target_net = QNetwork().to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=0.001)
replay_buffer = deque(maxlen=20000)
epsilon = 1.0
gamma = 0.95
batch_size = 256
num_episodes = 10000
max_steps = 200
rewards_per_episode = []
wins_per_episode = []

# Evaluation settings
eval_episodes = 50
eval_freq = 500
eval_rewards = []
eval_winrates = []
eval_x = []

# Dual-axis plotting setup
plt.ion()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

x_data, reward_data, winrate_data = [], [], []

line1, = ax1.plot([], [], 'b-', label='Avg Reward')
line2, = ax2.plot([], [], 'g--', label='Win Rate')

ax1.set_xlabel("Episode")
ax1.set_ylabel("Average Reward", color='b')
ax2.set_ylabel("Win Rate", color='g')
ax1.grid()

def update_plot(episode):
    avg_reward = np.mean(rewards_per_episode[-100:]) if rewards_per_episode else 0
    win_rate = np.mean(wins_per_episode[-100:]) if wins_per_episode else 0

    x_data.append(episode)
    reward_data.append(avg_reward)
    winrate_data.append(win_rate)

    line1.set_xdata(x_data)
    line1.set_ydata(reward_data)
    line2.set_xdata(x_data)
    line2.set_ydata(winrate_data)

    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()

    # Plot evaluation dots
    ax1.plot(eval_x, eval_rewards, 'ro', label='Eval Avg Reward' if episode == 0 else "")
    ax2.plot(eval_x, eval_winrates, 'ko', label='Eval Win Rate' if episode == 0 else "")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    plt.draw()
    plt.savefig("reward_plot_dqn_eval_bias.png")
    plt.pause(0.01)

def evaluate_policy(episode):
    total_eval_reward = 0
    total_eval_wins = 0
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        state, action_mask = obs
        state = torch.FloatTensor(state).to(device)
        done = False
        while not done:
            with torch.no_grad():
                q_values = q_net(state)
                masked_q_values = q_values.clone()
                masked_q_values[action_mask == 0] = float('-inf')
                action = torch.argmax(masked_q_values).item()
            obs, reward, done, truncated, info = env.step(action)
            next_state, action_mask = obs
            state = torch.FloatTensor(next_state).to(device)
            total_eval_reward += reward
        if info.get("Winner") == 0:
            total_eval_wins += 1

    avg_eval_reward = total_eval_reward / eval_episodes
    win_rate = total_eval_wins / eval_episodes
    eval_x.append(episode)
    eval_rewards.append(avg_eval_reward)
    eval_winrates.append(win_rate)
    print(f"[Eval @ Ep {episode}] Avg Reward: {avg_eval_reward:.2f}, Win Rate: {win_rate:.2%}")

total_steps = 0
for episode in range(num_episodes):
    observation, _ = env.reset()
    state, action_mask = observation
    state = torch.FloatTensor(state).to(device)
    total_reward = 0

    for step in range(max_steps):
        total_steps += 1
        if np.random.rand() < epsilon:
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(state)
                masked_q_values = q_values.clone()
                masked_q_values[action_mask == 0] = float('-inf')
                action = torch.argmax(masked_q_values).item()

        next_observation, reward, done, truncated, info = env.step(action)
        next_state, next_action_mask = next_observation
        next_state = torch.FloatTensor(next_state).to(device)
        total_reward += reward

        replay_buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
        state = next_state
        action_mask = next_action_mask

        if len(replay_buffer) >= batch_size:
            reward_transitions = [t for t in replay_buffer if t[2] != 0]
            if reward_transitions:
                batch = random.sample(reward_transitions, min(len(reward_transitions), batch_size // 2)) + \
                        random.sample(replay_buffer, batch_size - min(len(reward_transitions), batch_size // 2))
            else:
                batch = random.sample(replay_buffer, batch_size)

            states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(*batch)
            states = torch.stack(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.stack(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)
            next_action_masks = torch.stack([torch.FloatTensor(mask) for mask in next_action_masks]).to(device)

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
    agent_won = info.get("Winner")
    wins_per_episode.append(1 if agent_won == 0 else 0)

    epsilon = max(0.01, 1.0 - episode / 10000)

    if episode % 50 == 0 or total_steps % 1000 == 0:
        target_net.load_state_dict(q_net.state_dict())
        update_plot(episode)

    if episode % eval_freq == 0 and episode != 0:
        evaluate_policy(episode)

    sys.stdout = original_stdout
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {step+1}, Epsilon: {epsilon:.3f}")
    sys.stdout = log_file
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {step+1}, Epsilon: {epsilon:.3f}")

# Save model and close logs
torch.save(q_net.state_dict(), "trained_q_network.pth")
print("Model state dictionary saved to 'trained_q_network.pth'")

log_file.close()
sys.stdout = original_stdout
plt.ioff()
plt.show()
