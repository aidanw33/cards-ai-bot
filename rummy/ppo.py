import numpy as np
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
log_file = open("training_log.txt", "w")
sys.stdout = log_file

class ActorCritic(nn.Module):
    def __init__(self, input_dim=42, output_dim=20):
        super().__init__()
        # Shared backbone
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Actor head (policy)
        self.actor = nn.Linear(64, output_dim)
        # Critic head (value)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def sample_action(logits, action_mask):
    logits = logits.cpu().detach().numpy()
    logits[action_mask == 0] = -1e10  # Mask invalid actions
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    action = torch.multinomial(probs, 1).item()
    return action

# Initialize environment, model, and optimizer
env = CardGameEnv()
model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)  # Typical PPO learning rate
num_episodes = 50000
max_steps = 200
gamma = 0.99  # Discount factor
lam = 0.95    # GAE lambda
clip_epsilon = 0.2  # PPO clipping parameter
value_loss_coef = 0.5
entropy_coef = 0.01
ppo_epochs = 10  # Number of optimization epochs per batch
batch_size = 64  # Mini-batch size for PPO updates
rewards_per_episode = []
wins_per_episode = []

# Plotting setup
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

x_data, reward_data, winrate_data = [], [], []

line1, = ax1.plot([], [], 'b-', label='Avg Reward')
line2, = ax2.plot([], [], 'g--', label='Win Rate (last 100)')

ax1.set_xlabel("Episode")
ax1.set_ylabel("Average Reward", color='b')
ax2.set_ylabel("Win Rate", color='g')
ax1.grid()

total_steps = 0
for episode in range(num_episodes):
    observation, _ = env.reset()
    state, action_mask = observation
    state = torch.FloatTensor(state).to(device)
    total_reward = 0
    trajectory = []
    
    for step in range(max_steps):
        total_steps += 1
        # Get action and value
        with torch.no_grad():
            logits, value = model(state)
        action = sample_action(logits, action_mask)
        log_prob = torch.log_softmax(logits, dim=-1)[action]
        
        # Step environment
        next_observation, reward, done, truncated, info = env.step(action)
        next_state, next_action_mask = next_observation
        next_state = torch.FloatTensor(next_state).to(device)
        total_reward += reward
        
        # Store transition
        trajectory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'done': done,
            'action_mask': action_mask,
            'next_state': next_state
        })
        
        state = next_state
        action_mask = next_action_mask
        
        if done or truncated:
            break
    
    # Track the wins
    if info['Winner'] == 0 :
        wins_per_episode.append(1)
    else :
        wins_per_episode.append(0)
    print(wins_per_episode)

    # Compute returns and advantages using GAE
    returns = []
    advantages = []
    gae = 0
    with torch.no_grad():
        _, next_value = model(state)
        next_value = next_value.item()
    
    for t in reversed(trajectory):
        reward = t['reward']
        done = t['done']
        value = t['value'].item()
        if done:
            delta = reward - value
            gae = delta
        else:
            delta = reward + gamma * next_value - value
            gae = delta + gamma * lam * gae
        returns.insert(0, gae + value)
        advantages.insert(0, gae)
        next_value = value
    
    # Convert trajectory to tensors
    states = torch.stack([t['state'] for t in trajectory]).to(device)
    actions = torch.tensor([t['action'] for t in trajectory], dtype=torch.long).to(device)
    old_log_probs = torch.stack([t['log_prob'] for t in trajectory]).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update
    for _ in range(ppo_epochs):
        # Mini-batch updates
        indices = torch.randperm(len(trajectory))
        for i in range(0, len(trajectory), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]
            
            # Forward pass
            logits, values = model(batch_states)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            values = values.squeeze(1)
            
            # Compute ratios and clipped objective
            ratios = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, batch_returns)
            
            # Entropy bonus
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    rewards_per_episode.append(total_reward)
    
    # Update plot every 50 episodes
    if episode % 50 == 0 and episode != 0:
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

        plt.draw()
        plt.savefig("reward_plot.png")
        plt.pause(0.01)
            
    sys.stdout = original_stdout
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {step+1}")
    sys.stdout = log_file
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {step+1}")

# Save the model
torch.save(model.state_dict(), "trained_ppo_model.pth")
print("Model state dictionary saved to 'trained_ppo_model.pth'")

log_file.close()
sys.stdout = original_stdout
plt.ioff()
plt.show()