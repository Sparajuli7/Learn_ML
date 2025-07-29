# Reinforcement Learning Advanced: Deep RL, Multi-Agent Systems, and Inverse RL

*"Pushing the boundaries of autonomous learning with cutting-edge RL techniques"*

---

## 📚 Table of Contents

1. [Deep Reinforcement Learning](#deep-reinforcement-learning)
2. [Multi-Agent Reinforcement Learning](#multi-agent-reinforcement-learning)
3. [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
4. [Advanced Exploration Strategies](#advanced-exploration-strategies)
5. [Meta-Learning and Few-Shot RL](#meta-learning-and-few-shot-rl)
6. [Real-World Applications](#real-world-applications)
7. [Exercises and Projects](#exercises-and-projects)
8. [Further Reading](#further-reading)

---

## 🧠 Deep Reinforcement Learning

Deep RL combines deep neural networks with reinforcement learning to handle high-dimensional state spaces.

### Deep Q-Network (DQN)

DQN uses a neural network to approximate Q-values, enabling learning in complex environments.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """Deep Q-Network for complex state spaces"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Deep Q-Learning agent with experience replay and target network"""
    
    def __init__(self, state_size: int, action_size: int, 
                 lr: float = 0.001, gamma: float = 0.99, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 32):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.update_target_network()
        
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.detach().numpy())
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

class ContinuousEnvironment:
    """Continuous state space environment for DQN testing"""
    
    def __init__(self, state_dim=4, action_size=4):
        self.state_dim = state_dim
        self.action_size = action_size
        self.state = np.random.randn(state_dim)
        self.target = np.zeros(state_dim)
        
    def reset(self):
        self.state = np.random.randn(self.state_dim)
        return self.state.copy()
    
    def step(self, action):
        # Simple dynamics
        action_vector = np.zeros(self.state_dim)
        action_vector[action % self.state_dim] = 0.1 * (1 if action < self.state_dim else -1)
        
        self.state += action_vector
        
        # Reward based on distance to target
        distance = np.linalg.norm(self.state - self.target)
        reward = -distance
        
        done = distance < 0.1
        return self.state.copy(), reward, done

# Train DQN agent
env = ContinuousEnvironment()
agent = DQNAgent(state_size=4, action_size=4)

episode_rewards = []
for episode in range(500):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    
    if episode % 50 == 0:
        agent.update_target_network()
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
```

### Actor-Critic Methods

Actor-Critic methods combine policy gradients with value function approximation.

```python
class ActorCritic(nn.Module):
    """Actor-Critic network with shared features"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        # Shared feature layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor (policy) head
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return policy, value

class A2CAgent:
    """Advantage Actor-Critic agent"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.network = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.network(state_tensor)
        
        # Sample action from policy
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action), value
        
    def update(self, states, actions, rewards, log_probs, values):
        """Update actor-critic networks"""
        # Calculate advantages
        advantages = []
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        advantages = returns - torch.cat(values).detach()
        
        # Actor loss (policy gradient)
        actor_loss = -(torch.cat(log_probs) * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(torch.cat(values), returns)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
```

---

## 🤝 Multi-Agent Reinforcement Learning

Multi-agent RL deals with environments where multiple agents interact and learn simultaneously.

### Cooperative Multi-Agent Systems

```python
class MultiAgentEnvironment:
    """Environment with multiple cooperative agents"""
    
    def __init__(self, num_agents: int = 3, grid_size: int = 5):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agent_positions = [(0, 0)] * num_agents
        self.targets = [(grid_size-1, grid_size-1)] * num_agents
        self.obstacles = [(2, 2), (3, 3)]
        
    def reset(self):
        self.agent_positions = [(0, 0)] * self.num_agents
        return self.get_global_state()
    
    def get_global_state(self):
        """Get global state including all agent positions"""
        state = []
        for pos in self.agent_positions:
            state.extend(pos)
        return np.array(state)
    
    def get_agent_observation(self, agent_id: int):
        """Get local observation for specific agent"""
        agent_pos = self.agent_positions[agent_id]
        target_pos = self.targets[agent_id]
        
        # Local observation: agent position, target position, other agents
        obs = list(agent_pos) + list(target_pos)
        for i, pos in enumerate(self.agent_positions):
            if i != agent_id:
                obs.extend(pos)
        return np.array(obs)
    
    def step(self, actions):
        """Execute actions for all agents"""
        new_positions = []
        rewards = []
        
        for agent_id, action in enumerate(actions):
            new_pos = self.move_agent(self.agent_positions[agent_id], action)
            
            # Check collision with obstacles
            if new_pos in self.obstacles:
                new_pos = self.agent_positions[agent_id]  # Stay in place
                rewards.append(-10)
            else:
                self.agent_positions[agent_id] = new_pos
                rewards.append(-1)
            
            new_positions.append(new_pos)
        
        # Cooperative reward: bonus if all agents reach targets
        all_at_target = all(pos == target for pos, target in zip(new_positions, self.targets))
        if all_at_target:
            for i in range(len(rewards)):
                rewards[i] += 50
        
        done = all_at_target
        return self.get_global_state(), rewards, done
    
    def move_agent(self, pos, action):
        """Move agent based on action"""
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        dx, dy = moves[action]
        new_x = max(0, min(self.grid_size-1, pos[0] + dx))
        new_y = max(0, min(self.grid_size-1, pos[1] + dy))
        return (new_x, new_y)

class MultiAgentDQN:
    """Multi-agent DQN with centralized training, decentralized execution"""
    
    def __init__(self, num_agents: int, state_size: int, action_size: int):
        self.num_agents = num_agents
        self.agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
        
    def act(self, global_state, agent_observations):
        """Get actions for all agents"""
        actions = []
        for i, agent in enumerate(self.agents):
            # Use local observation for action selection
            action = agent.act(agent_observations[i])
            actions.append(action)
        return actions
    
    def train(self, experiences):
        """Train all agents on shared experiences"""
        for i, agent in enumerate(self.agents):
            # Train each agent on its own experiences
            for exp in experiences[i]:
                agent.remember(*exp)
            agent.replay()

# Training multi-agent system
env = MultiAgentEnvironment(num_agents=3)
agents = MultiAgentDQN(num_agents=3, state_size=6, action_size=4)

episode_rewards = []
for episode in range(1000):
    global_state = env.reset()
    total_rewards = [0] * env.num_agents
    
    for step in range(100):
        # Get observations for all agents
        observations = [env.get_agent_observation(i) for i in range(env.num_agents)]
        
        # Get actions from all agents
        actions = agents.act(global_state, observations)
        
        # Execute actions
        next_global_state, rewards, done = env.step(actions)
        
        # Store experiences for each agent
        experiences = [[] for _ in range(env.num_agents)]
        for i in range(env.num_agents):
            experiences[i].append((observations[i], actions[i], rewards[i], 
                                env.get_agent_observation(i), done))
            total_rewards[i] += rewards[i]
        
        global_state = next_global_state
        
        if done:
            break
    
    # Train agents
    agents.train(experiences)
    episode_rewards.append(np.mean(total_rewards))
    
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
```

### Competitive Multi-Agent Systems

```python
class CompetitiveEnvironment:
    """Environment with competitive agents"""
    
    def __init__(self, num_agents: int = 2, grid_size: int = 5):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agent_positions = [(0, 0), (grid_size-1, grid_size-1)]
        self.resources = [(grid_size//2, grid_size//2)]
        self.scores = [0, 0]
        
    def reset(self):
        self.agent_positions = [(0, 0), (self.grid_size-1, self.grid_size-1)]
        self.resources = [(self.grid_size//2, self.grid_size//2)]
        self.scores = [0, 0]
        return self.get_state()
    
    def get_state(self):
        """Get global state"""
        state = []
        for pos in self.agent_positions:
            state.extend(pos)
        for resource in self.resources:
            state.extend(resource)
        return np.array(state)
    
    def step(self, actions):
        """Execute competitive actions"""
        rewards = [0, 0]
        
        # Move agents
        for i, action in enumerate(actions):
            new_pos = self.move_agent(self.agent_positions[i], action)
            self.agent_positions[i] = new_pos
            
            # Check if agent collected resource
            if new_pos in self.resources:
                rewards[i] += 10
                self.resources.remove(new_pos)
                self.scores[i] += 1
        
        # Competitive penalty: agents lose points if opponent scores
        for i in range(2):
            opponent = 1 - i
            if rewards[opponent] > 0:
                rewards[i] -= 5
        
        done = len(self.resources) == 0
        return self.get_state(), rewards, done
    
    def move_agent(self, pos, action):
        """Move agent based on action"""
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = moves[action]
        new_x = max(0, min(self.grid_size-1, pos[0] + dx))
        new_y = max(0, min(self.grid_size-1, pos[1] + dy))
        return (new_x, new_y)
```

---

## 🔄 Inverse Reinforcement Learning

Inverse RL learns reward functions from expert demonstrations.

### Maximum Entropy IRL

```python
class ExpertDemonstrations:
    """Collection of expert demonstrations"""
    
    def __init__(self):
        self.demonstrations = []
        
    def add_demonstration(self, trajectory):
        """Add expert trajectory"""
        self.demonstrations.append(trajectory)
    
    def get_feature_expectations(self):
        """Calculate feature expectations from demonstrations"""
        feature_expectations = np.zeros(10)  # Example feature dimension
        
        for trajectory in self.demonstrations:
            for state, action, reward in trajectory:
                features = self.extract_features(state, action)
                feature_expectations += features
        
        return feature_expectations / len(self.demonstrations)
    
    def extract_features(self, state, action):
        """Extract features from state-action pair"""
        # Example feature extraction
        features = np.zeros(10)
        features[0] = state[0]  # x position
        features[1] = state[1]  # y position
        features[2] = action     # action taken
        features[3] = np.linalg.norm(state)  # distance from origin
        # Add more features as needed
        return features

class MaximumEntropyIRL:
    """Maximum Entropy Inverse Reinforcement Learning"""
    
    def __init__(self, env, feature_dim: int = 10):
        self.env = env
        self.feature_dim = feature_dim
        self.reward_weights = np.random.randn(feature_dim)
        
    def reward_function(self, state, action):
        """Reward function parameterized by weights"""
        features = self.extract_features(state, action)
        return np.dot(self.reward_weights, features)
    
    def extract_features(self, state, action):
        """Extract features from state-action pair"""
        features = np.zeros(self.feature_dim)
        features[0] = state[0]
        features[1] = state[1]
        features[2] = action
        features[3] = np.linalg.norm(state)
        return features
    
    def compute_expected_features(self, policy):
        """Compute expected features under current policy"""
        expected_features = np.zeros(self.feature_dim)
        
        # Monte Carlo estimation
        for _ in range(100):
            state = self.env.reset()
            for _ in range(50):
                action = policy.select_action(state)
                features = self.extract_features(state, action)
                expected_features += features
                
                state, _, done = self.env.step(action)
                if done:
                    break
        
        return expected_features / 100
    
    def update_reward_weights(self, expert_features, policy_features, lr=0.01):
        """Update reward weights using gradient ascent"""
        gradient = expert_features - policy_features
        self.reward_weights += lr * gradient
    
    def learn_reward_function(self, expert_demos, num_iterations=100):
        """Learn reward function from expert demonstrations"""
        expert_features = expert_demos.get_feature_expectations()
        
        for iteration in range(num_iterations):
            # Compute expected features under current reward function
            policy = self.create_policy_from_reward()
            policy_features = self.compute_expected_features(policy)
            
            # Update reward weights
            self.update_reward_weights(expert_features, policy_features)
            
            if iteration % 10 == 0:
                feature_diff = np.linalg.norm(expert_features - policy_features)
                print(f"Iteration {iteration}, Feature Difference: {feature_diff:.4f}")
    
    def create_policy_from_reward(self):
        """Create policy from current reward function"""
        # Simple policy that maximizes immediate reward
        class RewardBasedPolicy:
            def __init__(self, reward_func, env):
                self.reward_func = reward_func
                self.env = env
            
            def select_action(self, state):
                best_action = 0
                best_reward = float('-inf')
                
                for action in range(4):  # Assuming 4 actions
                    reward = self.reward_func(state, action)
                    if reward > best_reward:
                        best_reward = reward
                        best_action = action
                
                return best_action
        
        return RewardBasedPolicy(self.reward_function, self.env)
```

---

## 🔍 Advanced Exploration Strategies

### Curiosity-Driven Exploration

```python
class CuriosityModule(nn.Module):
    """Neural network for predicting next state"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        encoded = self.encoder(x)
        predicted_next_state = self.predictor(encoded)
        return predicted_next_state

class CuriosityAgent:
    """Agent with curiosity-driven exploration"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = DQN(state_size, action_size)
        self.curiosity_module = CuriosityModule(state_size, action_size)
        self.optimizer = optim.Adam(list(self.q_network.parameters()) + 
                                   list(self.curiosity_module.parameters()))
        
    def compute_curiosity_reward(self, state, action, next_state):
        """Compute curiosity reward based on prediction error"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor([action]).float().unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        predicted_next_state = self.curiosity_module(state_tensor, action_tensor)
        prediction_error = F.mse_loss(predicted_next_state, next_state_tensor)
        
        # Curiosity reward is the prediction error
        return prediction_error.item()
    
    def update(self, state, action, reward, next_state):
        """Update both Q-network and curiosity module"""
        # Standard Q-learning update
        q_loss = self.compute_q_loss(state, action, reward, next_state)
        
        # Curiosity update
        curiosity_loss = self.compute_curiosity_loss(state, action, next_state)
        
        total_loss = q_loss + 0.1 * curiosity_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
```

### Hierarchical Reinforcement Learning

```python
class HierarchicalAgent:
    """Hierarchical RL agent with high-level and low-level policies"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # High-level policy (meta-controller)
        self.meta_controller = DQN(state_size, 4)  # 4 high-level actions
        
        # Low-level policy (primitive actions)
        self.primitive_controller = DQN(state_size, action_size)
        
        # Options (temporally extended actions)
        self.options = {
            0: self.option_move_to_goal,
            1: self.option_explore,
            2: self.option_avoid_obstacles,
            3: self.option_idle
        }
        
    def select_meta_action(self, state):
        """Select high-level action (option)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.meta_controller(state_tensor)
        return np.argmax(q_values.detach().numpy())
    
    def execute_option(self, state, option_id, max_steps=10):
        """Execute a temporally extended action (option)"""
        option = self.options[option_id]
        return option(state, max_steps)
    
    def option_move_to_goal(self, state, max_steps):
        """Option: move towards goal"""
        steps = 0
        total_reward = 0
        
        while steps < max_steps:
            # Use primitive controller to move towards goal
            action = self.select_primitive_action(state)
            next_state, reward, done = self.env.step(action)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        return state, total_reward, done
    
    def option_explore(self, state, max_steps):
        """Option: explore environment"""
        steps = 0
        total_reward = 0
        
        while steps < max_steps:
            # Random exploration
            action = np.random.randint(self.action_size)
            next_state, reward, done = self.env.step(action)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        return state, total_reward, done
    
    def select_primitive_action(self, state):
        """Select low-level action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.primitive_controller(state_tensor)
        return np.argmax(q_values.detach().numpy())
```

---

## 🎯 Meta-Learning and Few-Shot RL

### Model-Agnostic Meta-Learning (MAML)

```python
class MAMLRLAgent:
    """Model-Agnostic Meta-Learning for RL"""
    
    def __init__(self, state_size: int, action_size: int, lr=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Meta-parameters (initial policy)
        self.meta_policy = DQN(state_size, action_size)
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=lr)
        
    def adapt_to_task(self, task_env, adaptation_steps=5):
        """Adapt to new task using few-shot learning"""
        # Copy meta-policy for task-specific adaptation
        task_policy = DQN(self.state_size, self.action_size)
        task_policy.load_state_dict(self.meta_policy.state_dict())
        task_optimizer = optim.Adam(task_policy.parameters(), lr=self.lr)
        
        # Few-shot adaptation
        for step in range(adaptation_steps):
            # Collect experience on new task
            state = task_env.reset()
            total_reward = 0
            
            for _ in range(50):  # Short episode
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = task_policy(state_tensor)
                action = np.argmax(q_values.detach().numpy())
                
                next_state, reward, done = task_env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update task-specific policy
            task_optimizer.zero_grad()
            # Simplified loss for demonstration
            loss = torch.tensor(-total_reward, requires_grad=True)
            loss.backward()
            task_optimizer.step()
        
        return task_policy
    
    def meta_update(self, task_batch):
        """Meta-update using multiple tasks"""
        meta_loss = 0
        
        for task_env in task_batch:
            # Adapt to task
            adapted_policy = self.adapt_to_task(task_env)
            
            # Evaluate adapted policy
            performance = self.evaluate_policy(adapted_policy, task_env)
            meta_loss += -performance  # Negative because we maximize performance
        
        # Update meta-parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
    
    def evaluate_policy(self, policy, env, episodes=5):
        """Evaluate policy performance"""
        total_rewards = []
        
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for _ in range(100):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy(state_tensor)
                action = np.argmax(q_values.detach().numpy())
                
                state, reward, done = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
```

---

## 🌍 Real-World Applications

### 1. **Autonomous Vehicle Navigation**

```python
class AutonomousVehicleRL:
    """RL for autonomous vehicle control"""
    
    def __init__(self, vehicle_env):
        self.env = vehicle_env
        self.safety_constraints = {
            'max_speed': 30.0,
            'min_distance': 2.0,
            'emergency_brake_threshold': 1.0
        }
        
        # Multi-objective reward function
        self.reward_weights = {
            'progress': 1.0,
            'safety': 10.0,
            'comfort': 0.5,
            'efficiency': 0.3
        }
    
    def compute_safety_reward(self, state):
        """Compute safety-based reward component"""
        distance_to_obstacle = state['distance_to_obstacle']
        speed = state['speed']
        
        if distance_to_obstacle < self.safety_constraints['min_distance']:
            return -10.0  # Safety violation
        elif distance_to_obstacle < self.safety_constraints['emergency_brake_threshold']:
            return -5.0   # Warning
        
        return 0.0
    
    def compute_progress_reward(self, state, prev_state):
        """Compute progress-based reward component"""
        current_position = state['position']
        prev_position = prev_state['position']
        
        progress = np.linalg.norm(current_position - prev_position)
        return progress
    
    def compute_comfort_reward(self, action, prev_action):
        """Compute comfort-based reward component"""
        action_change = np.linalg.norm(action - prev_action)
        return -action_change  # Penalize sudden changes
    
    def get_multi_objective_reward(self, state, action, prev_state, prev_action):
        """Compute multi-objective reward"""
        safety_reward = self.compute_safety_reward(state)
        progress_reward = self.compute_progress_reward(state, prev_state)
        comfort_reward = self.compute_comfort_reward(action, prev_action)
        
        total_reward = (
            self.reward_weights['safety'] * safety_reward +
            self.reward_weights['progress'] * progress_reward +
            self.reward_weights['comfort'] * comfort_reward
        )
        
        return total_reward
```

### 2. **Robotic Manipulation**

```python
class RoboticManipulationRL:
    """RL for robotic manipulation tasks"""
    
    def __init__(self, robot_env):
        self.env = robot_env
        self.task_hierarchy = {
            'grasp': self.subtask_grasp,
            'move': self.subtask_move,
            'place': self.subtask_place,
            'release': self.subtask_release
        }
    
    def subtask_grasp(self, state):
        """Subtask: grasp object"""
        # Implement grasp-specific policy
        pass
    
    def subtask_move(self, state):
        """Subtask: move to target"""
        # Implement movement policy
        pass
    
    def subtask_place(self, state):
        """Subtask: place object"""
        # Implement placement policy
        pass
    
    def subtask_release(self, state):
        """Subtask: release object"""
        # Implement release policy
        pass
    
    def hierarchical_policy(self, state, task_sequence):
        """Execute hierarchical policy"""
        for subtask in task_sequence:
            if subtask in self.task_hierarchy:
                action = self.task_hierarchy[subtask](state)
                state, reward, done = self.env.step(action)
                
                if done:
                    break
        
        return state, reward, done
```

### 3. **Game AI and Strategy**

```python
class GameAI:
    """Advanced game AI using RL"""
    
    def __init__(self, game_env):
        self.env = game_env
        self.strategy_network = DQN(game_env.state_size, game_env.action_size)
        self.tactics_network = DQN(game_env.state_size, game_env.action_size)
        
    def strategic_planning(self, state, time_horizon=10):
        """Long-term strategic planning"""
        # Use strategy network for high-level planning
        strategy_q_values = self.strategy_network(torch.FloatTensor(state).unsqueeze(0))
        return np.argmax(strategy_q_values.detach().numpy())
    
    def tactical_execution(self, state, strategy):
        """Short-term tactical execution"""
        # Use tactics network for immediate actions
        tactics_q_values = self.tactics_network(torch.FloatTensor(state).unsqueeze(0))
        return np.argmax(tactics_q_values.detach().numpy())
    
    def adaptive_strategy(self, opponent_history):
        """Adapt strategy based on opponent behavior"""
        # Analyze opponent patterns and adjust strategy
        opponent_tendencies = self.analyze_opponent(opponent_history)
        return self.adjust_strategy(opponent_tendencies)
```

---

## 🧪 Exercises and Projects

### Exercise 1: Deep Q-Network Implementation

Implement a DQN agent for the Atari game environment and compare performance with different architectures.

```python
class AtariDQN(nn.Module):
    """Deep Q-Network for Atari games"""
    
    def __init__(self, num_actions):
        super().__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Your task: Implement full DQN training loop for Atari games
# and experiment with different hyperparameters
```

### Exercise 2: Multi-Agent Coordination

Create a multi-agent system where agents must coordinate to solve a complex task.

```python
class CoordinatedMultiAgentSystem:
    """Multi-agent system with coordination mechanisms"""
    
    def __init__(self, num_agents, task_complexity):
        self.num_agents = num_agents
        self.task_complexity = task_complexity
        self.communication_protocol = self.create_communication_protocol()
        
    def create_communication_protocol(self):
        """Define communication protocol between agents"""
        protocol = {
            'message_types': ['request', 'response', 'coordinate', 'emergency'],
            'message_format': {
                'sender_id': int,
                'receiver_id': int,
                'message_type': str,
                'content': dict
            }
        }
        return protocol
    
    def coordinate_actions(self, agent_states, global_state):
        """Coordinate actions across all agents"""
        # Implement coordination logic
        coordinated_actions = []
        
        for i, agent_state in enumerate(agent_states):
            # Consider other agents' states for coordination
            other_agent_states = [s for j, s in enumerate(agent_states) if j != i]
            coordinated_action = self.compute_coordinated_action(
                agent_state, other_agent_states, global_state
            )
            coordinated_actions.append(coordinated_action)
        
        return coordinated_actions
    
    def compute_coordinated_action(self, agent_state, other_states, global_state):
        """Compute action considering other agents"""
        # Implement coordination algorithm
        pass

# Your task: Implement the coordination mechanism and train
# the multi-agent system on a complex task
```

### Exercise 3: Inverse Reinforcement Learning

Implement IRL to learn reward functions from expert demonstrations.

```python
class IRLSystem:
    """Inverse Reinforcement Learning system"""
    
    def __init__(self, env, feature_extractor):
        self.env = env
        self.feature_extractor = feature_extractor
        self.reward_function = self.initialize_reward_function()
        
    def initialize_reward_function(self):
        """Initialize parameterized reward function"""
        return lambda state, action, weights: np.dot(
            self.feature_extractor(state, action), weights
        )
    
    def collect_expert_demonstrations(self, expert_policy, num_demos=100):
        """Collect expert demonstrations"""
        demonstrations = []
        
        for _ in range(num_demos):
            trajectory = []
            state = self.env.reset()
            
            for _ in range(100):
                action = expert_policy(state)
                next_state, reward, done = self.env.step(action)
                
                trajectory.append((state, action, reward))
                state = next_state
                
                if done:
                    break
            
            demonstrations.append(trajectory)
        
        return demonstrations
    
    def compute_feature_expectations(self, demonstrations):
        """Compute feature expectations from demonstrations"""
        feature_expectations = np.zeros(self.feature_extractor.feature_dim)
        
        for trajectory in demonstrations:
            for state, action, reward in trajectory:
                features = self.feature_extractor(state, action)
                feature_expectations += features
        
        return feature_expectations / len(demonstrations)
    
    def learn_reward_function(self, expert_demos):
        """Learn reward function using IRL"""
        expert_features = self.compute_feature_expectations(expert_demos)
        
        # Implement IRL algorithm (e.g., Maximum Entropy IRL)
        learned_weights = self.maximum_entropy_irl(expert_features)
        
        return learned_weights
    
    def maximum_entropy_irl(self, expert_features):
        """Maximum Entropy IRL implementation"""
        # Implement the algorithm
        pass

# Your task: Implement the complete IRL system and test
# it on a simple environment with known reward function
```

### Project: Advanced Trading Bot

Create an advanced trading bot using deep RL with multiple strategies.

```python
class AdvancedTradingBot:
    """Advanced trading bot with multiple RL strategies"""
    
    def __init__(self, market_data, initial_capital=100000):
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.strategies = {
            'momentum': self.momentum_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'arbitrage': self.arbitrage_strategy,
            'risk_management': self.risk_management_strategy
        }
        
        # Multiple RL agents for different strategies
        self.strategy_agents = {
            'momentum': DQNAgent(state_size=10, action_size=3),
            'mean_reversion': DQNAgent(state_size=10, action_size=3),
            'arbitrage': DQNAgent(state_size=10, action_size=3)
        }
    
    def momentum_strategy(self, state):
        """Momentum-based trading strategy"""
        # Implement momentum strategy using RL
        pass
    
    def mean_reversion_strategy(self, state):
        """Mean reversion trading strategy"""
        # Implement mean reversion strategy using RL
        pass
    
    def arbitrage_strategy(self, state):
        """Arbitrage trading strategy"""
        # Implement arbitrage strategy using RL
        pass
    
    def risk_management_strategy(self, portfolio_state):
        """Risk management strategy"""
        # Implement risk management using RL
        pass
    
    def ensemble_decision(self, market_state):
        """Combine multiple strategies for final decision"""
        strategy_decisions = {}
        
        for strategy_name, agent in self.strategy_agents.items():
            decision = agent.act(market_state)
            strategy_decisions[strategy_name] = decision
        
        # Weighted combination of strategies
        final_decision = self.combine_strategies(strategy_decisions)
        
        return final_decision
    
    def combine_strategies(self, strategy_decisions):
        """Combine multiple strategy decisions"""
        # Implement ensemble method
        pass

# Your task: Implement all strategies and train the ensemble
# trading bot on historical market data
```

---

## 📚 Further Reading

### Books
- **"Deep Reinforcement Learning"** by Pieter Abbeel
- **"Multi-Agent Systems: Algorithmic, Game-Theoretic, and Logical Foundations"** by Yoav Shoham
- **"Inverse Reinforcement Learning"** by Stuart Russell

### Papers
- **"Human-level control through deep reinforcement learning"** (DQN)
- **"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"** (MADDPG)
- **"Maximum Entropy Inverse Reinforcement Learning"** (MaxEnt IRL)
- **"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"** (MAML)

### Online Resources
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym](https://www.gymlibrary.dev/)

### Tools and Frameworks
- **RLlib**: Scalable RL for production
- **Stable Baselines3**: Production-ready RL implementations
- **MuJoCo**: Physics-based simulation
- **Unity ML-Agents**: Game-based RL

---

## 🎯 Key Takeaways

1. **Deep RL** enables learning in high-dimensional state spaces
2. **Multi-agent RL** handles complex interactions between multiple agents
3. **Inverse RL** learns reward functions from expert demonstrations
4. **Advanced exploration** strategies improve learning efficiency
5. **Meta-learning** enables rapid adaptation to new tasks
6. **Real-world applications** require careful consideration of safety and constraints

The Core ML Fields section is now complete! The next logical step would be to move to the Advanced Topics section or continue with other specialized domains. 