# Reinforcement Learning Basics: Markov Processes, Q-Learning, and Value Functions

*"Teaching agents to learn optimal behavior through interaction with their environment"*

---

## 📚 Table of Contents

1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Markov Decision Processes](#markov-decision-processes)
3. [Value Functions](#value-functions)
4. [Q-Learning](#q-learning)
5. [Policy Gradient Methods](#policy-gradient-methods)
6. [Real-World Applications](#real-world-applications)
7. [Exercises and Projects](#exercises-and-projects)
8. [Further Reading](#further-reading)

---

## 🎯 Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards.

### Key Concepts

- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State**: Current situation of the environment
- **Action**: What the agent can do
- **Reward**: Feedback from the environment
- **Policy**: Strategy for selecting actions

### RL Framework

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import gym
from collections import defaultdict

class RLEnvironment:
    """Simple grid world environment for RL demonstrations"""
    
    def __init__(self, size: int = 4):
        self.size = size
        self.state = (0, 0)  # Start position
        self.goal = (size-1, size-1)  # Goal position
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
    def reset(self) -> Tuple[int, int]:
        """Reset environment to initial state"""
        self.state = (0, 0)
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Take action and return (next_state, reward, done)"""
        dx, dy = self.actions[action]
        new_x = max(0, min(self.size-1, self.state[0] + dx))
        new_y = max(0, min(self.size-1, self.state[1] + dy))
        self.state = (new_x, new_y)
        
        # Reward: -1 for each step, +10 for reaching goal
        reward = -1 if self.state != self.goal else 10
        done = self.state == self.goal
        
        return self.state, reward, done
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from current state"""
        return list(range(len(self.actions)))

# Example usage
env = RLEnvironment(4)
print(f"Initial state: {env.reset()}")
next_state, reward, done = env.step(0)  # Move right
print(f"After action 0: state={next_state}, reward={reward}, done={done}")
```

---

## 🔄 Markov Decision Processes

A Markov Decision Process (MDP) is the mathematical framework for RL problems.

### MDP Components

- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probability function
- **R**: Reward function
- **γ**: Discount factor

### Mathematical Foundation

The Bellman equation for value functions:

V(s) = max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]

```python
class MDP:
    """Markov Decision Process implementation"""
    
    def __init__(self, states: List, actions: List, gamma: float = 0.9):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.rewards = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
    def set_transition(self, state, action, next_state, probability: float):
        """Set transition probability P(next_state|state, action)"""
        self.transitions[state][action][next_state] = probability
    
    def set_reward(self, state, action, next_state, reward: float):
        """Set reward R(state, action, next_state)"""
        self.rewards[state][action][next_state] = reward
    
    def get_transition_prob(self, state, action, next_state) -> float:
        """Get transition probability"""
        return self.transitions[state][action][next_state]
    
    def get_reward(self, state, action, next_state) -> float:
        """Get reward"""
        return self.rewards[state][action][next_state]

# Example: Simple MDP with 3 states
states = ['s0', 's1', 's2']
actions = ['a0', 'a1']
mdp = MDP(states, actions, gamma=0.9)

# Define transitions and rewards
mdp.set_transition('s0', 'a0', 's1', 0.8)
mdp.set_transition('s0', 'a0', 's2', 0.2)
mdp.set_transition('s0', 'a1', 's1', 0.3)
mdp.set_transition('s0', 'a1', 's2', 0.7)

mdp.set_reward('s0', 'a0', 's1', 1.0)
mdp.set_reward('s0', 'a0', 's2', -1.0)
mdp.set_reward('s0', 'a1', 's1', 2.0)
mdp.set_reward('s0', 'a1', 's2', 0.5)

print("MDP Transition Probabilities:")
for state in states:
    for action in actions:
        for next_state in states:
            prob = mdp.get_transition_prob(state, action, next_state)
            if prob > 0:
                reward = mdp.get_reward(state, action, next_state)
                print(f"P({next_state}|{state},{action}) = {prob:.2f}, R = {reward:.2f}")
```

---

## 💰 Value Functions

Value functions estimate the expected return from a state or state-action pair.

### State Value Function

V(s) = E[Σ γ^t R_t | s_0 = s]

### Action-Value Function

Q(s,a) = E[Σ γ^t R_t | s_0 = s, a_0 = a]

```python
class ValueIteration:
    """Value Iteration algorithm for solving MDPs"""
    
    def __init__(self, mdp: MDP, epsilon: float = 0.01):
        self.mdp = mdp
        self.epsilon = epsilon
        self.values = {state: 0.0 for state in mdp.states}
        
    def value_iteration(self, max_iterations: int = 1000) -> Dict:
        """Solve MDP using value iteration"""
        for iteration in range(max_iterations):
            delta = 0
            new_values = {}
            
            for state in self.mdp.states:
                # Calculate value for each action
                action_values = []
                for action in self.mdp.actions:
                    value = 0
                    for next_state in self.mdp.states:
                        prob = self.mdp.get_transition_prob(state, action, next_state)
                        reward = self.mdp.get_reward(state, action, next_state)
                        value += prob * (reward + self.mdp.gamma * self.values[next_state])
                    action_values.append(value)
                
                # Take maximum over actions
                new_values[state] = max(action_values)
                delta = max(delta, abs(new_values[state] - self.values[state]))
            
            self.values = new_values
            
            if delta < self.epsilon:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self.values
    
    def get_policy(self) -> Dict:
        """Extract optimal policy from value function"""
        policy = {}
        for state in self.mdp.states:
            action_values = []
            for action in self.mdp.actions:
                value = 0
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    value += prob * (reward + self.mdp.gamma * self.values[next_state])
                action_values.append(value)
            policy[state] = self.mdp.actions[np.argmax(action_values)]
        
        return policy

# Solve the MDP
vi = ValueIteration(mdp)
optimal_values = vi.value_iteration()
optimal_policy = vi.get_policy()

print("\nOptimal Values:")
for state, value in optimal_values.items():
    print(f"V({state}) = {value:.3f}")

print("\nOptimal Policy:")
for state, action in optimal_policy.items():
    print(f"π({state}) = {action}")
```

---

## 🎯 Q-Learning

Q-Learning is a model-free RL algorithm that learns action-value functions directly.

### Q-Learning Algorithm

Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

```python
class QLearning:
    """Q-Learning implementation with epsilon-greedy exploration"""
    
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: defaultdict(float))
        
    def get_action(self, state) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(self.env.get_valid_actions())
        else:
            # Exploitation: best action
            q_values = [self.q_table[state][a] for a in self.env.get_valid_actions()]
            return self.env.get_valid_actions()[np.argmax(q_values)]
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action]
        next_q_max = max([self.q_table[next_state][a] for a in self.env.get_valid_actions()])
        new_q = current_q + self.alpha * (reward + self.gamma * next_q_max - current_q)
        self.q_table[state][action] = new_q
    
    def train(self, episodes: int = 1000) -> List[float]:
        """Train the agent"""
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 100:  # Prevent infinite loops
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                
                self.update_q_value(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            self.epsilon *= self.epsilon_decay
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards

# Train Q-Learning agent
env = RLEnvironment(4)
q_agent = QLearning(env, alpha=0.1, gamma=0.9, epsilon=0.1)
rewards = q_agent.train(episodes=500)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Q-Learning Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

# Test the learned policy
def test_policy(agent, env, episodes=10):
    """Test the learned policy"""
    total_rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            action = agent.get_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

avg_reward, std_reward = test_policy(q_agent, env)
print(f"\nTest Performance: {avg_reward:.2f} ± {std_reward:.2f}")
```

---

## 📈 Policy Gradient Methods

Policy gradient methods directly optimize the policy function.

### REINFORCE Algorithm

∇J(θ) = E[∇_θ log π_θ(a|s) Q^π(s,a)]

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Neural network policy for REINFORCE"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class REINFORCE:
    """REINFORCE algorithm implementation"""
    
    def __init__(self, env, state_size: int, action_size: int, 
                 lr: float = 0.001, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state) -> Tuple[int, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def train_episode(self) -> Tuple[List[float], List[float], float]:
        """Train on one episode"""
        state = self.env.reset()
        log_probs = []
        rewards = []
        total_reward = 0
        
        while True:
            action, log_prob = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return log_probs, rewards, total_reward
    
    def update_policy(self, log_probs: List[float], rewards: List[float]):
        """Update policy using REINFORCE"""
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()

# Example with CartPole environment
def create_cartpole_env():
    """Create and configure CartPole environment"""
    env = gym.make('CartPole-v1')
    return env

# Note: This would require gym installation
# env = create_cartpole_env()
# state_size = 4  # CartPole state size
# action_size = 2  # CartPole action size
# reinforce_agent = REINFORCE(env, state_size, action_size)

# Training loop (commented out as it requires gym)
"""
episode_rewards = []
for episode in range(1000):
    log_probs, rewards, total_reward = reinforce_agent.train_episode()
    reinforce_agent.update_policy(log_probs, rewards)
    episode_rewards.append(total_reward)
    
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
"""
```

---

## 🌍 Real-World Applications

### 1. **Game AI and Strategy**

```python
class GameRLAgent:
    """RL agent for game playing"""
    
    def __init__(self, game_env, learning_rate=0.001):
        self.env = game_env
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.lr = learning_rate
        
    def train_on_game(self, episodes=1000):
        """Train agent on game episodes"""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
```

### 2. **Robotics and Control**

```python
class RobotControlRL:
    """RL for robotic control systems"""
    
    def __init__(self, robot_env):
        self.env = robot_env
        self.policy_network = PolicyNetwork(
            state_size=robot_env.observation_space.shape[0],
            action_size=robot_env.action_space.shape[0]
        )
        
    def train_robotic_task(self, task_name: str):
        """Train robot for specific task"""
        print(f"Training robot for task: {task_name}")
        # Implementation would include task-specific reward shaping
        # and safety constraints
```

### 3. **Autonomous Systems**

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
        
    def safe_action_selection(self, state, action_probs):
        """Select action with safety constraints"""
        # Check safety constraints before action execution
        if self.check_safety_violation(state, action_probs):
            return self.get_safe_action(state)
        return np.argmax(action_probs)
```

### 4. **Resource Management**

```python
class ResourceManagementRL:
    """RL for dynamic resource allocation"""
    
    def __init__(self, resource_env):
        self.env = resource_env
        self.resource_constraints = resource_env.get_constraints()
        
    def optimize_allocation(self, current_demand, available_resources):
        """Optimize resource allocation using RL"""
        state = self.create_state_vector(current_demand, available_resources)
        action = self.select_optimal_action(state)
        return self.decode_allocation_action(action)
```

---

## 🧪 Exercises and Projects

### Exercise 1: Grid World Navigation

Create a custom grid world environment and implement Q-learning to find the optimal path.

```python
class CustomGridWorld:
    """Custom grid world with obstacles and multiple goals"""
    
    def __init__(self, size=8):
        self.size = size
        self.grid = np.zeros((size, size))
        self.obstacles = [(2, 2), (3, 3), (4, 4)]
        self.goals = [(7, 7), (0, 7)]
        self.current_pos = (0, 0)
        
        # Set obstacles and goals
        for obs in self.obstacles:
            self.grid[obs] = -1
        for goal in self.goals:
            self.grid[goal] = 1
    
    def reset(self):
        self.current_pos = (0, 0)
        return self.current_pos
    
    def step(self, action):
        # Action: 0=right, 1=down, 2=left, 3=up
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = moves[action]
        
        new_x = max(0, min(self.size-1, self.current_pos[0] + dx))
        new_y = max(0, min(self.size-1, self.current_pos[1] + dy))
        
        # Check if new position is obstacle
        if (new_x, new_y) in self.obstacles:
            return self.current_pos, -10, False
        
        self.current_pos = (new_x, new_y)
        
        # Check if reached goal
        if self.current_pos in self.goals:
            return self.current_pos, 100, True
        
        return self.current_pos, -1, False

# Your task: Implement Q-learning for this environment
# and find the optimal path to the goals
```

### Exercise 2: Multi-Agent Coordination

Implement a simple multi-agent system where agents must coordinate to achieve a common goal.

```python
class MultiAgentEnvironment:
    """Environment with multiple agents"""
    
    def __init__(self, num_agents=3, grid_size=5):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agent_positions = [(0, 0)] * num_agents
        self.target = (grid_size-1, grid_size-1)
        
    def reset(self):
        self.agent_positions = [(0, 0)] * self.num_agents
        return self.agent_positions
    
    def step(self, actions):
        # actions is a list of actions for each agent
        rewards = []
        new_positions = []
        
        for i, action in enumerate(actions):
            # Move agent
            new_pos = self.move_agent(self.agent_positions[i], action)
            new_positions.append(new_pos)
            
            # Calculate reward
            if new_pos == self.target:
                rewards.append(10)
            else:
                rewards.append(-1)
        
        self.agent_positions = new_positions
        done = all(pos == self.target for pos in new_positions)
        
        return new_positions, rewards, done

# Your task: Implement a multi-agent Q-learning algorithm
# where agents learn to coordinate their movements
```

### Exercise 3: Continuous Control with Policy Gradients

Implement a continuous control environment and train an agent using policy gradients.

```python
class ContinuousControlEnv:
    """Simple continuous control environment"""
    
    def __init__(self, state_dim=4, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = np.random.randn(state_dim)
        self.target = np.zeros(state_dim)
        
    def reset(self):
        self.state = np.random.randn(self.state_dim)
        return self.state
    
    def step(self, action):
        # Simple dynamics: state += action
        self.state += action * 0.1
        
        # Reward based on distance to target
        distance = np.linalg.norm(self.state - self.target)
        reward = -distance
        
        done = distance < 0.1
        return self.state, reward, done

# Your task: Implement a policy gradient algorithm
# for this continuous control problem
```

### Project: Trading Bot with RL

Create a reinforcement learning trading bot that learns to make profitable trades.

```python
class TradingEnvironment:
    """RL environment for trading"""
    
    def __init__(self, price_data, initial_balance=10000):
        self.price_data = price_data
        self.balance = initial_balance
        self.shares = 0
        self.current_step = 0
        
    def reset(self):
        self.balance = 10000
        self.shares = 0
        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        # State includes: current price, balance, shares, price history
        return np.array([
            self.price_data[self.current_step],
            self.balance,
            self.shares,
            # Add more features as needed
        ])
    
    def step(self, action):
        # Action: 0=hold, 1=buy, 2=sell
        current_price = self.price_data[self.current_step]
        
        if action == 1 and self.balance >= current_price:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2 and self.shares > 0:  # Sell
            self.balance += self.shares * current_price
            self.shares = 0
        
        self.current_step += 1
        
        # Calculate reward (portfolio value change)
        portfolio_value = self.balance + self.shares * current_price
        reward = portfolio_value - 10000  # Relative to initial balance
        
        done = self.current_step >= len(self.price_data) - 1
        
        return self.get_state(), reward, done

# Your task: Implement Q-learning or policy gradients
# to train a profitable trading strategy
```

---

## 📚 Further Reading

### Books
- **"Reinforcement Learning: An Introduction"** by Sutton & Barto
- **"Deep Reinforcement Learning"** by Pieter Abbeel
- **"Algorithms for Reinforcement Learning"** by Csaba Szepesvári

### Papers
- **"Playing Atari with Deep Reinforcement Learning"** (DQN)
- **"Trust Region Policy Optimization"** (TRPO)
- **"Proximal Policy Optimization Algorithms"** (PPO)

### Online Resources
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Deep RL Course by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [RL Course by Sergey Levine](http://rail.eecs.berkeley.edu/deeprlcourse/)

### Tools and Frameworks
- **Stable Baselines3**: Production-ready RL implementations
- **RLlib**: Scalable RL library
- **OpenAI Gym**: Standard RL environments
- **MuJoCo**: Physics-based simulation

---

## 🎯 Key Takeaways

1. **Markov Decision Processes** provide the mathematical foundation for RL
2. **Value functions** estimate expected returns and guide decision-making
3. **Q-Learning** learns optimal action-value functions through experience
4. **Policy gradients** directly optimize policies for better performance
5. **Exploration vs exploitation** is crucial for effective learning
6. **Real-world applications** span gaming, robotics, finance, and more

The next module will cover advanced RL topics including Deep RL, multi-agent systems, and inverse reinforcement learning! 