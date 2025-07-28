# Agentic AI

## Overview
Agentic AI refers to artificial intelligence systems that can act autonomously in complex environments, make decisions, and achieve goals with minimal human intervention. These systems combine perception, reasoning, planning, and action to operate effectively in dynamic, uncertain environments.

## Agent Fundamentals

### Agent Architecture
```python
class Agent:
    def __init__(self, state_dim, action_dim, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.memory_size = memory_size
        
    def perceive(self, environment):
        """Perceive the current state of the environment"""
        return environment.get_state()
    
    def think(self, state):
        """Process information and make decisions"""
        raise NotImplementedError
    
    def act(self, action):
        """Execute actions in the environment"""
        return action
    
    def learn(self, experience):
        """Learn from experience"""
        self.memory.append(experience)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
```

### Agent Types
- **Reactive Agents**: Simple stimulus-response behavior
- **Deliberative Agents**: Plan before acting
- **B-DI Agents**: Belief-Desire-Intention architecture
- **Learning Agents**: Adapt behavior through experience
- **Multi-Agent Systems**: Multiple agents working together

## Reinforcement Learning Agents

### 1. Q-Learning Agent
```python
import numpy as np
import torch
import torch.nn as nn

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_state_key(self, state):
        """Convert state to hashable key"""
        return tuple(state)
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        
        if done:
            target_q = reward
        else:
            target_q = reward + 0.9 * max_next_q
        
        self.q_table[state_key][action] = current_q + self.lr * (target_q - current_q)
```

### 2. Deep Q-Network (DQN) Agent
```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.epsilon = epsilon
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = []
        self.batch_size = 32
        
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def train(self):
        """Train the Q-network"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = torch.LongTensor([self.memory[i][1] for i in batch])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.BoolTensor([self.memory[i][4] for i in batch])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.9 * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

### 3. Actor-Critic Agent
```python
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
    def get_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def update(self, states, actions, rewards, next_states, dones):
        """Update actor and critic networks"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.9 * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Critic loss
        values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        
        # Actor loss
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        advantages = returns - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        
        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
```

## Multi-Agent Systems

### 1. Cooperative Multi-Agent System
```python
class CooperativeMultiAgent:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.agents = [DQNAgent(state_dim, action_dim) for _ in range(num_agents)]
        
    def get_joint_action(self, joint_state):
        """Get actions from all agents"""
        actions = []
        for i, agent in enumerate(self.agents):
            state = joint_state[i]
            action = agent.get_action(state)
            actions.append(action)
        return actions
    
    def update_agents(self, joint_experience):
        """Update all agents with shared experience"""
        for i, agent in enumerate(self.agents):
            state, action, reward, next_state, done = joint_experience[i]
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
```

### 2. Competitive Multi-Agent System
```python
class CompetitiveMultiAgent:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.agents = [DQNAgent(state_dim, action_dim) for _ in range(num_agents)]
        
    def get_competitive_actions(self, joint_state):
        """Get competitive actions from agents"""
        actions = []
        for i, agent in enumerate(self.agents):
            state = joint_state[i]
            action = agent.get_action(state)
            actions.append(action)
        return actions
    
    def update_with_competition(self, joint_experience, rewards):
        """Update agents considering competitive rewards"""
        for i, agent in enumerate(self.agents):
            state, action, reward, next_state, done = joint_experience[i]
            # Adjust reward based on competition
            competitive_reward = reward - np.mean(rewards) + rewards[i]
            agent.store_experience(state, action, competitive_reward, next_state, done)
            agent.train()
```

## Planning Agents

### 1. A* Planning Agent
```python
import heapq

class AStarAgent:
    def __init__(self, environment):
        self.environment = environment
        
    def heuristic(self, state, goal):
        """Manhattan distance heuristic"""
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])
    
    def get_neighbors(self, state):
        """Get valid neighboring states"""
        x, y = state
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.environment.width and 
                0 <= new_y < self.environment.height and
                not self.environment.is_obstacle(new_x, new_y)):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def plan_path(self, start, goal):
        """Find optimal path using A*"""
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
```

### 2. Monte Carlo Tree Search (MCTS) Agent
```python
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        
    def is_fully_expanded(self):
        return len(self.children) == len(self.get_possible_actions())
    
    def get_possible_actions(self):
        # Define possible actions for the state
        return [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def get_ucb1_value(self, exploration_constant=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits + 
                exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits))

class MCTSAgent:
    def __init__(self, environment, iterations=1000):
        self.environment = environment
        self.iterations = iterations
        
    def select_action(self, state):
        """Select action using MCTS"""
        root = MCTSNode(state)
        
        for _ in range(self.iterations):
            node = root
            
            # Selection
            while node.is_fully_expanded() and not self.is_terminal(node.state):
                node = self.select_child(node)
            
            # Expansion
            if not self.is_terminal(node.state):
                node = self.expand_node(node)
            
            # Simulation
            result = self.simulate(node.state)
            
            # Backpropagation
            self.backpropagate(node, result)
        
        # Choose best action
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def select_child(self, node):
        """Select child using UCB1"""
        return max(node.children, key=lambda c: c.get_ucb1_value())
    
    def expand_node(self, node):
        """Expand node with new child"""
        possible_actions = node.get_possible_actions()
        used_actions = [child.action for child in node.children]
        available_actions = [a for a in possible_actions if a not in used_actions]
        
        if available_actions:
            action = random.choice(available_actions)
            new_state = self.get_next_state(node.state, action)
            child = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child)
            return child
        
        return node
    
    def simulate(self, state):
        """Simulate random playout"""
        current_state = state
        while not self.is_terminal(current_state):
            actions = self.get_possible_actions(current_state)
            action = random.choice(actions)
            current_state = self.get_next_state(current_state, action)
        
        return self.get_reward(current_state)
    
    def backpropagate(self, node, result):
        """Backpropagate result up the tree"""
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        return self.environment.is_goal(state) or self.environment.is_failure(state)
    
    def get_next_state(self, state, action):
        """Get next state after action"""
        # Implementation depends on environment
        return state  # Placeholder
    
    def get_reward(self, state):
        """Get reward for state"""
        if self.environment.is_goal(state):
            return 1.0
        elif self.environment.is_failure(state):
            return -1.0
        return 0.0
```

## Belief-Desire-Intention (BDI) Agents

### 1. BDI Agent Implementation
```python
class BDIAgent:
    def __init__(self):
        self.beliefs = {}  # Knowledge about the world
        self.desires = []  # Goals to achieve
        self.intentions = []  # Plans being executed
        self.plans = {}  # Plan library
        
    def update_beliefs(self, observation):
        """Update beliefs based on observation"""
        self.beliefs.update(observation)
    
    def add_desire(self, desire):
        """Add a new desire/goal"""
        self.desires.append(desire)
    
    def select_intention(self):
        """Select intention from desires"""
        if self.desires and not self.intentions:
            # Simple selection: choose first desire
            intention = self.desires.pop(0)
            self.intentions.append(intention)
            return intention
        return None
    
    def execute_intention(self, intention):
        """Execute current intention"""
        if intention in self.plans:
            plan = self.plans[intention]
            return self.execute_plan(plan)
        return None
    
    def execute_plan(self, plan):
        """Execute a plan"""
        for action in plan:
            result = self.perform_action(action)
            if not result:
                return False
        return True
    
    def perform_action(self, action):
        """Perform a single action"""
        # Implementation depends on environment
        return True  # Placeholder
```

## Intelligent Decision Making

### 1. Utility-Based Decision Making
```python
class UtilityBasedAgent:
    def __init__(self):
        self.utilities = {}
        self.preferences = {}
        
    def set_utility(self, outcome, utility):
        """Set utility for an outcome"""
        self.utilities[outcome] = utility
    
    def set_preference(self, attribute, weight):
        """Set preference weight for an attribute"""
        self.preferences[attribute] = weight
    
    def evaluate_action(self, action, possible_outcomes):
        """Evaluate action based on expected utility"""
        expected_utility = 0
        
        for outcome, probability in possible_outcomes.items():
            if outcome in self.utilities:
                expected_utility += probability * self.utilities[outcome]
        
        return expected_utility
    
    def choose_action(self, actions):
        """Choose action with highest expected utility"""
        best_action = None
        best_utility = float('-inf')
        
        for action in actions:
            utility = self.evaluate_action(action, self.get_outcomes(action))
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        return best_action
    
    def get_outcomes(self, action):
        """Get possible outcomes and probabilities for action"""
        # Implementation depends on domain
        return {}  # Placeholder
```

### 2. Bayesian Decision Making
```python
import numpy as np
from scipy.stats import norm

class BayesianAgent:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.utilities = {}
        
    def set_prior(self, hypothesis, probability):
        """Set prior probability for hypothesis"""
        self.priors[hypothesis] = probability
    
    def set_likelihood(self, evidence, hypothesis, probability):
        """Set likelihood of evidence given hypothesis"""
        if evidence not in self.likelihoods:
            self.likelihoods[evidence] = {}
        self.likelihoods[evidence][hypothesis] = probability
    
    def set_utility(self, action, outcome, utility):
        """Set utility for action-outcome pair"""
        if action not in self.utilities:
            self.utilities[action] = {}
        self.utilities[action][outcome] = utility
    
    def update_beliefs(self, evidence):
        """Update beliefs using Bayes' rule"""
        posteriors = {}
        total_probability = 0
        
        for hypothesis in self.priors:
            likelihood = self.likelihoods.get(evidence, {}).get(hypothesis, 0)
            posterior = self.priors[hypothesis] * likelihood
            posteriors[hypothesis] = posterior
            total_probability += posterior
        
        # Normalize
        for hypothesis in posteriors:
            posteriors[hypothesis] /= total_probability
        
        self.priors = posteriors
        return posteriors
    
    def choose_action(self, actions):
        """Choose action with highest expected utility"""
        best_action = None
        best_expected_utility = float('-inf')
        
        for action in actions:
            expected_utility = 0
            
            for hypothesis, probability in self.priors.items():
                if action in self.utilities and hypothesis in self.utilities[action]:
                    expected_utility += probability * self.utilities[action][hypothesis]
            
            if expected_utility > best_expected_utility:
                best_expected_utility = expected_utility
                best_action = action
        
        return best_action
```

## Agent Communication

### 1. Contract Net Protocol
```python
class ContractNetAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.tasks = []
        self.capabilities = {}
        
    def announce_task(self, task, task_manager):
        """Announce task to other agents"""
        announcement = {
            'type': 'task_announcement',
            'task': task,
            'sender': self.agent_id
        }
        task_manager.broadcast(announcement)
    
    def bid_on_task(self, task):
        """Bid on a task based on capabilities"""
        if task['type'] in self.capabilities:
            cost = self.capabilities[task['type']]
            return {
                'type': 'bid',
                'task': task,
                'cost': cost,
                'sender': self.agent_id
            }
        return None
    
    def award_contract(self, task, contractor):
        """Award contract to winning bidder"""
        contract = {
            'type': 'contract',
            'task': task,
            'contractor': contractor,
            'sender': self.agent_id
        }
        return contract
    
    def execute_contract(self, contract):
        """Execute awarded contract"""
        task = contract['task']
        if task['type'] in self.capabilities:
            # Execute task
            result = self.perform_task(task)
            return {
                'type': 'result',
                'task': task,
                'result': result,
                'sender': self.agent_id
            }
        return None
```

### 2. Message Passing System
```python
class MessagePassingAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.message_queue = []
        self.neighbors = []
        
    def send_message(self, recipient, message):
        """Send message to another agent"""
        message['sender'] = self.agent_id
        message['recipient'] = recipient
        recipient.receive_message(message)
    
    def broadcast_message(self, message):
        """Broadcast message to all neighbors"""
        for neighbor in self.neighbors:
            self.send_message(neighbor, message)
    
    def receive_message(self, message):
        """Receive message from another agent"""
        self.message_queue.append(message)
    
    def process_messages(self):
        """Process all messages in queue"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            self.handle_message(message)
    
    def handle_message(self, message):
        """Handle specific message types"""
        message_type = message.get('type')
        
        if message_type == 'task_request':
            self.handle_task_request(message)
        elif message_type == 'task_response':
            self.handle_task_response(message)
        elif message_type == 'coordination':
            self.handle_coordination(message)
    
    def handle_task_request(self, message):
        """Handle task request from another agent"""
        # Implementation depends on agent capabilities
        pass
    
    def handle_task_response(self, message):
        """Handle task response from another agent"""
        # Implementation depends on agent behavior
        pass
    
    def handle_coordination(self, message):
        """Handle coordination message"""
        # Implementation depends on coordination protocol
        pass
```

## Evaluation Metrics

### 1. Agent Performance Metrics
```python
class AgentEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_agent(self, agent, environment, episodes=100):
        """Evaluate agent performance"""
        total_reward = 0
        success_rate = 0
        average_steps = 0
        
        for episode in range(episodes):
            state = environment.reset()
            episode_reward = 0
            steps = 0
            
            while not environment.is_done():
                action = agent.get_action(state)
                next_state, reward, done = environment.step(action)
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            total_reward += episode_reward
            if episode_reward > 0:
                success_rate += 1
            average_steps += steps
        
        self.metrics = {
            'average_reward': total_reward / episodes,
            'success_rate': success_rate / episodes,
            'average_steps': average_steps / episodes
        }
        
        return self.metrics
```

### 2. Multi-Agent System Metrics
```python
class MultiAgentEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_cooperation(self, agents, environment):
        """Evaluate cooperation in multi-agent system"""
        joint_rewards = []
        coordination_effort = []
        
        for episode in range(100):
            joint_state = environment.reset()
            episode_reward = 0
            coordination_count = 0
            
            while not environment.is_done():
                joint_action = [agent.get_action(state) for agent, state in zip(agents, joint_state)]
                next_joint_state, rewards, done = environment.step(joint_action)
                
                episode_reward += sum(rewards)
                coordination_count += self.count_coordination(joint_action)
                joint_state = next_joint_state
            
            joint_rewards.append(episode_reward)
            coordination_effort.append(coordination_count)
        
        self.metrics = {
            'average_joint_reward': np.mean(joint_rewards),
            'coordination_effort': np.mean(coordination_effort),
            'reward_variance': np.var(joint_rewards)
        }
        
        return self.metrics
    
    def count_coordination(self, joint_action):
        """Count coordination events in joint action"""
        # Implementation depends on coordination definition
        return 0  # Placeholder
```

## Tools and Libraries

- **Gym**: Reinforcement learning environments
- **PettingZoo**: Multi-agent environments
- **Stable Baselines3**: RL algorithms
- **PyTorch**: Deep learning framework
- **NetworkX**: Graph algorithms for planning

## Best Practices

1. **Modular Design**: Separate perception, reasoning, and action
2. **Robust Decision Making**: Handle uncertainty and partial information
3. **Scalable Communication**: Efficient protocols for multi-agent systems
4. **Continuous Learning**: Adapt to changing environments
5. **Safety Considerations**: Ensure safe operation in real-world scenarios

## Next Steps

1. **Hierarchical Agents**: Multi-level decision making
2. **Swarm Intelligence**: Emergent behavior in large groups
3. **Human-Agent Interaction**: Natural language and gesture communication
4. **Autonomous Systems**: Self-driving cars, drones, robots
5. **Agent Programming Languages**: Specialized languages for agent development

---

*Agentic AI represents the frontier of autonomous intelligent systems, combining sophisticated decision-making with robust action execution. From single agents to complex multi-agent systems, these technologies are enabling new forms of automation and cooperation.* 