"""
Reinforcement Learning Agents for Neural Synchronization Control

Implements Q-learning and Deep Q-Network (DQN) agents that learn
to modulate external input to optimize synchronization and energy.

Based on Hall et al. (2025) Section 3.2.4-3.2.5.

Author: Clemens Bauer
Date: April 2026
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from collections import deque
import pickle


@dataclass
class AgentParams:
    """Parameters for RL agents"""
    learning_rate: float = 0.1      # η
    discount_factor: float = 0.95   # γ
    epsilon: float = 0.1            # ε-greedy exploration
    epsilon_decay: float = 0.995    # Decay rate
    epsilon_min: float = 0.01       # Minimum exploration
    
    # State discretization for tabular Q-learning
    n_sync_bins: int = 20           # Discretize R ∈ [0,1]
    n_energy_bins: int = 20         # Discretize E
    
    # Action space
    n_actions: int = 11             # Discrete input levels
    action_min: float = -5.0        # Minimum input
    action_max: float = 5.0         # Maximum input
    
    # Target synchronization
    r_target: float = 0.9           # Desired R(t) for focused state


class QLearningAgent:
    """
    Tabular Q-learning agent for synchronization control.
    
    State: (R(t), E(t)) discretized into bins
    Action: External input level
    Reward: -|R_target - R(t)| - E(t)
    """
    
    def __init__(self, params: Optional[AgentParams] = None):
        """
        Initialize Q-learning agent.
        
        Args:
            params: Agent hyperparameters
        """
        self.params = params if params is not None else AgentParams()
        
        # Initialize Q-table Q(s, a)
        self.q_table = np.zeros((
            self.params.n_sync_bins,
            self.params.n_energy_bins,
            self.params.n_actions
        ))
        
        # Action space
        self.actions = np.linspace(
            self.params.action_min,
            self.params.action_max,
            self.params.n_actions
        )
        
        # Training history
        self.episode_rewards = []
        self.epsilon_history = []
        self.current_epsilon = self.params.epsilon
        
    def discretize_state(self, R: float, E: float) -> Tuple[int, int]:
        """
        Discretize continuous state into bins.
        
        Args:
            R: Synchronization level [0, 1]
            E: Energy cost (normalized)
        
        Returns:
            (sync_bin, energy_bin): Discretized state indices
        """
        sync_bin = int(np.clip(R, 0, 0.999) * self.params.n_sync_bins)
        energy_bin = int(np.clip(E / 10.0, 0, 0.999) * self.params.n_energy_bins)
        
        return sync_bin, energy_bin
    
    def compute_reward(self, R: float, E: float) -> float:
        """
        Compute reward signal.
        
        r = -|R_target - R(t)| - E(t)
        
        Args:
            R: Current synchronization
            E: Current energy cost
        
        Returns:
            reward: Scalar reward signal
        """
        sync_error = abs(self.params.r_target - R)
        reward = -sync_error - E
        
        return reward
    
    def select_action(self, R: float, E: float) -> Tuple[int, float]:
        """
        Select action using ε-greedy policy.
        
        Args:
            R: Current synchronization
            E: Current energy
        
        Returns:
            action_idx: Action index
            action_value: Continuous action value (input level)
        """
        sync_bin, energy_bin = self.discretize_state(R, E)
        
        # ε-greedy: explore with probability ε
        if np.random.rand() < self.current_epsilon:
            action_idx = np.random.randint(self.params.n_actions)
        else:
            action_idx = np.argmax(self.q_table[sync_bin, energy_bin, :])
        
        action_value = self.actions[action_idx]
        
        return action_idx, action_value
    
    def update(
        self,
        R: float,
        E: float,
        action_idx: int,
        R_next: float,
        E_next: float,
        reward: float
    ):
        """
        Update Q-values using Bellman equation.
        
        Q(s,a) ← Q(s,a) + η[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            R, E: Current state
            action_idx: Action taken
            R_next, E_next: Next state
            reward: Observed reward
        """
        s_bin, e_bin = self.discretize_state(R, E)
        s_next_bin, e_next_bin = self.discretize_state(R_next, E_next)
        
        # Current Q-value
        q_current = self.q_table[s_bin, e_bin, action_idx]
        
        # Max Q-value for next state
        q_next_max = np.max(self.q_table[s_next_bin, e_next_bin, :])
        
        # TD error
        td_target = reward + self.params.discount_factor * q_next_max
        td_error = td_target - q_current
        
        # Q-value update
        self.q_table[s_bin, e_bin, action_idx] += self.params.learning_rate * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.current_epsilon = max(
            self.params.epsilon_min,
            self.current_epsilon * self.params.epsilon_decay
        )
        self.epsilon_history.append(self.current_epsilon)
    
    def save(self, filepath: str):
        """Save Q-table to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'params': self.params,
                'episode_rewards': self.episode_rewards
            }, f)
    
    def load(self, filepath: str):
        """Load Q-table from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.params = data['params']
            self.episode_rewards = data.get('episode_rewards', [])


class DQNAgent:
    """
    Deep Q-Network agent using neural network function approximator.
    
    Uses experience replay and target network for stable learning.
    Suitable for continuous state spaces.
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 11,
        params: Optional[AgentParams] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state vector (default: R, E)
            action_dim: Number of discrete actions
            params: Agent hyperparameters
        """
        self.params = params if params is not None else AgentParams()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Action space
        self.actions = np.linspace(
            self.params.action_min,
            self.params.action_max,
            self.params.n_actions
        )
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Neural network weights (simplified 2-layer network)
        # Input: [R, E] → Hidden: 64 → Hidden: 64 → Output: n_actions
        self.w1 = np.random.randn(state_dim, 64) * 0.1
        self.b1 = np.zeros(64)
        self.w2 = np.random.randn(64, 64) * 0.1
        self.b2 = np.zeros(64)
        self.w3 = np.random.randn(64, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)
        
        # Target network (frozen copy)
        self.target_w1 = self.w1.copy()
        self.target_w2 = self.w2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b1 = self.b1.copy()
        self.target_b2 = self.b2.copy()
        self.target_b3 = self.b3.copy()
        
        # Training history
        self.episode_rewards = []
        self.loss_history = []
        self.current_epsilon = self.params.epsilon
    
    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def predict(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """
        Forward pass through Q-network.
        
        Args:
            state: State vector [state_dim]
            use_target: Use target network if True
        
        Returns:
            q_values: Q-values for all actions [action_dim]
        """
        if use_target:
            w1, b1 = self.target_w1, self.target_b1
            w2, b2 = self.target_w2, self.target_b2
            w3, b3 = self.target_w3, self.target_b3
        else:
            w1, b1 = self.w1, self.b1
            w2, b2 = self.w2, self.b2
            w3, b3 = self.w3, self.b3
        
        # Forward pass
        h1 = self._relu(state @ w1 + b1)
        h2 = self._relu(h1 @ w2 + b2)
        q_values = h2 @ w3 + b3
        
        return q_values
    
    def select_action(self, R: float, E: float) -> Tuple[int, float]:
        """
        Select action using ε-greedy policy with DQN.
        
        Args:
            R: Current synchronization
            E: Current energy
        
        Returns:
            action_idx: Action index
            action_value: Continuous action value
        """
        state = np.array([R, E / 10.0])  # Normalize energy
        
        if np.random.rand() < self.current_epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            q_values = self.predict(state)
            action_idx = np.argmax(q_values)
        
        action_value = self.actions[action_idx]
        
        return action_idx, action_value
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def compute_reward(self, R: float, E: float) -> float:
        """Same reward function as Q-learning."""
        sync_error = abs(self.params.r_target - R)
        return -sync_error - E
    
    def train_step(self) -> float:
        """
        Sample batch and perform gradient descent update.
        
        Returns:
            loss: Mean squared TD error
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Compute target Q-values
        q_next = np.array([self.predict(s, use_target=True) for s in next_states])
        q_target = rewards + self.params.discount_factor * np.max(q_next, axis=1) * (1 - dones)
        
        # Compute current Q-values
        q_current = np.array([self.predict(s) for s in states])
        q_pred = q_current[np.arange(self.batch_size), actions]
        
        # Loss (MSE)
        loss = np.mean((q_target - q_pred) ** 2)
        
        # Gradient descent (simplified - not actual backprop)
        # In practice, use PyTorch or TensorFlow for proper gradients
        error = q_pred - q_target
        
        # Simple weight update (placeholder)
        lr = self.params.learning_rate
        self.w3 -= lr * 0.01 * np.outer(error, np.ones(64))
        
        return loss
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_w1 = self.w1.copy()
        self.target_w2 = self.w2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b1 = self.b1.copy()
        self.target_b2 = self.b2.copy()
        self.target_b3 = self.b3.copy()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.current_epsilon = max(
            self.params.epsilon_min,
            self.current_epsilon * self.params.epsilon_decay
        )


# Example usage
if __name__ == "__main__":
    # Initialize Q-learning agent
    agent = QLearningAgent()
    
    # Simulate one episode
    R, E = 0.5, 3.0
    action_idx, action_val = agent.select_action(R, E)
    print(f"Selected action: {action_val:.2f} Hz input")
    
    # Simulate environment response
    R_next = 0.6  # Increased synchronization
    E_next = 3.5  # Slightly higher energy
    reward = agent.compute_reward(R_next, E_next)
    
    # Update Q-values
    agent.update(R, E, action_idx, R_next, E_next, reward)
    print(f"Reward: {reward:.3f}")
    
    # DQN example
    dqn = DQNAgent()
    action_idx, action_val = dqn.select_action(R, E)
    print(f"DQN selected action: {action_val:.2f} Hz")
