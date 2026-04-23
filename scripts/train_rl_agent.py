"""
Main Simulation Script: Train RL Agent to Control Kuramoto Synchronization

Integrates:
1. Kuramoto oscillator model
2. Energy function (EEG + fMRI)
3. Q-learning agent
4. Real DMNELF data for validation

Based on Hall et al. (2025) framework.

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.kuramoto import KuramotoModel, KuramotoParams
from models.energy import EnergyFunction, EnergyParams
from models.agents import QLearningAgent, AgentParams
from data_loaders.eeg_loader import EEGLoader
from data_loaders.fmri_loader import fMRILoader


class SynchronizationEnvironment:
    """
    Environment for RL agent to control neural synchronization.
    
    State: (R(t), E(t)) - synchronization and energy
    Action: External input level
    Reward: -|R_target - R(t)| - E(t)
    """
    
    def __init__(
        self,
        kuramoto_params: KuramotoParams,
        energy_params: EnergyParams,
        episode_duration: float = 10.0,
        target_sync: float = 0.9
    ):
        """
        Initialize environment.
        
        Args:
            kuramoto_params: Kuramoto model parameters
            energy_params: Energy function parameters
            episode_duration: Episode length in seconds
            target_sync: Target synchronization level R_target
        """
        self.kuramoto = KuramotoModel(kuramoto_params)
        self.energy_fn = EnergyFunction(energy_params)
        
        self.episode_duration = episode_duration
        self.target_sync = target_sync
        
        self.dt = kuramoto_params.dt
        self.n_steps = int(episode_duration / self.dt)
        
        # Episode state
        self.current_step = 0
        self.sync_history = []
        self.energy_history = []
        self.action_history = []
    
    def reset(self) -> tuple:
        """
        Reset environment to initial state.
        
        Returns:
            (R, E): Initial state
        """
        self.kuramoto.reset(random_phases=True)
        self.current_step = 0
        
        # Initialize with one step
        R = self.kuramoto.step()
        
        # Compute initial energy (placeholder)
        self.sync_history = [R]
        E = self.energy_fn.compute_energy(np.array([R]), dt=self.dt)[0]
        self.energy_history = [E]
        self.action_history = []
        
        return R, E
    
    def step(self, action: float) -> tuple:
        """
        Take one step in environment.
        
        Args:
            action: External input level (uniform across oscillators)
        
        Returns:
            (R_next, E_next, reward, done)
        """
        # Apply action as external input
        external_input = np.ones(self.kuramoto.params.n_oscillators) * action
        
        # Step Kuramoto model
        R_next = self.kuramoto.step(external_input)
        
        # Compute energy (using simulated features)
        self.sync_history.append(R_next)
        sync_array = np.array(self.sync_history)
        E_array = self.energy_fn.compute_energy(sync_array, dt=self.dt)
        E_next = E_array[-1]
        self.energy_history.append(E_next)
        self.action_history.append(action)
        
        # Compute reward
        reward = -abs(self.target_sync - R_next) - E_next
        
        # Check if episode done
        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        return R_next, E_next, reward, done


def train_agent(
    agent: QLearningAgent,
    env: SynchronizationEnvironment,
    n_episodes: int = 100,
    verbose: bool = True
) -> dict:
    """
    Train RL agent to control synchronization.
    
    Args:
        agent: Q-learning agent
        env: Synchronization environment
        n_episodes: Number of training episodes
        verbose: Print progress
    
    Returns:
        results: Training metrics
    """
    episode_rewards = []
    episode_sync_means = []
    episode_energy_means = []
    
    iterator = tqdm(range(n_episodes)) if verbose else range(n_episodes)
    
    for episode in iterator:
        # Reset environment
        R, E = env.reset()
        
        episode_reward = 0.0
        
        while True:
            # Agent selects action
            action_idx, action_value = agent.select_action(R, E)
            
            # Environment step
            R_next, E_next, reward, done = env.step(action_value)
            
            # Agent update
            agent.update(R, E, action_idx, R_next, E_next, reward)
            
            # Accumulate
            episode_reward += reward
            
            # Next state
            R, E = R_next, E_next
            
            if done:
                break
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_sync_means.append(np.mean(env.sync_history))
        episode_energy_means.append(np.mean(env.energy_history))
        
        if verbose and (episode + 1) % 10 == 0:
            iterator.set_description(
                f"Ep {episode+1}: Reward={episode_reward:.2f}, "
                f"R̄={episode_sync_means[-1]:.3f}, "
                f"Ē={episode_energy_means[-1]:.2f}"
            )
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_sync_means': episode_sync_means,
        'episode_energy_means': episode_energy_means,
        'epsilon_history': agent.epsilon_history
    }
    
    return results


def plot_training_results(results: dict, save_path: Optional[str] = None):
    """
    Plot training curves.
    
    Args:
        results: Training metrics from train_agent()
        save_path: Path to save figure (None = display)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Cumulative reward
    axes[0, 0].plot(results['episode_rewards'])
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].grid(alpha=0.3)
    
    # Mean synchronization
    axes[0, 1].plot(results['episode_sync_means'])
    axes[0, 1].axhline(0.9, color='r', linestyle='--', label='Target R')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Mean R(t)')
    axes[0, 1].set_title('Synchronization Level')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Mean energy
    axes[1, 0].plot(results['episode_energy_means'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Mean E(t)')
    axes[1, 0].set_title('Energy Consumption')
    axes[1, 0].grid(alpha=0.3)
    
    # Exploration rate
    axes[1, 1].plot(results['epsilon_history'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('ε')
    axes[1, 1].set_title('Exploration Rate')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    """Main training loop."""
    
    print("=" * 60)
    print("Adaptive Neural Synchronization - RL Training")
    print("=" * 60)
    
    # Initialize models
    print("\n1. Initializing models...")
    
    kuramoto_params = KuramotoParams(
        n_oscillators=31,  # DMNELF has 31 channels
        coupling_strength=5.0,
        freq_mean=10.0,
        freq_std=2.0,
        dt=0.01
    )
    
    energy_params = EnergyParams(
        alpha=10.01,
        beta=5.00,
        gamma=3.00,
        delta=2.00
    )
    
    agent_params = AgentParams(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
        r_target=0.9
    )
    
    # Create environment
    env = SynchronizationEnvironment(
        kuramoto_params,
        energy_params,
        episode_duration=10.0,
        target_sync=0.9
    )
    print("  ✓ Environment created")
    
    # Create agent
    agent = QLearningAgent(agent_params)
    print("  ✓ Q-learning agent initialized")
    
    # Train
    print("\n2. Training agent...")
    results = train_agent(
        agent,
        env,
        n_episodes=100,
        verbose=True
    )
    
    # Plot results
    print("\n3. Plotting results...")
    plot_training_results(results, save_path='./training_results.png')
    
    # Save agent
    print("\n4. Saving trained agent...")
    agent.save('./trained_qlearning_agent.pkl')
    print("  ✓ Saved to trained_qlearning_agent.pkl")
    
    # Final stats
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final episode reward: {results['episode_rewards'][-1]:.2f}")
    print(f"Final mean R(t): {results['episode_sync_means'][-1]:.3f} (target: 0.9)")
    print(f"Final mean E(t): {results['episode_energy_means'][-1]:.2f}")
    print(f"Final ε: {results['epsilon_history'][-1]:.3f}")


if __name__ == "__main__":
    main()
