"""
Kuramoto Oscillator Model for Neural Synchronization

Implements the Kuramoto model for coupled phase oscillators:
    dθᵢ/dt = ωᵢ + (K/N)Σⱼsin(θⱼ - θᵢ) + Iᵢ(t)

Based on Hall et al. (2025), Frontiers in Computational Neuroscience.

Author: Clemens Bauer
Date: April 2026
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class KuramotoParams:
    """Parameters for Kuramoto model"""
    n_oscillators: int = 31
    coupling_strength: float = 5.0
    freq_mean: float = 10.0
    freq_std: float = 2.0
    dt: float = 0.01
    
    
class KuramotoModel:
    """Kuramoto oscillator model for neural synchronization."""
    
    def __init__(self, params: Optional[KuramotoParams] = None):
        self.params = params if params is not None else KuramotoParams()
        
        # Initialize natural frequencies
        self.natural_freqs = np.random.normal(
            self.params.freq_mean,
            self.params.freq_std,
            self.params.n_oscillators
        )
        
        # Initialize phases randomly in [0, 2π]
        self.phases = np.random.uniform(0, 2*np.pi, self.params.n_oscillators)
        
        self.phase_history = []
        self.sync_history = []
        
    def compute_order_parameter(self) -> Tuple[float, float]:
        """Compute Kuramoto order parameter R(t)."""
        complex_phases = np.exp(1j * self.phases)
        mean_field = np.mean(complex_phases)
        R = np.abs(mean_field)
        Psi = np.angle(mean_field)
        return R, Psi
    
    def step(self, external_input: Optional[np.ndarray] = None) -> float:
        """Integrate phase dynamics for one timestep."""
        if external_input is None:
            external_input = np.zeros(self.params.n_oscillators)
            
        N = self.params.n_oscillators
        K = self.params.coupling_strength
        dt = self.params.dt
        
        # CORRECT: (K/N)Σⱼsin(θⱼ - θᵢ)
        phase_diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        coupling_term = (K / N) * np.sum(np.sin(phase_diff), axis=1)
        
        # dθ/dt = ω + coupling + I(t)
        dtheta_dt = self.natural_freqs + coupling_term + external_input
        
        # Euler integration
        self.phases += dtheta_dt * dt
        self.phases = self.phases % (2 * np.pi)
        
        R, _ = self.compute_order_parameter()
        return R
    
    def simulate(
        self,
        duration: float,
        external_input_func: Optional[callable] = None,
        store_history: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Kuramoto dynamics."""
        n_steps = int(duration / self.params.dt)
        
        sync_trajectory = np.zeros(n_steps)
        phase_trajectory = np.zeros((n_steps, self.params.n_oscillators))
        
        for t_idx in range(n_steps):
            current_time = t_idx * self.params.dt
            
            if external_input_func is not None:
                ext_input = external_input_func(current_time)
            else:
                ext_input = None
            
            R = self.step(external_input=ext_input)
            
            sync_trajectory[t_idx] = R
            phase_trajectory[t_idx, :] = self.phases.copy()
        
        if store_history:
            self.sync_history = sync_trajectory
            self.phase_history = phase_trajectory
        
        return sync_trajectory, phase_trajectory
    
    def reset(self, random_phases: bool = True):
        """Reset model to initial conditions."""
        if random_phases:
            self.phases = np.random.uniform(0, 2*np.pi, self.params.n_oscillators)
        else:
            self.phases = np.zeros(self.params.n_oscillators)
        
        self.phase_history = []
        self.sync_history = []


if __name__ == "__main__":
    print("Testing Kuramoto Model (Hall et al. 2025)")
    print("=" * 60)
    
    conditions = {
        'Focused (K=10)': {'K': 10.0, 'input': lambda t: np.ones(31) * 5.0},
        'Multitask (K=5)': {'K': 5.0, 'input': lambda t: np.concatenate([np.ones(15)*5.0, -np.ones(16)*5.0])},
        'Rest (K=1)': {'K': 1.0, 'input': None}
    }
    
    for name, cfg in conditions.items():
        model = KuramotoModel(KuramotoParams(coupling_strength=cfg['K']))
        sync, _ = model.simulate(duration=10.0, external_input_func=cfg['input'])
        print(f"\n{name}:")
        print(f"  Mean R: {np.mean(sync):.3f}, Max R: {np.max(sync):.3f}, Final R: {sync[-1]:.3f}")