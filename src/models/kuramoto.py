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
    n_oscillators: int = 31  # Number of oscillators (EEG channels)
    coupling_strength: float = 5.0  # K parameter
    freq_mean: float = 10.0  # Mean natural frequency (Hz)
    freq_std: float = 2.0    # Std of natural frequencies
    dt: float = 0.01         # Integration timestep (s)
    
    
class KuramotoModel:
    """
    Kuramoto oscillator model for neural synchronization.
    
    Simulates phase dynamics of coupled oscillators representing neurons/channels.
    Computes order parameter R(t) to quantify synchronization level.
    """
    
    def __init__(self, params: Optional[KuramotoParams] = None):
        """
        Initialize Kuramoto model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params if params is not None else KuramotoParams()
        
        # Initialize natural frequencies from Gaussian distribution
        self.natural_freqs = np.random.normal(
            self.params.freq_mean,
            self.params.freq_std,
            self.params.n_oscillators
        )
        
        # Initialize phases randomly in [0, 2π]
        self.phases = np.random.uniform(0, 2*np.pi, self.params.n_oscillators)
        
        # History storage
        self.phase_history = []
        self.sync_history = []
        
    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter R(t).
        
        R(t) = |1/N Σⱼ e^(iθⱼ)| measures global synchronization.
        
        Returns:
            R: Order parameter magnitude (0=desync, 1=perfect sync)
            Ψ: Mean phase direction
        """
        # Complex representation: e^(iθ)
        complex_phases = np.exp(1j * self.phases)
        mean_field = np.mean(complex_phases)
        
        R = np.abs(mean_field)  # Synchronization level
        Psi = np.angle(mean_field)  # Mean phase
        
        return R, Psi
    
    def step(self, external_input: Optional[np.ndarray] = None) -> float:
        """
        Integrate phase dynamics for one timestep using Euler method.
        
        Args:
            external_input: External input I(t) for each oscillator [n_oscillators]
                           If None, uses zero input
        
        Returns:
            R: Current synchronization level
        """
        if external_input is None:
            external_input = np.zeros(self.params.n_oscillators)
            
        N = self.params.n_oscillators
        K = self.params.coupling_strength
        dt = self.params.dt
        
        # Compute coupling term: (K/N)Σⱼsin(θⱼ - θᵢ)
        # Using broadcasting to compute all pairwise differences
        phase_diff = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
        coupling_term = (K / N) * np.sum(np.sin(phase_diff), axis=1)
        
        # Phase derivatives: dθ/dt = ω + coupling + I(t)
        dtheta_dt = self.natural_freqs + coupling_term + external_input
        
        # Euler integration
        self.phases += dtheta_dt * dt
        
        # Wrap phases to [0, 2π]
        self.phases = self.phases % (2 * np.pi)
        
        # Compute synchronization
        R, _ = self.compute_order_parameter()
        
        return R
    
    def simulate(
        self,
        duration: float,
        external_input_func: Optional[callable] = None,
        store_history: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Kuramoto dynamics over specified duration.
        
        Args:
            duration: Simulation time in seconds
            external_input_func: Function f(t) -> [n_oscillators] providing input
                                If None, no external input
            store_history: Whether to store phase/sync trajectories
        
        Returns:
            sync_trajectory: Time series of R(t) values [n_steps]
            phase_trajectory: Phase values over time [n_steps, n_oscillators]
        """
        n_steps = int(duration / self.params.dt)
        
        sync_trajectory = np.zeros(n_steps)
        phase_trajectory = np.zeros((n_steps, self.params.n_oscillators))
        
        for t_idx in range(n_steps):
            current_time = t_idx * self.params.dt
            
            # Get external input at current time
            if external_input_func is not None:
                ext_input = external_input_func(current_time)
            else:
                ext_input = None
            
            # Step forward
            R = self.step(external_input=ext_input)
            
            # Store
            sync_trajectory[t_idx] = R
            phase_trajectory[t_idx, :] = self.phases.copy()
        
        if store_history:
            self.sync_history = sync_trajectory
            self.phase_history = phase_trajectory
        
        return sync_trajectory, phase_trajectory
    
    def reset(self, random_phases: bool = True):
        """
        Reset model to initial conditions.
        
        Args:
            random_phases: If True, randomize phases; else set to zero
        """
        if random_phases:
            self.phases = np.random.uniform(0, 2*np.pi, self.params.n_oscillators)
        else:
            self.phases = np.zeros(self.params.n_oscillators)
        
        self.phase_history = []
        self.sync_history = []
    
    def set_cognitive_state(self, state: str):
        """
        Configure model parameters for specific cognitive states.
        
        Args:
            state: One of 'resting', 'focused', 'multitasking'
        """
        if state == 'resting':
            self.params.coupling_strength = 1.0  # Weak coupling
        elif state == 'focused':
            self.params.coupling_strength = 10.0  # Strong coupling
        elif state == 'multitasking':
            self.params.coupling_strength = 5.0  # Moderate coupling
        else:
            raise ValueError(f"Unknown state: {state}")


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = KuramotoModel()
    
    # Simulate resting state (no external input)
    model.set_cognitive_state('resting')
    sync_rest, phases_rest = model.simulate(duration=10.0)
    print(f"Resting state - Mean R(t): {np.mean(sync_rest):.3f}")
    
    # Simulate focused state with uniform external input
    model.reset()
    model.set_cognitive_state('focused')
    
    def focused_input(t):
        return np.ones(31) * 5.0  # Uniform 5 Hz stimulus
    
    sync_focused, phases_focused = model.simulate(
        duration=10.0,
        external_input_func=focused_input
    )
    print(f"Focused state - Mean R(t): {np.mean(sync_focused):.3f}")
    
    # Simulate multitasking with competing inputs
    model.reset()
    model.set_cognitive_state('multitasking')
    
    def multitask_input(t):
        input_vec = np.zeros(31)
        input_vec[:15] = 5.0   # First half: +5 Hz
        input_vec[15:] = -5.0  # Second half: -5 Hz
        return input_vec
    
    sync_multi, phases_multi = model.simulate(
        duration=10.0,
        external_input_func=multitask_input
    )
    print(f"Multitasking state - Mean R(t): {np.mean(sync_multi):.3f}")
