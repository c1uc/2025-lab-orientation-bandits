"""
2025 Lab Orientation - Multi-Armed Bandits Assignment

STUDENT NAME: ___________________
DATE: ___________________

ASSIGNMENT OVERVIEW:
===================
In this assignment, you will implement two bandit algorithms to solve the 
exploration-exploitation problem in a 3-armed bandit environment.

LEARNING OBJECTIVES:
===================
1. Understand the multi-armed bandit problem
2. Implement two bandit algorithms:
   - Explore-Then-Commit (ETC)
   - Upper Confidence Bound (UCB)
3. Analyze the performance trade-offs between different exploration strategies

INSTRUCTIONS:
============
1. Complete the algorithm implementations marked with "TODO"
2. The environment, experiment runner, and plotting are provided as reference
3. Test your implementation by running: uv run python main.py
4. Compare your results with the reference solution in solution.py
5. The final plot should be saved as result.png

GRADING CRITERIA:
================
- Correct implementation of ETC algorithm (50%)
- Correct implementation of UCB algorithm (50%)
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from typing import Optional, Tuple, Any


class ThreeArmedBanditEnv(gym.Env):
    """
    A 3-armed bandit environment with Bernoulli rewards.
    
    Each arm has a different probability of success:
    - Arm 0: p = 0.1
    - Arm 1: p = 0.5  
    - Arm 2: p = 0.8 (optimal arm)
    
    This environment is provided as reference - you don't need to modify it.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 3 arms: [0, 1, 2]
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Bernoulli probabilities for each arm
        self.arm_probabilities = np.array([0.1, 0.5, 0.8])
        
        # Reset environment
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Return dummy observation (bandits are non-contextual)
        observation = np.array([0.0], dtype=np.float32)
        info = {}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: The arm to pull (0, 1, or 2)
            
        Returns:
            observation: Dummy observation (non-contextual)
            reward: Bernoulli reward (0 or 1)
            terminated: Always False (infinite horizon)
            truncated: Always False
            info: Empty dict
        """
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")
        
        # Sample Bernoulli reward
        reward = float(np.random.binomial(1, self.arm_probabilities[action]))
        
        # Return dummy observation (non-contextual)
        observation = np.array([0.0], dtype=np.float32)
        terminated = False
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info


class ExploreCommitAlgorithm:
    """
    TODO 1: IMPLEMENT EXPLORE-THEN-COMMIT ALGORITHM
    ===============================================
    
    The Explore-Then-Commit algorithm works in two phases:
    1. EXPLORATION: Pull each arm equally for m rounds total
    2. COMMITMENT: Pull the arm with highest empirical mean for remaining rounds
    
    ALGORITHM DETAILS:
    - For first m rounds: cycle through arms (0, 1, 2, 0, 1, 2, ...)
    - After m rounds: calculate empirical mean for each arm
    - Select arm with highest empirical mean and stick with it
    
    HINTS:
    - Use modulo operator (%) for cycling through arms
    - Track total rewards and counts for each arm
    - Calculate empirical mean = total_reward / count
    - Use np.argmax() to find best arm
    """
    
    def __init__(self, n_arms: int, exploration_rounds: int):
        self.n_arms = n_arms
        self.exploration_rounds = exploration_rounds
        self.reset()
    
    def reset(self):
        """Reset the algorithm state."""
        # TODO 1.1: Initialize algorithm state variables
        # Hint: You need to track time step, arm counts, total rewards, and best arm
        pass
    
    def select_action(self) -> int:
        """Select an action based on the Explore-Then-Commit strategy."""
        # TODO 1.2: Implement the algorithm logic
        # Hint: 
        # - Increment time step
        # - If in exploration phase: cycle through arms using modulo
        # - If in commitment phase: calculate empirical means and select best arm
        pass
    
    def update(self, action: int, reward: float):
        """Update the algorithm with the observed reward."""
        # TODO 1.3: Update arm statistics
        # Hint: Update arm counts and total rewards for the selected arm
        pass


class UCBAlgorithm:
    """
    TODO 2: IMPLEMENT UCB (UPPER CONFIDENCE BOUND) ALGORITHM
    ========================================================
    
    UCB balances exploration and exploitation by selecting the arm with 
    highest upper confidence bound:
    
    UCB_i(t) = empirical_mean_i + sqrt(2 * log(t) / n_i)
    
    WHERE:
    - empirical_mean_i = total_reward_i / count_i
    - t = current time step
    - n_i = number of times arm i was pulled
    
    ALGORITHM:
    1. Pull each arm once initially
    2. For subsequent rounds, select arm with highest UCB value
    
    HINTS:
    - Use np.sqrt() and np.log() for the confidence bound calculation
    - Handle division by zero by pulling each arm once first
    - Use np.argmax() to select arm with highest UCB
    """
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.reset()
    
    def reset(self):
        """Reset the algorithm state."""
        # TODO 2.1: Initialize algorithm state
        # Hint: You need to track time step, arm counts, and total rewards
        pass
    
    def select_action(self) -> int:
        """Select an action based on the UCB strategy."""
        # TODO 2.2: Implement UCB algorithm
        # Hint:
        # - Increment time step
        # - Pull each arm once first (initialization)
        # - Calculate UCB values for all arms
        # - Return arm with highest UCB value
        pass
    
    def update(self, action: int, reward: float):
        """Update the algorithm with the observed reward."""
        # TODO 2.3: Update arm statistics
        # Hint: Update arm counts and total rewards for the selected arm
        pass


def run_experiment(algorithm, env, n_steps: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Run a single experiment with the given algorithm and environment.
    Track cumulative rewards over time.
    
    This function is provided as reference - you don't need to modify it.
    
    Args:
        algorithm: The bandit algorithm to use
        env: The bandit environment
        n_steps: Number of steps to run
        seed: Random seed for reproducibility
        
    Returns:
        Array of cumulative rewards over time
    """
    if seed is not None:
        np.random.seed(seed)
    
    env.reset(seed=seed)
    algorithm.reset()
    
    rewards = []
    cumulative_reward = 0
    
    for step in range(n_steps):
        # Select action
        action = algorithm.select_action()
        
        # Take step in environment
        _, reward, _, _, _ = env.step(action)
        
        # Update algorithm
        algorithm.update(action, reward)
        
        # Track cumulative reward
        cumulative_reward += reward
        rewards.append(cumulative_reward)
    
    return np.array(rewards)


def calculate_optimal_m(arm_probabilities, n_steps):
    """
    Calculate theoretical optimal exploration rounds for ETC.
    
    For a bandit with K arms, the optimal m is approximately:
    m* ≈ (4 * log(n) / Δ²)^(2/3) * n^(1/3)
    
    where Δ is the gap between best and second-best arm.
    
    This function is provided as reference - you don't need to modify it.
    """
    sorted_probs = np.sort(arm_probabilities)[::-1]  # Sort in descending order
    delta = sorted_probs[0] - sorted_probs[1]  # Gap between best and second-best
    
    # Theoretical optimal formula (approximation)
    optimal_m = int((4 * np.log(n_steps) / (delta**2))**(2/3) * n_steps**(1/3))
    
    return optimal_m, delta


def plot_results_three_algorithms(etc_small_rewards, etc_large_rewards, ucb_rewards, 
                                 n_steps: int, m_small: int, m_large: int, m_optimal: int):
    """
    Plot the cumulative reward curves for three algorithms.
    
    This function is provided as reference - you don't need to modify it.
    It creates a two-panel plot showing:
    1. Cumulative rewards over time
    2. Cumulative regret over time
    """
    plt.figure(figsize=(15, 10))
    
    time_steps = np.arange(1, n_steps + 1)
    
    # Plot cumulative rewards
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, etc_small_rewards, label=f'ETC (m={m_small}, small)', 
             color='blue', linewidth=2, linestyle='--')
    plt.plot(time_steps, etc_large_rewards, label=f'ETC (m={m_large}, large)', 
             color='green', linewidth=2, linestyle='-.')
    plt.plot(time_steps, ucb_rewards, label='UCB', color='red', linewidth=2)
    
    # Add vertical line for optimal m
    plt.axvline(x=m_optimal, color='gray', linestyle=':', alpha=0.7, 
                label=f'Theoretical optimal m={m_optimal}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Comparison: ETC (Different m) vs UCB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot regret (difference from optimal)
    plt.subplot(2, 1, 2)
    optimal_reward = 0.8 * time_steps  # Best arm has probability 0.8
    etc_small_regret = optimal_reward - etc_small_rewards
    etc_large_regret = optimal_reward - etc_large_rewards
    ucb_regret = optimal_reward - ucb_rewards
    
    plt.plot(time_steps, etc_small_regret, label=f'ETC (m={m_small}, small) Regret', 
             color='blue', linewidth=2, linestyle='--')
    plt.plot(time_steps, etc_large_regret, label=f'ETC (m={m_large}, large) Regret', 
             color='green', linewidth=2, linestyle='-.')
    plt.plot(time_steps, ucb_regret, label='UCB Regret', color='red', linewidth=2)
    
    plt.axvline(x=m_optimal, color='gray', linestyle=':', alpha=0.7, 
                label=f'Theoretical optimal m={m_optimal}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'result.png'")
    plt.show()


def main():
    """
    Main function to run the bandit experiment with different ETC parameters.
    
    This function is provided as reference - you don't need to modify it.
    It orchestrates the complete experiment comparing different algorithms.
    """
    print("Running 3-Armed Bandit Experiment: ETC vs UCB Comparison")
    print("=" * 60)
    
    # Environment setup
    env = ThreeArmedBanditEnv()
    print(f"Arm probabilities: {env.arm_probabilities}")
    print(f"Optimal arm: {np.argmax(env.arm_probabilities)} (p={np.max(env.arm_probabilities)})")
    
    # Experiment parameters
    n_steps = 1000
    n_runs = 10  # Average over multiple runs
    
    # Calculate theoretical optimal m and define test values
    m_optimal, delta = calculate_optimal_m(env.arm_probabilities, n_steps)
    m_small = max(30, int(m_optimal * 0.3))  # Smaller than optimal
    m_large = min(300, int(m_optimal * 2.0))  # Larger than optimal
    
    print(f"Gap between best and second-best arm (Δ): {delta:.2f}")
    print(f"Theoretical optimal m: {m_optimal}")
    print(f"Testing with:")
    print(f"  - Small m: {m_small} (suboptimal - too little exploration)")
    print(f"  - Large m: {m_large} (suboptimal - too much exploration)")
    print(f"  - UCB: adaptive exploration")
    print()
    
    # Initialize algorithms
    etc_small_algorithm = ExploreCommitAlgorithm(n_arms=3, exploration_rounds=m_small)
    etc_large_algorithm = ExploreCommitAlgorithm(n_arms=3, exploration_rounds=m_large)
    ucb_algorithm = UCBAlgorithm(n_arms=3)
    
    print(f"Running {n_runs} experiments with {n_steps} steps each...")
    
    # Run multiple experiments and average results
    etc_small_all_rewards = []
    etc_large_all_rewards = []
    ucb_all_rewards = []
    
    try:
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            
            # Run ETC with small m
            etc_small_rewards = run_experiment(etc_small_algorithm, env, n_steps, seed=run)
            etc_small_all_rewards.append(etc_small_rewards)
            
            # Run ETC with large m
            etc_large_rewards = run_experiment(etc_large_algorithm, env, n_steps, seed=run)
            etc_large_all_rewards.append(etc_large_rewards)
            
            # Run UCB
            ucb_rewards = run_experiment(ucb_algorithm, env, n_steps, seed=run)
            ucb_all_rewards.append(ucb_rewards)
        
        # Average results
        etc_small_mean_rewards = np.mean(etc_small_all_rewards, axis=0)
        etc_large_mean_rewards = np.mean(etc_large_all_rewards, axis=0)
        ucb_mean_rewards = np.mean(ucb_all_rewards, axis=0)
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL RESULTS COMPARISON:")
        print("="*60)
        
        print(f"ETC with SMALL m={m_small} (under-exploration):")
        print(f"  Final cumulative reward: {etc_small_mean_rewards[-1]:.2f}")
        print(f"  Average reward per step: {etc_small_mean_rewards[-1]/n_steps:.4f}")
        print(f"  Final regret: {(0.8 * n_steps - etc_small_mean_rewards[-1]):.2f}")
        
        print(f"\nETC with LARGE m={m_large} (over-exploration):")
        print(f"  Final cumulative reward: {etc_large_mean_rewards[-1]:.2f}")
        print(f"  Average reward per step: {etc_large_mean_rewards[-1]/n_steps:.4f}")
        print(f"  Final regret: {(0.8 * n_steps - etc_large_mean_rewards[-1]):.2f}")
        
        print(f"\nUCB (adaptive exploration):")
        print(f"  Final cumulative reward: {ucb_mean_rewards[-1]:.2f}")
        print(f"  Average reward per step: {ucb_mean_rewards[-1]/n_steps:.4f}")
        print(f"  Final regret: {(0.8 * n_steps - ucb_mean_rewards[-1]):.2f}")
        
        print(f"\nOptimal performance:")
        print(f"  Theoretical optimal reward per step: {np.max(env.arm_probabilities):.4f}")
        print(f"  Theoretical optimal cumulative reward: {0.8 * n_steps:.2f}")
        
        # Determine best performing algorithm
        final_rewards = [
            etc_small_mean_rewards[-1],
            etc_large_mean_rewards[-1], 
            ucb_mean_rewards[-1]
        ]
        best_idx = np.argmax(final_rewards)
        algorithms = [f"ETC (m={m_small})", f"ETC (m={m_large})", "UCB"]
        
        print(f"\nBest performing algorithm: {algorithms[best_idx]} with {final_rewards[best_idx]:.2f} cumulative reward")
        
        # Plot results
        plot_results_three_algorithms(etc_small_mean_rewards, etc_large_mean_rewards, 
                                     ucb_mean_rewards, n_steps, m_small, m_large, m_optimal)
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        print("\nThis likely means your algorithm implementations are incomplete.")
        print("Please complete the TODO sections in the algorithm classes.")
        print("Check solution.py for the complete reference implementation.")


if __name__ == "__main__":
    main()


"""
IMPLEMENTATION GUIDE:
====================

TODO 1: Explore-Then-Commit Algorithm
-------------------------------------
1.1 reset(): Initialize self.t=0, self.arm_counts=np.zeros(n_arms), 
             self.arm_rewards=np.zeros(n_arms), self.best_arm=None

1.2 select_action(): 
    - Increment self.t
    - If t <= exploration_rounds: return (t-1) % n_arms
    - Else: if best_arm is None, calculate empirical means and set best_arm
            return best_arm

1.3 update(): 
    - self.arm_counts[action] += 1
    - self.arm_rewards[action] += reward

TODO 2: UCB Algorithm  
--------------------
2.1 reset(): Initialize self.t=0, self.arm_counts=np.zeros(n_arms), 
             self.arm_rewards=np.zeros(n_arms)

2.2 select_action():
    - Increment self.t
    - For each arm with count=0, return that arm
    - Calculate UCB = mean + sqrt(2*log(t)/count) for each arm
    - Return arm with highest UCB

2.3 update():
    - self.arm_counts[action] += 1  
    - self.arm_rewards[action] += reward

DEBUGGING TIPS:
==============
- Test each algorithm separately first
- Use print statements to debug your logic
- Start with small n_steps for faster testing
- Compare intermediate results with solution.py

EXPECTED RESULTS:
================
- UCB should perform best overall
- ETC with small m should beat ETC with large m
- All algorithms should improve over random (≈0.467 reward per step)
"""