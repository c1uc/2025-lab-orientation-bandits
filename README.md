# 2025 Lab Orientation - Multi-Armed Bandits Assignment

## 📋 Assignment Overview

This assignment introduces you to the **multi-armed bandit problem** - a fundamental challenge in reinforcement learning that balances exploration (trying new actions) with exploitation (using known good actions).

You will implement:
1. Two bandit algorithms: Explore-Then-Commit (ETC) and Upper Confidence Bound (UCB)
2. The 3-armed bandit environment, experiment runner, and plotting are provided as reference

## 🎯 Learning Objectives

- Understand the exploration-exploitation tradeoff
- Implement bandit algorithms with different exploration strategies
- Compare algorithm performance through experimentation
- Analyze the theoretical vs practical trade-offs

## 📁 File Structure

```
2025-Lab-Orientation-Bandits/
├── main.py          # Student template (YOUR WORK HERE)
├── result.png       # Generated comparison plot
├── pyproject.toml   # Project dependencies
└── README.md        # This file
```

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Install dependencies using uv
uv sync

# Test that everything works
uv run python --version
```

### 2. Implementation Steps

Complete the TODOs in `main.py` in this order:

#### **TODO 1: Explore-Then-Commit Algorithm** 🔍
- Implement exploration phase (cycle through arms)
- Implement commitment phase (choose best empirical arm)
- Key insight: Fixed exploration budget

#### **TODO 2: UCB Algorithm** 📊
- Implement Upper Confidence Bound selection
- Formula: `UCB_i(t) = mean_i + sqrt(2 * log(t) / n_i)`
- Key insight: Adaptive exploration based on uncertainty

**Note**: The bandit environment, experiment runner, and plotting functions are provided as reference code to help you understand the complete system.

### 3. Testing Your Implementation

```bash
# Run your implementation
uv run python main.py
```

## 🔍 Key Concepts

### Multi-Armed Bandit Problem
- **Setting**: Multiple slot machines (arms) with unknown reward probabilities
- **Goal**: Maximize cumulative reward over time
- **Challenge**: Balance exploration (learning) vs exploitation (earning)

### Explore-Then-Commit (ETC)
```
Phase 1 (Exploration): Pull each arm equally for m rounds
Phase 2 (Commitment): Pull best arm for remaining rounds
```
- **Parameter**: m (exploration rounds)
- **Trade-off**: Small m → insufficient exploration, Large m → wasted exploitation

### Upper Confidence Bound (UCB)
```
Select arm with highest: mean_reward + confidence_bound
```
- **Adaptive**: Automatically balances exploration/exploitation
- **Optimistic**: Chooses arms with high uncertainty
- **No parameters**: Self-tuning algorithm

## 📊 Expected Results

When correctly implemented, you should observe:

1. **UCB performs best overall** - adaptive exploration advantage
2. **ETC with small m > ETC with large m** - for finite horizons
3. **Clear visualization** of exploration-exploitation tradeoffs

### Sample Output
```
============================================================
FINAL RESULTS COMPARISON:
============================================================
ETC with SMALL m=136 (under-exploration):
  Final cumulative reward: 755.10
  Final regret: 44.90

ETC with LARGE m=300 (over-exploration):
  Final cumulative reward: 701.30
  Final regret: 98.70

UCB (adaptive exploration):
  Final cumulative reward: 765.30
  Final regret: 34.70

Best performing algorithm: UCB with 765.30 cumulative reward
```

## 🐛 Debugging Tips

1. **Start incrementally**: Implement and test each TODO section separately
2. **Use small experiments**: Set `n_steps=100` during development
3. **Print intermediate values**: Debug your algorithm logic
4. **Validate environment**: Test bandit environment first

### Common Issues
- **Division by zero**: Handle arm counts = 0 in UCB
- **Wrong action selection**: Check modulo operator in ETC
- **Incorrect updates**: Ensure arm statistics are updated properly

## 📈 Analysis Questions

After completing the implementation, consider:

1. **Why does UCB outperform ETC?**
2. **What happens if you change the arm probabilities?**
3. **How does the optimal m for ETC change with problem parameters?**
4. **What are the theoretical regret bounds for each algorithm?**

## 📝 Submission

Submit the following files:
1. `main.py` - Your completed implementation
2. `result.png` - Generated comparison plot

---

**Good luck! 🍀 Remember: the goal is to understand the exploration-exploitation tradeoff, not just to get the code working.**
