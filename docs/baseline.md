## Baseline Reinforcement Learning Methods

We evaluate our hierarchical reasoning framework against a set of widely adopted reinforcement learning baselines. These methods represent standard flat, value-based, and memory-augmented control architectures and are included to assess whether improvements stem from hierarchical abstraction rather than optimization strength, exploration strategy, or temporal modeling capacity.

### 1. Method Definitions and Mathematical Formulation

#### (1) Basic Reinforcement Learning Baseline (Vanilla Policy Gradient)

The basic baseline is a stochastic policy optimized via Monte Carlo policy gradients:

$$\pi_\theta(a_t \mid s_t)$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]$$

where $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ denotes the return.

---

#### (2) Actor–Critic (AC)

Actor–Critic augments the policy with a learned state-value function:

$$\pi_\theta(a_t \mid s_t), \quad V_\phi(s_t)$$

The policy is updated using the one-step temporal-difference error:

$$\nabla_\theta J = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right) \right]$$

---

#### (3) Proximal Policy Optimization (PPO)

PPO constrains policy updates via a clipped surrogate objective:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

$$L^{\text{PPO}} = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

---

#### (4) Recurrent Actor–Critic (LSTM)

Temporal dependencies are captured through a recurrent hidden state:

$$h_t = f_{\text{LSTM}}(h_{t-1}, s_t)$$

$$\pi(a_t \mid h_t), \quad V(h_t)$$

---

#### (5) Deep Q-Network (DQN)

DQN learns an action-value function using temporal-difference learning with a target network:

$$Q_\theta(s_t, a_t)$$

$$L_{\text{DQN}}(\theta) = \mathbb{E} \left[ \left( r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a') - Q_\theta(s_t, a_t) \right)^2 \right]$$

where $\theta^-$ denotes the parameters of the target network.

---

#### (6) Soft Actor–Critic (SAC)

Soft Actor–Critic optimizes a maximum-entropy reinforcement learning objective:

$$J(\pi) = \mathbb{E} \left[ \sum_t r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right]$$

The soft Q-function target is defined as:

$$y_t = r_t + \gamma \mathbb{E}_{a' \sim \pi} \left[ Q_{\phi^-}(s_{t+1}, a') - \alpha \log \pi(a' \mid s_{t+1}) \right]$$

This entropy regularization encourages exploration and improves robustness under sparse rewards.

---

### 2. Baseline Checklist and Capability Comparison

| Method                             | Implemented |
|------------------------------------|-------------|
| Basic Policy Gradient              | ✓           |
| Actor–Critic (AC)                  | ✓           |
| Proximal Policy Optimization (PPO) | ✓           |
| LSTM Actor–Critic                  | ✗           |
| Deep Q-Network (DQN)               | ✗           |
| Soft Actor–Critic (SAC)            | ✗           |