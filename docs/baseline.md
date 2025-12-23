## Baseline Reinforcement Learning Methods

We evaluate our hierarchical reasoning framework against a set of widely adopted reinforcement learning baselines. These methods represent standard flat and memory-augmented control architectures and are included to assess whether improvements stem from hierarchical abstraction rather than optimization strength or temporal modeling capacity.

### 1. Method Definitions and Mathematical Formulation

#### (1) Basic Reinforcement Learning Baseline (Vanilla Policy Gradient)

The basic baseline is a stochastic policy optimized via Monte Carlo policy gradients:

\[
\pi_\theta(a_t \mid s_t)
\]

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t \right]
\]

where \( G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} \) is the return.

---

#### (2) Actor–Critic (AC)

Actor–Critic augments the policy with a learned state-value function:

\[
\pi_\theta(a_t \mid s_t), \quad V_\phi(s_t)
\]

Policy gradient update:

\[
\nabla_\theta J = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\left( r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right) \right]
\]

---

#### (3) Advantage Actor–Critic (A2C)

A2C replaces the one-step TD error with a multi-step advantage estimate:

\[
A_t = \sum_{k=0}^{K-1} \gamma^k r_{t+k} + \gamma^K V(s_{t+K}) - V(s_t)
\]

\[
\nabla_\theta J = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) A_t \right]
\]

---

#### (4) Proximal Policy Optimization (PPO)

PPO constrains policy updates via a clipped surrogate objective:

\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\]

\[
L^{\text{PPO}} =
\mathbb{E}\left[
\min \left(
r_t(\theta) A_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
\]

---

#### (5) Recurrent Actor–Critic (LSTM)

Temporal dependencies are captured through a recurrent hidden state:

\[
h_t = f_{\text{LSTM}}(h_{t-1}, s_t)
\]

\[
\pi(a_t \mid h_t), \quad V(h_t)
\]

---

#### (6) Transformer-Based Policy

The policy attends over the full history of observations:

\[
H_t = \text{Transformer}(s_0, s_1, \dots, s_t)
\]

\[
\pi(a_t \mid H_t)
\]

---

### 2. Baseline Checklist and Capability Comparison

| Method | Implemented |
|-------|-------------|
| Basic Policy Gradient | ✓ |
| Actor–Critic (AC) | ✗ |
| Advantage Actor–Critic (A2C) | ✗ |
| Proximal Policy Optimization (PPO) | ✗ |
| LSTM Actor–Critic | ✗ |
| Transformer-Based Policy | ✗ |