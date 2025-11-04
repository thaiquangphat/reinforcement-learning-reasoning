# Hierarchical Reinforcement Learning for Multi-Hop Question Answering on Knowledge Graphs

## 1. Context and Objectives

The problem of **multi-hop question answering (MHQA)** on a knowledge graph $G = (V, E)$ requires an agent to perform a sequence of discrete reasoning steps (multi-hop) across entities to find the final answer.

This is modeled as a sequential decision process (MDP):

$$(S, A, P, R, \gamma)$$

where:

- $S$: state space (includes query and current position in the graph)
- $A$: action space (selecting the next edge or node)
- $P(s' | s, a)$: environment dynamics (transition to new node)
- $R(s, a)$: reward
- $\gamma$: discount factor

**Main Challenge**: The action space is enormous, dependent on graph structure, and reward signals are very sparse (only appearing when reaching the goal).

Therefore, the model uses **Hierarchical Reinforcement Learning (HRL)** – a more structured approach:

Decomposing the problem into a high-level (manager) that selects latent subgoals, and a low-level (worker) that learns policies to achieve subgoals.



## 2. State Representation

At time step $t$, the agent is at node $v_t \in V$. We encode this node into a vector $h_t = f_{\text{enc}}(v_t)$ via **Relational Graph Attention Network (RGAT)**:

$$h_t = \text{RGAT}(v_t) \in \mathbb{R}^d$$

In parallel, the agent maintains:

- **query hidden** $q_t$ (encodes the question, updated via LSTM)
- **path hidden** $p_t$ (encodes the trajectory, updated via GRU)

The complete state:

$$s_t = [q_t, p_t, h_t, v_t]$$

**Meaning:**

- $q_t$ helps maintain semantic context of the question
- $p_t$ stores traces of reasoning steps already taken
- $h_t$ encodes information about the current node in latent space

This combination allows the agent to capture the **reasoning state** rather than just position.



## 3. High-Level (Manager) – Selecting Latent Subgoals

The Manager operates every $K$ steps (to create temporal abstraction):

$$\text{If } t \bmod K = 0 \Rightarrow \text{Manager selects subgoal}$$

### 3.1. Input and Policy Function

The Manager receives:

$$x_t^m = [q_t, h_t, p_t, v_{\text{target}}]$$

and generates a distribution over a set of prototype embeddings $P = [p_1, \ldots, p_L]^\top \in \mathbb{R}^{L \times d}$:

$$\pi_m(i | s_t) = \text{softmax}(W_m f_m(x_t^m))_i$$

Select prototype index:

$$g_t \sim \pi_m(\cdot | s_t)$$

and the corresponding latent subgoal is:

$$z_t = p_{g_t}$$

**Meaning:**

- $p_i$ is a "latent behavior prototype" — representing a reasoning direction or sub-goal action
- The Manager acts as a coordinator: decomposing the reasoning problem into abstract subgoals

**Academic Sources:**

- [**FeUdal Networks**](https://arxiv.org/abs/1703.01161) (Vezhnevets et al., ICML 2017): hierarchical agent with manager selecting latent subgoal vectors
- [**HIRO**](https://arxiv.org/abs/1805.08296) (Nachum et al., NeurIPS 2018): uses goal embeddings and subgoal transitions in latent space



## 4. Low-Level (Worker) – Navigating the Graph to Achieve Subgoals

The Worker operates at each individual step.

**Input:**

$$x_t^w = [q_t, p_t, h_t, z_t]$$

### 4.1. Action Selection Policy

For the neighborhood $\mathcal{N}(v_t)$, each candidate $v_j$ has embedding $h_j$. The Worker computes scores:

$$s_{t,j} = W_2 \sigma(W_1[\phi(x_t^w), h_j, \phi(x_t^w) \odot h_j])$$

then:

$$\pi_w(a_t = j | s_t, z_t) = \frac{e^{s_{t,j}}}{\sum_{k \in \mathcal{N}(v_t)} e^{s_{t,k}}}$$

Select action:

$$a_t \sim \pi_w(\cdot | s_t, z_t)$$

then update node:

$$v_{t+1} = \text{next}(v_t, a_t)$$

### 4.2. Value Function

$$V_w(s_t, z_t) = W_v^w f_v^w(x_t^w)$$

**Meaning:**

The Worker acts as detailed controller, "interpolating" trajectories between two latent subgoals.

This helps separate decision frequency and abstraction level, making learning more efficient (following Option-Critic and FeUdal principles).



## 5. Context Update Mechanism

After each step:

$$\begin{aligned}
h_{t+1} &= f_{\text{enc}}(v_{t+1}) \\
p_{t+1} &= \text{GRU}_p(p_t, h_{t+1}) \\
q_{t+1} &= \text{LSTM}_q(q_t, h_{t+1})
\end{aligned}$$

**Meaning:**

Allows storing reasoning information over time (multi-hop).

GRU helps remember the sequence of nodes visited; LSTM helps "anchor" reasoning context with the query.



## 6. Reward Function

Total reward:

$$r_t = r_t^{\text{ext}} + r_t^{\text{int}}$$

### 6.1. Extrinsic Reward (Goal-Oriented)

$$r_t^{\text{ext}} = \alpha_{\text{sim}}[\cos(v_{t+1}, v_{\text{target}}) - \cos(v_t, v_{\text{target}})] - \beta_{\text{len}} - \gamma_{\text{cycle}} \mathbb{I}[v_{t+1} \in \text{Visited}]$$

If goal is reached:

$$r_t^{\text{succ}} = R_{\text{success}} \mathbb{I}[v_{t+1} = v_{\text{target}}]$$

**Academic Sources:**

- Reward shaping via cosine distance is common in KG-RL ([**MINERVA**](https://arxiv.org/abs/1711.05851), Das et al., ICLR 2018)
- Cycle and length penalties ensure reasonable trajectories, avoiding loops

### 6.2. Intrinsic Reward (Toward Prototype)

$$d_t^{\text{prev}} = 1 - \cos(v_t, z_t), \quad d_t^{\text{next}} = 1 - \cos(v_{t+1}, z_t)$$

Intrinsic reward:

$$r_t^{\text{int}} = \alpha_{\text{int}}(d_t^{\text{prev}} - d_t^{\text{next}}) - \beta_{\text{len}} + R_{\text{subgoal}} \mathbb{I}[d_t^{\text{next}} \leq \epsilon]$$

**Meaning:**

The Worker is encouraged to move so that node embedding approaches the subgoal embedding $z_t$.

This helps reduce reward signal delay (credit assignment problem).

**Academic Sources:**

- Intrinsic motivation via latent distance (HIRO, DIAYN, SkewFit)
- FeUdal reward = direction of latent goal – this is the prototype of this formula



## 7. Value Estimation and GAE

Uses **Generalized Advantage Estimation** ([**GAE**](https://arxiv.org/abs/1506.02438), Schulman et al., ICLR 2016) for both manager and worker:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$$$A_t = \delta_t + (\gamma\lambda) A_{t+1}$$$$R_t = A_t + V(s_t)$$

**Meaning:**

Smooths the advantage sequence, reduces variance, stabilizes actor-critic updates.



## 8. Loss Functions and Optimization (PPO-style)

Combined loss for each level:

### Worker:

$$\mathcal{L}_w = -\mathbb{E}_t[\log \pi_w(a_t | s_t, z_t) A_t^w] + \beta_v(V_w(s_t, z_t) - R_t)^2 - \beta_H H[\pi_w]$$

### Manager:

$$\mathcal{L}_m = -\mathbb{E}_t[\log \pi_m(g_t | s_t) A_t^m] + \beta_v(V_m(s_t) - R_t)^2 - \beta_H H[\pi_m]$$

With [**PPO**](https://arxiv.org/abs/1707.06347) clipping (Schulman et al., 2017):

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t[\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]$$

where:

$$\rho_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$



## 9. Hindsight Experience Relabeling (HER)

With probability $p_{\text{her}}$, select a node $v_k$ in the trajectory, set as new target:

$$v_{\text{target}}' = v_k$$

and recalculate reward according to formula $r_t'(v_{\text{target}}')$.

**Meaning:**

Converts failed episodes into hypothetical successful episodes, improving sample learning.

This is a variant of [**HER**](https://arxiv.org/abs/1707.01495) (Andrychowicz et al., NeurIPS 2017).



## 10. Overall Learning Process

1. **Sampling**: Run episodes to collect trajectory $(s_t, a_t, r_t, \pi_t, V_t)$

2. **Relabeling**: Apply HER to some trajectories

3. **GAE**: Compute $A_t$ and $R_t$ for each level

4. **Gradient Update**: Optimize $\mathcal{L}_m$ and $\mathcal{L}_w$ via Adam, different learning rates

5. **Backpropagation**: Gradients propagate through prototype embeddings $P$  helping learn useful subgoal space



## 11. Complete Model Summary via Equations

$$
\begin{aligned}
(1)\;& \text{Subgoal selection: } && g_t \sim \pi_m(\cdot \mid s_t), \quad z_t = p_{g_t} \\
(2)\;& \text{Action selection: } && a_t \sim \pi_w(\cdot \mid s_t, z_t) \\
(3)\;& \text{Transition: } && s_{t+1} = f_{\text{trans}}(s_t, a_t) \\
(4)\;& \text{Rewards: } && r_t = f_{\text{ext}}(s_t, a_t) + f_{\text{int}}(s_t, z_t) \\
(5)\;& \text{Advantage: } && A_t = \text{GAE}(r_t, V(s_t)) \\
(6)\;& \text{Loss: } && \mathcal{L} = \mathcal{L}_m + \mathcal{L}_w \\
(7)\;& \text{Gradient update: } && \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\end{aligned}
$$



## 12. Overall Significance

| Component | Goal / Meaning | Theoretical Basis |
|--|-|-|
| RGAT Encoder | Extract knowledge context embedding from graph | [GAT](https://arxiv.org/abs/1710.10903) (Veličković et al., ICLR 2018) |
| Manager (prototype policy) | Separate abstraction, define latent goals | [FeUdal Networks](https://arxiv.org/abs/1703.01161) (2017) |
| Worker (goal-conditioned) | Implement micro-behavior to reach subgoal | [HIRO](https://arxiv.org/abs/1805.08296) (2018) |
| Intrinsic reward (cosine distance) | Help stable learning, reduce credit delay | Goal-conditioned RL |
| HER relabeling | Increase sample efficiency, combat reward sparsity | [HER](https://arxiv.org/abs/1707.01495) (2017) |
| GAE + PPO update | Stabilize gradients, reduce variance | [GAE](https://arxiv.org/abs/1506.02438) (Schulman et al., 2016), [PPO](https://arxiv.org/abs/1707.06347) (2017) |
| Reward shaping via similarity | Improve guidance toward answer | [MINERVA](https://arxiv.org/abs/1711.05851) (ICLR 2018) |



## 13. Architecture Philosophy Summary

**Why this design:**

1. **Multi-hop reasoning** is equivalent to temporal abstraction  needs HRL

2. **Reasoning in KG** has discrete space but continuous latent semantics  use prototype embeddings to "bridge" between discrete and continuous space

3. **Reward is very sparse**  need intrinsic motivation and HER to stabilize learning

4. **PPO + GAE**  ensure stable gradients in discrete environment

**In summary:**

$$\text{Model} = \text{HRL} + \text{Goal-conditioned RL} + \text{Prototype Subgoals} + \text{Reward Shaping} + \text{HER} + \text{PPO/GAE}$$



## Key References

1. **FeUdal Networks for Hierarchical Reinforcement Learning**  
   Vezhnevets et al., ICML 2017  [arXiv:1703.01161](https://arxiv.org/abs/1703.01161) | [PMLR](https://proceedings.mlr.press/v70/vezhnevets17a.html)

2. **Data-Efficient Hierarchical Reinforcement Learning (HIRO)**  
   Nachum et al., NeurIPS 2018  [arXiv:1805.08296](https://arxiv.org/abs/1805.08296) | [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)

3. **Go for a Walk and Arrive at the Answer (MINERVA)**  
   Das et al., ICLR 2018  [arXiv:1711.05851](https://arxiv.org/abs/1711.05851) | [OpenReview](https://openreview.net/forum?id=Syg-YfWCW)

4. **Graph Attention Networks (GAT)**  
   Veličković et al., ICLR 2018  [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) | [OpenReview](https://openreview.net/forum?id=rJXMpikCZ)

5. **Proximal Policy Optimization Algorithms (PPO)**  
   Schulman et al., 2017  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

6. **High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)**  
   Schulman et al., ICLR 2016  [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

7. **Hindsight Experience Replay (HER)**  
   Andrychowicz et al., NeurIPS 2017  [arXiv:1707.01495](https://arxiv.org/abs/1707.01495) | [NeurIPS](https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html)