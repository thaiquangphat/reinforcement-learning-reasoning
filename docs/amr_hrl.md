# Complete Step-by-Step Mathematical Transformations
# AMR-Guided Hierarchical Reinforcement Learning for Query Path Reasoning

## 1. Notation and Initialization

### 1.1 Basic Sets
```
V = {v_1, v_2, ..., v_n}           # Graph vertices
E ⊆ V × V                         # Edges
q ∈ Q                             # Natural language question
v_0 ∈ V                           # Start entity
v^* ∈ V                           # Target entity
T ∈ ℕ                            # Maximum path length
d ∈ ℕ                            # Embedding dimension
```

### 1.2 Embedding Initialization
For each $v \in V$:
$$h_v^{(0)} = E_v \in \mathbb{R}^d \quad \text{(Pre-trained node embeddings)}$$

$$q_0 = \text{BERT}(q) \in \mathbb{R}^d \quad \text{(Question embedding)}$$

$$P = \{p_1, \ldots, p_L\} \subset \mathbb{R}^d \quad \text{(Prototype embeddings - learned)}$$

$$C = \{c_1, \ldots, c_M\} \subset \mathbb{R}^d \quad \text{(AMR concept embeddings)}$$

---

## 2. State Representation at Step $t$

### 2.1 Complete State Vector
State at time $t$:
$$s_t = (q_t, h_t, p_t, v_{\text{target}}, g_m, \text{visited}_t)$$

Where:
- $q_t \in \mathbb{R}^d$ — Current query state (LSTM output)
- $h_t \in \mathbb{R}^d$ — Current node embedding
- $p_t \in \mathbb{R}^d$ — Path encoding (GRU output)
- $v_{\text{target}} \in \mathbb{R}^d$ — Target node embedding
- $g_m \in \mathbb{R}^d$ — Current subgoal embedding
- $\text{visited}_t \subset V$ — Set of visited nodes

### 2.2 Concatenation Operations
Worker state vector:
$$w_t = \text{concat}(q_t, h_t, p_t, g_m) \in \mathbb{R}^{4d}$$

Manager state vector (when $t \bmod K = 0$):
$$s_t^m = \text{concat}(q_t, h_t, p_t, v_{\text{target}}) \in \mathbb{R}^{4d}$$

---

## 3. Manager Step ($t \bmod K = 0$)

### 3.1 Prototype Selection Distribution

**Step 1: Compute logits for each prototype**

For $i = 1$ to $L$:
$$z_i = W_m s_t^m \cdot p_i + b_i$$

where $W_m \in \mathbb{R}^{d \times 4d}$, $b_i \in \mathbb{R}$

**Step 2: Softmax transformation**
$$\pi_m(i|s_t^m) = \frac{\exp(z_i)}{\sum_{j=1}^L \exp(z_j)}$$

**Step 3: Sample subgoal index**
$$g_{\text{idx}} \sim \text{Categorical}(\pi_m(\cdot|s_t^m))$$

**Step 4: Subgoal embedding**
$$g_m = p_{g_{\text{idx}}}$$

### 3.2 AMR-Override Mechanism

If AMR available:

**Step 1: Compute similarities**

For each $c \in C$:
$$\text{sim}(c) = \frac{h_t \cdot c}{\|h_t\| \|c\|}$$

**Step 2: Select AMR concept**
$$c^* = \arg\max_{c \in C} \text{sim}(c)$$

**Step 3: Navigate AMR graph**

Let $G_A = (C, E_A)$ be AMR graph. For neighbor $c'$ of $c^*$ in $G_A$:
- Find $v \in V$ with $\max \text{sim}(h_v, c')$

**Step 4: Set subgoal**
$$g_m = h_v \quad \text{where } v \text{ is selected node}$$

---

## 4. Worker Action Selection

### 4.1 Neighborhood Definition
Let $\mathcal{N}(v_t) = \{v_j : (v_t, v_j) \in E\}$ be neighbors

For each $v_j \in \mathcal{N}(v_t)$, $h_j \in \mathbb{R}^d$ is its embedding

### 4.2 Action Scoring Function

For each candidate $v_j \in \mathcal{N}(v_t)$:

**Step 1: Form candidate vector**
$$c_j = \text{concat}(w_t, h_j, w_t \odot h_j) \in \mathbb{R}^{6d}$$

where $\odot$ denotes element-wise multiplication

**Step 2: Apply transformation**
$$\ell_j = \sigma(W_2 \text{ReLU}(W_1 c_j + b_1) + b_2)$$

where:
- $W_1 \in \mathbb{R}^{d_h \times 6d}$, $b_1 \in \mathbb{R}^{d_h}$
- $W_2 \in \mathbb{R}^{1 \times d_h}$, $b_2 \in \mathbb{R}$
- $\sigma$ is activation function

**Step 3: Normalize to probabilities**
$$\pi_w(j|s_t, g_m) = \frac{\exp(\ell_j)}{\sum_{k \in \mathcal{N}(v_t)} \exp(\ell_k)}$$

### 4.3 Action Selection

**Sampling (training):**
$$a_t \sim \text{Categorical}(\pi_w(\cdot|s_t, g_m))$$
$$v_{t+1} = v_{a_t}$$

**Deterministic (inference):**
$$a_t = \arg\max_{j \in \mathcal{N}(v_t)} \pi_w(j|s_t, g_m)$$
$$v_{t+1} = v_{a_t}$$

---

## 5. State Transition Dynamics

### 5.1 Node Embedding Update
$$h_{t+1} = E_{v_{t+1}} \quad \text{(Lookup from embedding matrix)}$$

### 5.2 Path Encoder (GRU) Update

**Complete GRU equations:**

Update gate:
$$z_t = \sigma(W_z h_{t+1} + U_z p_t + b_z)$$

Reset gate:
$$r_t = \sigma(W_r h_{t+1} + U_r p_t + b_r)$$

Candidate activation:
$$\tilde{n}_t = \tanh(W_n h_{t+1} + U_n (r_t \odot p_t) + b_n)$$

New path encoding:
$$p_{t+1} = (1 - z_t) \odot p_t + z_t \odot \tilde{n}_t$$

Where $z_t, r_t, \tilde{n}_t \in \mathbb{R}^d$ and $\sigma$ is sigmoid function

### 5.3 Query State (LSTM) Update

**Complete LSTM equations:**

Forget gate:
$$f_t = \sigma(W_f h_{t+1} + U_f q_t + V_f c_t + b_f)$$

Input gate:
$$i_t = \sigma(W_i h_{t+1} + U_i q_t + V_i c_t + b_i)$$

Output gate:
$$o_t = \sigma(W_o h_{t+1} + U_o q_t + V_o c_t + b_o)$$

Candidate cell state:
$$\hat{c}_t = \tanh(W_c h_{t+1} + U_c q_t + b_c)$$

New cell state:
$$c_{t+1} = f_t \odot c_t + i_t \odot \hat{c}_t$$

New query state:
$$q_{t+1} = o_t \odot \tanh(c_{t+1})$$

Where $f_t, i_t, o_t \in \mathbb{R}^d$ are gates and $c_t \in \mathbb{R}^d$ is cell state

### 5.4 Visited Set Update
$$\text{visited}_{t+1} = \text{visited}_t \cup \{v_{t+1}\}$$

---

## 6. Reward Computation

### 6.1 Cosine Similarity Calculation

For vectors $u, v \in \mathbb{R}^d$:
$$\cos(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

where $\|u\| = \sqrt{\sum_{i=1}^d u_i^2}$

### 6.2 Extrinsic Reward

Similarity change:
$$\Delta_{\text{sim}} = \cos(h_{t+1}, v^*) - \cos(h_t, v^*)$$

Cycle penalty:
$$\text{cycle\_penalty} = \begin{cases} 
\gamma_{\text{cycle}} & \text{if } v_{t+1} \in \text{visited}_t \\
0 & \text{otherwise}
\end{cases}$$

Extrinsic reward:
$$r_t^{\text{ext}} = \alpha_{\text{sim}} \Delta_{\text{sim}} - \beta_{\text{len}} - \text{cycle\_penalty}$$

At terminal step $T$:
$$\text{if } v_T = v^* : \quad r_T^{\text{ext}} \gets r_T^{\text{ext}} + R_{\text{success}}$$

### 6.3 Intrinsic Reward

Distance to subgoal:
$$d(v, g_m) = 1 - \cos(h_v, g_m)$$

Progress measure:
$$\Delta_d = d(v_t, g_m) - d(v_{t+1}, g_m)$$

Intrinsic reward:
$$r_t^{\text{int}} = \alpha_{\text{int}} \Delta_d - \beta_{\text{len}}$$

### 6.4 Subgoal Achievement Reward
$$r_t^{\text{sg}} = \begin{cases}
R_{\text{reach\_subgoal}} & \text{if } \cos(h_{t+1}, g_m) > \tau \\
0 & \text{otherwise}
\end{cases}$$

### 6.5 Total Reward
$$r_t = r_t^{\text{ext}} + r_t^{\text{int}} + r_t^{\text{sg}}$$

---

## 7. Value Function Computation

### 7.1 Worker Value Function
$$V_w(s_t) = W_v^w w_t + b_v^w$$

where $W_v^w \in \mathbb{R}^{1 \times 4d}$, $b_v^w \in \mathbb{R}$

### 7.2 Manager Value Function
$$V_m(s_t^m) = W_v^m s_t^m + b_v^m$$

where $W_v^m \in \mathbb{R}^{1 \times 4d}$, $b_v^m \in \mathbb{R}$

---

## 8. Advantage Estimation (GAE)

### 8.1 Temporal Difference Error

**For worker:**
$$\delta_t^w = r_t + \gamma_w V_w(s_{t+1}) - V_w(s_t)$$

**For manager (at manager steps $t = mK$):**
$$\delta_m^m = \sum_{i=0}^{K-1} \gamma_w^i r_{mK+i} + \gamma_w^K V_m(s_{(m+1)K}) - V_m(s_{mK})$$

### 8.2 GAE Computation

General formula for sequence $\delta_0, \delta_1, \ldots, \delta_{T-1}$:
$$A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

**Worker advantage:**
$$A_t^w = \sum_{l=0}^{T-t-1} (\gamma_w \lambda_w)^l \delta_{t+l}^w$$

**Manager advantage:**
$$A_m^m = \sum_{k=0}^{M-m-1} (\gamma_m \lambda_m)^k \delta_{m+k}^m$$

where $M = \lfloor T/K \rfloor$

---

## 9. Policy Gradient Calculation

### 9.1 Worker Policy Gradient
$$\nabla_{\theta_w} J_w = \mathbb{E}\left[\sum_{t=0}^{T-1} \nabla_{\theta_w} \log \pi_w(a_t|s_t, g_m) A_t^w\right]$$

### 9.2 Score Function Derivative

For softmax policy $\pi_w(j|s_t, g_m) = \exp(\ell_j)/Z$:

$$\frac{\partial}{\partial\theta_w} \log \pi_w(j|s_t, g_m) = \frac{\partial\ell_j}{\partial\theta_w} - \sum_{k \in \mathcal{N}(v_t)} \pi_w(k|s_t, g_m) \frac{\partial\ell_k}{\partial\theta_w}$$

### 9.3 Manager Policy Gradient
$$\nabla_{\theta_m} J_m = \mathbb{E}\left[\sum_{m=0}^{M-1} \nabla_{\theta_m} \log \pi_m(g_m|s_{mK}^m) A_m^m\right]$$

### 9.4 Complete Gradient Update
$$\theta_w \gets \theta_w + \alpha_w \nabla_{\theta_w} J_w$$

$$\theta_m \gets \theta_m + \alpha_m \nabla_{\theta_m} J_m$$

$$\theta_{V_w} \gets \theta_{V_w} - \alpha_v \nabla_{\theta_{V_w}} (V_w(s_t) - R_t)^2$$

$$\theta_{V_m} \gets \theta_{V_m} - \alpha_v \nabla_{\theta_{V_m}} (V_m(s_t^m) - R_m)^2$$

---

## 10. Loss Function Components

### 10.1 Policy Loss
$$\mathcal{L}_{\text{policy}} = -\mathbb{E}\left[\sum_t \log \pi(a_t|s_t) A_t\right]$$

### 10.2 Value Loss
$$\mathcal{L}_{\text{value}} = \mathbb{E}\left[\sum_t (V(s_t) - R_t)^2\right]$$

### 10.3 Entropy Regularization

For discrete distribution $\pi$:
$$H(\pi) = -\sum_a \pi(a) \log \pi(a)$$

$$\mathcal{L}_{\text{entropy}} = -\beta_e \mathbb{E}\left[\sum_t H(\pi(\cdot|s_t))\right]$$

### 10.4 Total Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \beta_v \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{entropy}} + \lambda\|\theta\|^2$$

---

## 11. Convergence Check

### 11.1 Gradient Norm
$$\text{if } \|\nabla_\theta J\| < \varepsilon : \quad \text{convergence} = \text{True}$$

### 11.2 Policy Change
$$\Delta\pi = \mathbb{E}\left[\sum_t D_{KL}(\pi_{\text{new}}(\cdot|s_t) \| \pi_{\text{old}}(\cdot|s_t))\right]$$

$$\text{if } \Delta\pi < \delta : \quad \text{convergence} = \text{True}$$

---

## 12. Full Episode Transformation

### 12.1 Initialization ($t=0$)
$$v_0 = \text{given\_start}$$
$$h_0 = E_{v_0}$$
$$p_0 = \mathbf{0} \in \mathbb{R}^d$$
$$(q_0, c_0) = \text{LSTM\_init}(h_0)$$
$$\text{visited}_0 = \{v_0\}$$
$$g_0 = p_1 \quad \text{(Initial prototype)}$$

### 12.2 Step $t \to t+1$ Transformation

1. **Form state:** $s_t = (q_t, h_t, p_t, v_{\text{target}}, g_m, \text{visited}_t)$
2. **Manager step:** If $t \bmod K = 0$: $g_m = \text{Manager}(s_t^m)$
3. **Worker state:** $w_t = \text{concat}(q_t, h_t, p_t, g_m)$
4. **Compute scores:** For each $v_j \in \mathcal{N}(v_t)$: $\ell_j = f_w(\text{concat}(w_t, h_j, w_t \odot h_j))$
5. **Policy:** $\pi_w(j) = \exp(\ell_j)/\sum_k \exp(\ell_k)$
6. **Sample:** $a_t \sim \pi_w$
7. **Transition:** $v_{t+1} = v_{a_t}$
8. **Update embedding:** $h_{t+1} = E_{v_{t+1}}$
9. **Update path:** $p_{t+1} = \text{GRU}(h_{t+1}, p_t)$
10. **Update query:** $(q_{t+1}, c_{t+1}) = \text{LSTM}(h_{t+1}, (q_t, c_t))$
11. **Update visited:** $\text{visited}_{t+1} = \text{visited}_t \cup \{v_{t+1}\}$
12. **Compute reward:** $r_t = \text{Reward}(h_t, h_{t+1}, g_m, v_{\text{target}}, \text{visited}_{t+1})$
13. **Store transition:** $(s_t, a_t, r_t, s_{t+1})$

---

## 13. Mathematical Properties

### 13.1 Markov Property Preservation
$$P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_t, a_t, s_{t-1}, \ldots)$$

because:
- $h_{t+1}$ depends only on $v_{t+1}$ (deterministic given $a_t$)
- $p_{t+1}$ depends only on $(h_{t+1}, p_t)$
- $q_{t+1}$ depends only on $(h_{t+1}, q_t, c_t)$
- $c_{t+1}$ depends only on $(h_{t+1}, q_t, c_t)$
- $\text{visited}_{t+1} = \text{visited}_t \cup \{v_{t+1}\}$ (deterministic)

### 13.2 Policy Gradient Variance
$$\text{Var}[\nabla_\theta \log \pi(a|s)A] = \text{Var}[\nabla_\theta \log \pi(a|s)] \text{Var}[A] + (\mathbb{E}[\nabla_\theta \log \pi(a|s)])^2 \text{Var}[A] + (\mathbb{E}[A])^2 \text{Var}[\nabla_\theta \log \pi(a|s)]$$

### 13.3 Convergence Rate

For learning rate $\alpha_t$ satisfying Robbins-Monro conditions:
$$\sum_t \alpha_t = \infty, \quad \sum_t \alpha_t^2 < \infty$$

Then $\theta_t$ converges almost surely to $\theta^*$ where $\nabla_\theta J(\theta^*) = 0$

---

## 14. Computational Complexity

### 14.1 Per-Step Complexity

Let $|\mathcal{N}|$ = average neighborhood size

**Worker step:**
$$O(d^2 + |\mathcal{N}|d) \text{ operations}$$

**Manager step (every $K$ steps):**
$$O(Ld + d^2) \text{ without AMR}$$
$$O(Md + |E_A|d) \text{ with AMR}$$

**Total per episode:**
$$O\left(T(d^2 + |\mathcal{N}|d) + \frac{T}{K}(Ld + d^2)\right)$$

### 14.2 Memory Requirements
- **Parameters:** $O((4d)^2 + (6d)^2 + Ld + d_h d)$
- **Activations:** $O(T \times (4d + |\mathcal{N}|))$
- **Gradients:** Same as parameters

---

## 15. Complete Algorithm Pseudocode

```
Algorithm: AMR-HRL for Query Path Reasoning

Input: G=(V,E), q, v_0, v^*, T, K, γ_w, γ_m
Output: Path τ = (v_0, v_1, ..., v_T), Total reward R

Initialize:
    h_v for all v ∈ V
    q = BERT(q)
    θ_w, θ_m, θ_v^w, θ_v^m randomly
    
for episode = 1 to N:
    v_t = v_0
    p_t = 0, (q_t, c_t) = LSTM_init(h_0)
    visited = {v_0}
    g_m = p_1  # Default prototype
    
    for t = 0 to T-1:
        # State formation
        s_t = (q_t, h_t, p_t, v_target, g_m, visited)
        
        # Manager step
        if t mod K == 0:
            s_t^m = concat(q_t, h_t, p_t, v_target)
            if AMR_available:
                g_m = AMR_override(s_t^m, C, G_A)
            else:
                π_m = softmax(W_m s_t^m)
                g_idx ∼ π_m
                g_m = P[g_idx]
        
        # Worker action
        w_t = concat(q_t, h_t, p_t, g_m)
        scores = []
        for v_j in N(v_t):
            c_j = concat(w_t, h_j, w_t ⊙ h_j)
            ℓ_j = f_w(c_j; θ_w)
            scores.append(ℓ_j)
        π_w = softmax(scores)
        a_t ∼ π_w
        v_{t+1} = N(v_t)[a_t]
        
        # State update
        h_{t+1} = E_{v_{t+1}}
        p_{t+1} = GRU(h_{t+1}, p_t)
        (q_{t+1}, c_{t+1}) = LSTM(h_{t+1}, (q_t, c_t))
        visited.add(v_{t+1})
        
        # Reward
        r_t = compute_reward(h_t, h_{t+1}, g_m, v_target, visited)
        
        # Store transition
        store(s_t, a_t, r_t, s_{t+1})
        
        # Update v_t
        v_t = v_{t+1}
    
    # Update parameters
    compute_advantages()  # GAE
    update_parameters()   # Policy gradient
    
    if convergence_check():
        break
```

---

## Key Transformations Summary

1. **State Space:** $\mathbb{R}^d \to \mathbb{R}^{4d}$ via concatenation
2. **Policy:** $\pi(a|s) = \pi_w(a|s,g)\pi_m(g|s)$ (hierarchical factorization)
3. **Embedding:** $h_v = f_{\text{GNN}}(v)$ or lookup table
4. **Memory:** $p_t = \text{GRU}(h_t, p_{t-1})$, $q_t = \text{LSTM}(h_t, q_{t-1})$
5. **Reward:** $r_t = f_{\text{extrinsic}} + f_{\text{intrinsic}} + f_{\text{subgoal}}$
6. **Update:** $\theta \gets \theta + \alpha\nabla_\theta J$ with GAE variance reduction

This complete step-by-step transformation shows every mathematical operation in the AMR-HRL pipeline, providing full transparency for reproduction and analysis.