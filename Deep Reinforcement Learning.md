# Deep Reinforcement Learning

Review notes for [Berkeley CS285 2020F](http://rail.eecs.berkeley.edu/deeprlcourse/).

## Introduction

### Goal of DRL

Maximize Expected Reward:

$$
p_\theta(\tau)=p_\theta(s_1,a_1,...,s_T,a_T)=p(s_1)\prod_t^T \pi_\theta(a_t|s_t) p(s_{t+1}|s_t,a_t)
$$

$$
\theta^*=\arg\max_\theta \mathbb{E}_{p_\theta}\Big[\sum_t r(s_t,a_t)\Big]
$$

$$
J(\theta)=\mathbb{E}_{p_\theta}\Big[\sum_t r(s_t,a_t)\Big] \approx \frac{1}{N}\sum_i\sum_t r(s_t,a_t)
$$

## Model-Free
### 1. Policy Gradients

Directly differentiate the objective.  Unbiased but has high variance. On-policy.
$$
\begin{align}
\nabla_\theta J(\theta)&\approx\frac{1}{N}\sum_i\Big(\sum_t\nabla_\theta\log\pi(a_t|s_t)\Big)\Big(\sum_t r(s_t,a_t)\Big)\\
&=\frac{1}{N}\sum_i\sum_t\nabla_\theta\log\pi(a_t|s_t)\Big(\sum_{t'=t}r(s_{t'},a_{t'})\Big) \ \text{(causality trick)}\\
&=\frac{1}{N}\sum_i\sum_t\nabla_\theta\log\pi(a_t|s_t)\Big(\sum_{t'=t}r(s_{t'},a_{t'})-b\Big) \ \text{(baseline)}
\end{align}
$$

**REINFORCE**:
$$
\begin{align}
&1. \text{sample \{$\tau$\} using $\pi_\theta(a_t|s_t)$}\\
&2. \nabla_\theta J(\theta)\\
&3. \text{update $\theta$}
\end{align}
$$

### 2. Value Function Methods

$$
\begin{align}
&Q^{\pi}(s_t,a_t)=\sum_{t'=t}^T\mathbb{E}_{\pi}[r(s_{t'},a_{t'})|s_t,a_t]\\
&V^{\pi}(s_t)=\mathbb{E}_{a_t\sim\pi(a_t|s_t)}[Q^{\pi}(s_t,a_t)]\\
&A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^\pi(s_t)\\
&Q^\pi(s_t,a_t)\approx r(s_t,a_t)+V^\pi(s_{t+1})\\
&A^\pi(s_t,a_t)\approx r(s_t,a_t)+V^\pi(s_{t+1})-V^\pi(s_t)
\end{align}
$$

### Actor-Critic

Lower variance but biased because the estimator is not accurate.

**Batch Actor-Critic**:
$$
\begin{align}
&1.\text{sample $\tau$ to collect }\{(s_t, \underbrace{r(s_t,a_t)+\gamma \hat{V}^\pi_\phi(s_{t+1}))}_{y}\} \text{(bootstrap estimate)}\\
&2.\text{fit }L(\phi)=||\hat{V}^\pi_\phi(s_t)-y||^2\\
&3.\text{evaluate }\hat{A}^\pi(s_t,a_t)\approx r(s_t,a_t)+\gamma \hat{V}_\phi^\pi(s_{t+1})-\hat{V}_\phi^\pi(s_t)\\
&4.\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_i\sum_t\nabla_\theta\log\pi(a_t|s_t)\hat{A}^\pi(s_t,a_t)\\
&5.\text{update } \theta
\end{align}
$$

### Eligibility traces & n-step returns

### Argmax Policy

$$
\pi'(a_t|s_t)=I(a_t=\arg\max_{a_t} A^\pi(s_t,a_t))
$$

$$
Q^\pi(s,a)\leftarrow r(s,a)+\gamma\mathbb{E}_{s'\sim p(s'|s,a)}[V^\pi(s')]\\
V^\pi(s)\leftarrow \max_a Q^\pi(s,a)
$$

Removing the requirement for the transition dynamics:
$$
\begin{align}
&1.\text{collect }\{(s,a,s',r)\}\\
&K\times\begin{cases}
2.y\leftarrow \max(r(s,a)+\gamma\mathbb{E}_{s'\sim p(s'|s,a)}[\hat{Q}_\phi(s',\pi(s'))])\approx r(s,a)+\gamma\max_{a'}\hat{Q}_\phi(s',a')\\
3.\text{fit } L(\phi)=||\hat{Q}_\phi(s,a)-y||^2 \text{ (moving target!)}
\end{cases}\\
\end{align}
$$
Off-policy because that **max**. Given **(s,a,r)**, transition is independent of **\pi**. **No converge guarantees.** The regression problem isn't giving the true gradients because **y** actually depends on **\phi**.

### Q-learning With Target Networks

$$
\begin{align}
&1.\text{target network }\phi'\leftarrow\phi\\
&N\times\begin{cases}
    2.\text{collect $\{(s,a,s',r)\}$ and add to }\mathcal{B}\\
    K\times\begin{cases}
        3.\text{sample a batch from $\mathcal{B}$}\\
        4.y\leftarrow r(s,a)+\gamma\max_{a'}\hat{Q}_{\phi'}(s',a')\\
        5.\text{fit } L(\phi)=||\hat{Q}_\phi(s,a)-y||^2 
    \end{cases}\\
\end{cases}
\end{align}
$$



## Model-Based

### 4.1 Planning

Cross-entropy method (CEM)

Monte Carlo tree search (MCTS)

LRQ, DDP

### 4.2 Learning the Model

$$
s_{t+1}=f(s_t,a_t)\ \text{ or } \ p(s_{t+1}|s_t,a_t)
$$

Naive:
$$
\begin{align}
&1. \text{run }\pi_0(a_t|s_t)\text{ to collect }\mathcal{D}=\{(s,a,s')_i\}\\
&2. \text{learn }f(s,a)\min||f(s_i,a_i)-s_i'||^2\\
&3. \text{plan through }f(s,a)\\
&\text{distribution mismatch: }p_{\pi_f}(s_t)\ne p_{\pi_0}(s_t)\\
&4. \text{execute the first planned action (MPC)} \\
&5. \text{append }(s,a,s')
\end{align}
$$
Problem:



### 4.3 Uncertainty

#### Uncertainty Estimation

Help with the over-fitting problem of $f$: choose actions according to the uncertainty dynamics.

#### Uncertainty-Aware Neural Network Model

1. Use output entropy. Bad, wrong kind of uncertainty: statistical uncertainty.
2. Estimate model uncertainty. "The model is certain about the data, but we are not certain about the model". 

#### Bayesian Neural Networks

Weight Uncertainty in Neural Networks.

#### Bootstrap Ensembles

$$
p(\theta|\mathcal{D})\approx\frac{1}{N}\sum_i\delta(\theta_i)\\
\frac{1}{N}\sum_i p(s_{t+1}|s_t,a_t,\theta_i)\\
\text{each $\theta_i$ is trained on $\mathcal{D}_i$ sampled with replacement from $\mathcal{D}$}
$$

#### Planning With Uncertainty?

### 4.4 (Latent) State Space Model

$$
\mathbb{E}\Big[\log p_\phi(s_{t+1}|s_t,a_t)+\log p_\phi(o_t|s_t) \Big]
$$

Simple single-step encoder:
$$
q_\psi(s_t|o_t)=g_\psi(o_t)\\
\max_{\phi,\psi}\sum\sum\log p_\phi(g_\psi(o_{t+1})|g_\psi(o_t),a_t)+\log p_\phi(o_t|g_\psi(o_t))+\log p_\phi(r_t|g_\phi(o_t))
$$


### 4.5 Learning the Policy

Open-loop (suboptimal)
$$
p_\theta(s_1,...,s_T|a_1,...,a_T)=p(s_1)\prod_{t=1}^T p(s_{t+1}|s_t,a_t)\\
a_1,...,a_T=\arg\max\mathbb{E}\Big[\sum_t^T r(s_t,a_t)|a_1,...,a_T\Big]
$$

### 4.6 Model-Free RL with a Model

 #### Dyna-like Models

Online Q-learning that performs model-free RL with a model.
$$
\begin{align}
&1.\text{collect}\\
&2.\\
&3.\\
\end{align}
$$
problem: distributional shift

#### Local Models

## Exploration

$$
Reg(T)=TE[r(a^*)]-\sum_t^T r(a_t)
$$

Optimistic Exploration:
$$
a=\arg\max(\hat{\mu}_a+C\sigma_a)
$$
Probability matching: maintain a belief
$$

$$


Information Gain
$$
H(p(z))\\
H(p(z)|y) (e.g.y=r(a))\\
IG(z,y|a)=E_y[H(p(z))-H(p(z)|y)|a]
$$

# Variational Inference

$$
\mathbb{E}_{z\sim p(z|x)}[\log p_\theta(x,z)]
$$

# Inference and Control

$$
p(\mathcal{O}_t|s_t,a_t)=\exp(r(s_t,a_t))\\
p(\tau|\mathcal{O}_{1:T})=\frac{p(\tau,\mathcal{O}_{1:T})}{p(\mathcal{O}_{1:T})}\\
$$



