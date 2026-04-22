---
title: "Deep RL for Anatomical Landmark Navigation in 3D Brain MRI"
author: "Chris Drake"
date: "CSCE 775, April 2026"
aspectratio: 169
theme: "Madrid"
colortheme: "seahorse"
fonttheme: "structurebold"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{array}
  - \setbeamertemplate{navigation symbols}{}
  - \setbeamerfont{frametitle}{size=\large}
  - \setbeamerfont{block title}{size=\small}
---

# Problem & formulation

\footnotesize

**Task.** Drop an agent at a random voxel in a 3D brain MRI; learn to step
one voxel at a time to a named subcortical landmark (15 MNI152 targets).
**Why RL.** No per-voxel labels, inspectable trajectories, one
goal-conditioned policy in principle.

\begin{center}
\includegraphics[height=0.45\textheight,keepaspectratio]{figures/system_diagram.png}
\end{center}

State $s_t = (\mathbf{n}_t,\ k\cdot\hat{\mathbf{d}}_t) \in \mathbb{R}^{346}$:
a $7^3$ voxel patch concatenated with a scaled unit direction vector to
the target. Actions $\{\pm x, \pm y, \pm z\}$. Reward
$-(d_{t+1}{-}d_t) - 0.1$ with $+10$ on success. Episode cap 200 steps.
Built on TF.js + Niivue so the whole loop runs in the browser.

# Agents and network architectures

\footnotesize

**Learners** (flat 346-d input, Adam 3e-4). DQN: $\epsilon$-greedy Q-net
with replay + target sync. A2C: GAE ($\lambda=0.95$), entropy 0.01. PPO:
clipped surrogate ($\epsilon_\text{clip}=0.2$), rollout 4 / minibatch 64.

\begin{center}
\includegraphics[height=0.58\textheight,keepaspectratio]{figures/network_architectures.png}
\end{center}

# Sanity baselines

\small

**Non-learners.** Oracle picks $\arg\max_i |\hat d_i|$ with the sign of
$\hat d_i$. Random samples uniformly. Training for learners: 300 episodes
per seed, 3 seeds per configuration.

\centering
\footnotesize
\begin{tabular}{lrrrr}
\toprule
Landmark (5 of 15) & Oracle succ. & Oracle steps & Random succ. & Random rwd. \\
\midrule
Thalamus           & 100\% & 48 & 0\% & $-20.9$ \\
Hippocampus        & 100\% & 53 & 0\% & $-22.5$ \\
Lateral-Ventricle  & 100\% & 50 & 0\% & $-21.5$ \\
Brain-Stem         & 100\% & 51 & 0\% & $-22.4$ \\
Putamen            & 100\% & 51 & 0\% & $-22.1$ \\
\bottomrule
\end{tabular}

\raggedright
\footnotesize Sanity check: oracle reaches the target in roughly 50 steps
(consistent with a greedy walk from 50 voxels away); random never reaches
it in 500 episodes. The environment is well-posed.

# Three-algorithm comparison (15 landmarks)

\begin{center}
\includegraphics[height=0.55\textheight,keepaspectratio]{figures/main-comparison/reward_curves.png}
\end{center}

\small

- PPO is the strongest learner: best on 11/15 landmarks by reward, 13/15 by
  final distance (mean $R$: $-18.7$ vs. A2C $-37.6$ vs. DQN $-32.6$).
- A2C comes second but diverges on Hippocampus and Putamen (advantage
  variance under the small batch).
- DQN *regresses* as $\epsilon$ decays, consistent with Q-value
  overestimation plus catastrophic forgetting in the replay buffer.

# Direction-scale $k$: weighting the goal vector

\small

The state concatenates 343 voxel intensities with a 3-dim unit direction
vector. The voxel patch outweighs the direction signal by two orders of
magnitude in input components, and the network appears to under-use it. We
multiply $\hat{\mathbf{d}}$ by a scalar $k$ before concatenation.

\centering
\begin{tabular}{lrrr}
\toprule
Landmark          & $k=10$                  & $k=30$              & $k=100$              \\
\midrule
Lateral-Ventricle & \textbf{28.0 $\pm$ 20.0\%} & 20.7 $\pm$ 29.1\%  & 0.0 $\pm$ 0.0\%      \\
Thalamus          & \textbf{20.7 $\pm$ 3.1\%}  & 3.3 $\pm$ 5.8\%    & 17.3 $\pm$ 30.0\%    \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\raggedright

- A one-line change to the observation lifts mean last-50 success from
  roughly 3\% (default $k=1$) to roughly 24\% on these landmarks, an
  order-of-magnitude improvement.
- Over-amplifying ($k=30, 100$) destabilizes training. $k=100$ collapses
  Lateral-Ventricle on all 3 seeds.
- Going the other way (enlarging the voxel window from $7^3$ to $15^3$,
  both flat and conv variants) **also hurts**. The direction vector, not
  the voxel context, is the load-bearing signal in our setup.
- The takeaway for concatenated-state RL: when one component is much
  lower-dimensional than the rest, the default encoding under-weights it,
  and the cheapest fix is explicit rescaling at the input.

# Direction-scale learning curves

\begin{center}
\includegraphics[height=0.65\textheight,keepaspectratio]{figures/dirscale-sweep/success_rate.png}
\end{center}

\small
$k=10$ (lightest green) dominates on both landmarks. The default $k=1$
(not plotted; reported on the previous slide) sits at 2--4\% on the same
landmarks.

# Stabilization: curriculum + larger rollouts

\small

**Problem.** Across-seed std in last-50 success is 20--30 pp, which
dominates our error bars. Plausible cause: tiny on-policy rollouts
(roughly 200 transitions per update) feeding a softmax-sensitive policy
head. A seed that commits to the wrong action axis early gets little
corrective gradient and stays stuck.

**Intervention** at $k=10$, 5 seeds: a linear curriculum on starting
radius (20 $\to$ 50 voxels over 150 episodes), plus 4$\times$ larger
on-policy rollouts (rollout 16, minibatch 256, up from 4 / 64).

\centering
\begin{tabular}{lrr}
\toprule
Landmark          & Baseline (3 seeds, roll 4) & Curriculum + roll 16 (5 seeds) \\
\midrule
Lateral-Ventricle & 28.0 $\pm$ 20.0\%          & \textbf{44.4 $\pm$ 19.7\%} (+16 pp) \\
Thalamus          & 20.7 $\pm$ 3.1\%           & 21.6 $\pm$ 24.3\%              \\
\bottomrule
\end{tabular}

\vspace{0.4em}
\raggedright

- **Lateral-Ventricle.** All 5 seeds learn (per-seed last-50:
  26 / 24 / 44 / **68** / **60**\%); the 68\% seed is our best
  single-seed result anywhere in the project.
- **Thalamus.** 2 of 5 seeds still collapse to 0--4\%. Curriculum helps
  the seeds that *can* bootstrap a useful direction signal; it does not
  prevent wrong-axis commitment.
- **Next thing to try.** Online collapse detection and seed restart, not
  more hyperparameter sweeping.

# Multi-scale (coarse-fine) observations

\footnotesize

Adding a stride-4 channel: the agent sees the same $7^3$ patch at 1 mm
(7 mm field of view) **and** at 4 mm (28 mm field). State width
$343 \to 686$ voxel components. Same PPO config; both arms run to 600
episodes for a budget-matched comparison.

\centering
\begin{tabular}{lrrrr}
\toprule
Landmark / agent           & Last-50 succ.        & Best-100 succ.       & Dist & Steps  \\
\midrule
LV \quad single-scale $[1]$  & 13.3 $\pm$ 8.1\%     & 25.7 $\pm$ 7.0\%     & 25.2         & 184    \\
LV \quad multi-scale $[1,4]$ & \textbf{38.7 $\pm$ 21.4\%} & \textbf{46.0 $\pm$ 14.5\%} & \textbf{16.3} & \textbf{153} \\
\midrule
Thal. single-scale $[1]$  & 20.0 $\pm$ 32.9\% & \textbf{52.7 $\pm$ 5.7\%} & 65.8     & 177    \\
Thal. multi-scale $[1,4]$ & \textbf{32.0 $\pm$ 15.9\%} & 39.7 $\pm$ 7.4\% & \textbf{25.2} & \textbf{170} \\
\bottomrule
\end{tabular}

\vspace{0.3em}
\raggedright

- **Lateral-Ventricle.** Multi-scale roughly *triples* last-50 success
  and roughly halves final distance. Clean win.
- **Thalamus.** Single-scale peaks higher (best-100 53\%) but collapses
  on 2/3 seeds; multi-scale plateaus lower (40\%) but holds (last-50 std
  drops from 33 pp to 16 pp). It trades peak performance for stability.
- **At the matched 300-episode budget** (not in the table) multi-scale
  loses badly: the doubled state width takes roughly 400 episodes to
  amortize under a shared trunk. A per-scale (Ghesu-style hierarchical)
  agent would avoid that slow start.

# Findings and future work

\small

**What we found.**

- **PPO is the strongest learner**, winning on 11/15 landmarks by reward
  and 13/15 by final distance.
- **Direction-scale $k=10$**, a one-line rescaling of the goal component,
  raises last-50 success by roughly an order of magnitude on the two
  landmarks we tested.
- **Curriculum + larger rollouts** push Lateral-Ventricle to 44\%
  (best seed 68\%); the wrong-axis collapse on Thalamus is the
  remaining failure mode.
- **Multi-scale observations** triple Lateral-Ventricle success at
  600 episodes and roughly halve Thalamus variance, at the cost of a
  slow start that a per-scale agent would avoid.

\vspace{0.3em}

**Future work.**

1. 10+ seeds with online collapse detection and seed restart.
2. Cross-subject training on OpenNeuro cohorts (template $\to$ real subjects).
3. Architectural direction-conditioning (FiLM-style) as a principled
   replacement for the scalar-$k$ trick.
4. Hierarchical multi-scale agents in the Ghesu form: a coarse-stage policy
   hands off to a fine-stage policy at a learned distance threshold.
5. Multi-agent shared-representation (Vlontzos 2019) across the 15 landmarks.

\vspace{0.5em}

\centering
\Large Thank you. Questions?
