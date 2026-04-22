---
title: "Deep RL for Anatomical Landmark Navigation in 3D Brain MRI"
author: "Chris Drake"
date: "CSCE 775 --- April 2026"
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

\small

**Task.** Drop an agent at a random voxel in a 3D brain MRI; learn to step
one voxel at a time to a named subcortical landmark (15 MNI152 targets).
**Why RL?** Weak supervision, interpretable trajectory, open-vocabulary.

\begin{center}
\includegraphics[height=0.5\textheight,keepaspectratio]{figures/system_diagram.png}
\end{center}

State $s_t = (\mathbf{n}_t,\ k\cdot\hat{\mathbf{d}}_t) \in \mathbb{R}^{346}$ ---
$7^3$ voxel patch + scaled unit direction. Actions $\{\pm x, \pm y, \pm z\}$.
Reward $-(d_{t+1}{-}d_t) - 0.1$ ($+10$ on success). 200-step cap.
Platform: TF.js + Niivue, runs in the browser.

# Agents + sanity baselines

\small

**Learners** (flat 346-d input, Adam 3e-4):

- **DQN** --- Q-net 256$\to$128$\to$64, $\epsilon$-greedy, replay, target sync.
- **A2C** --- 128$\to$64 trunk, GAE($\lambda{=}0.95$), entropy 0.01.
- **PPO** --- same trunk, clipped ($\epsilon_\text{clip}{=}0.2$), roll 4 / mb 64.

**Non-learners.** Oracle: $\arg\max_i |\hat d_i|$ (sign). Random: uniform.
Training: 300 eps / seed, 3 seeds / config.

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
\footnotesize Env well-posed: oracle wins in roughly 50 steps; random never
wins in 500 eps.

# Three-algorithm comparison (15 landmarks)

\begin{center}
\includegraphics[height=0.55\textheight,keepaspectratio]{figures/main-comparison/reward_curves.png}
\end{center}

\small

- **PPO wins 11/15 by reward, 13/15 by distance** (mean $R$: $-18.7$ vs.
  A2C $-37.6$ vs. DQN $-32.6$).
- A2C second; diverges on Hippocampus/Putamen (advantage variance).
- DQN *regresses* as $\epsilon$ decays --- overestimation + forgetting.

# Key finding: direction-scale $k$ --- a $20\times$ jump

\small

State concatenates 343 voxel values with a 3-dim unit direction vector.
Default encoding drowns the direction signal. We multiply $\hat{\mathbf{d}}$
by $k$ before concatenation.

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

- **One-line change** lifts mean last-50 success from roughly 3\% to
  roughly 24\% --- a $20\times$ improvement on these landmarks.
- Over-amplifying ($k{=}30, 100$) destabilizes training: $k{=}100$
  collapses Lateral-Ventricle on all 3 seeds.
- Opposite direction --- enlarging the voxel window ($7^3 \to 15^3$, both
  flat and conv) --- **also hurts**. The direction vector, not the voxel
  context, is the load-bearing signal.
- **General lesson:** when one component of a concatenated state is much
  lower-dim, default encoding under-weights it.

# Direction-scale learning curves

\begin{center}
\includegraphics[height=0.65\textheight,keepaspectratio]{figures/dirscale-sweep/success_rate.png}
\end{center}

\small
$k=10$ (lightest green) dominates on both landmarks. Compare with $k=1$
baseline (2--4\% on the same landmarks).

# Stabilization: curriculum + larger rollouts

\small

**Problem.** Across-seed std in last-50 success is 20--30 pp --- dominates
our error bars. Cause: tiny rollouts (roughly 200 transitions per update) +
softmax-sensitive policy head $\Rightarrow$ a seed that commits to the wrong
axis early has no gradient to escape.

**Intervention** at $k{=}10$, 5 seeds: curriculum (start radius 20 $\to$ 50
voxels over 150 eps) + roll 16, mb 256 (up from 4 / 64).

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

- **Lateral-Ventricle:** all 5 seeds learn (26 / 24 / 44 / **68** / **60**\%);
  new best single-seed result in the project.
- **Thalamus:** 2 of 5 seeds still collapse to 0--4\%. Curriculum helps the
  seeds that *can* bootstrap, doesn't stop wrong-axis commitment.
- **Next lever:** online collapse detection + restart --- not more
  hyperparameter search.

# Summary & future work

\small

**What we showed.**

- **PPO wins** on 11/15 landmarks by reward, 13/15 by distance.
- **Direction-scale $k{=}10$** --- one-line state rescaling --- gives a
  $20\times$ success improvement; suggests a general lesson for
  concatenative RL state representations.
- **Curriculum + larger rollouts** push Lateral-Ventricle to 44\% (best seed
  68\%); Thalamus wrong-axis collapse remains the failure mode.

\vspace{0.3em}

**Future work.**

1. 10+ seeds with online collapse detection + restart.
2. Cross-subject training on OpenNeuro cohorts (template $\to$ real data).
3. Architectural direction-conditioning (FiLM-style) as the principled
   replacement for the scalar-$k$ trick.
4. Multi-agent shared-representation (Vlontzos 2019) across the 15 landmarks.

\vspace{0.5em}

\centering
\Large Thank you --- questions?
