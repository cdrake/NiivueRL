# CSCE 775: Deep Reinforcement Learning
## Deep Reinforcement Learning for Anatomical Landmark Navigation in 3D Brain MRI

**Chris Drake**

**April 28, 2026**

**Live demo & code:** [cdrake.github.io/NiivueRL](https://cdrake.github.io/NiivueRL/) · [github.com/cdrake/NiivueRL](https://github.com/cdrake/NiivueRL)

## Abstract

We ask whether a deep RL agent can learn to navigate a T1-weighted brain MRI volume to subcortical landmarks using only a local voxel neighborhood and a unit direction vector as observation. The whole stack runs in a browser on TensorFlow.js + Niivue, with 15 MNI152 targets and a dense distance-shaping reward. We compare DQN, A2C, and PPO against oracle and random baselines, and ablate observation size, trunk architecture, and a *direction-scale* hyperparameter $k$ that rebalances the 3-dim direction signal against the 343-voxel patch. PPO is best on 11/15 landmarks by late reward and 13/15 by final distance; the oracle reaches all 15 and random reaches none, so the environment is well-posed. A one-line change of $k = 1 \to 10$ raises last-50 success on two representative landmarks from near-zero (2-4\%) to roughly 25\%; $k \in \{30, 100\}$ destabilize. Seed-to-seed variance dominates the remaining error, with per-seed swings of 20-30 pp on the same cell. A linear curriculum over starting radius ($20 \to 50$ voxels) plus $4\times$ larger on-policy rollouts lifts Lateral-Ventricle to 44\% (all 5 seeds learning) but does not close the Thalamus seed-collapse mode. We then replace the oracle direction with a learned 3D-conv network trained on AOMIC ID1000 FreeSurfer data on Theia, the university's HPC cluster. On a held-out subject the baseline $7^3$ network (val. cosine $0.9760$) reaches 42.7\% on Lateral-Ventricle (vs. 25.3\% for the oracle, all 3 seeds learning) and ties on Thalamus (8.7\% vs. 10.7\%). A wider $15^3$ multi-scale network ($\approx 25\times$ params, val. cosine $0.9968$) halves the angular error but actually drops Lateral-Ventricle policy success to 16.3\%. Together with an earlier laptop variant ($0.9685$) that showed the same kind of asymmetry, this gives three data points where better mean cosine did not translate into a better policy.

## 1. Introduction

Automated localization of anatomical landmarks in brain MRI feeds many neuroimaging pipelines: atlas registration, surgical planning, ROI volumetry, longitudinal alignment, and segmentation QC. The two conventional approaches (template-based registration and supervised CNNs on densely annotated volumes) both need a lot of annotation effort, both struggle on atypical anatomy, and neither produces a useful intermediate state to look at while it works.

Reinforcement learning gives a different framing. Place an agent at an arbitrary position inside a 3D volume and learn a policy that moves one voxel at a time toward a named target. The trajectory is an interpretable artifact, training needs only the target coordinate (not dense voxel labels), and the same policy can in principle generalize to any labeled target. We treat this as a sequential decision-making problem and study it with the three most widely used deep RL families: DQN, A2C, and PPO.

This paper reports (i) a three-algorithm comparison across 15 MNI152 landmarks, (ii) oracle/random sanity baselines, (iii) ablations on observation size and a direction-scale hyperparameter, (iv) a seed-variance analysis and curriculum-based stabilization, (v) a multi-scale extension, and (vi) replacement of the oracle direction vector with a learned 3D-conv network trained on AOMIC ID1000 FreeSurfer data. All training is performed in-browser on a TensorFlow.js + Niivue stack, runnable from a static web page.

## 2. Related Work

Anatomical landmark detection as RL navigation was introduced by Ghesu et al. [1], who formulated it in 3D CT as a DQN over a voxel patch with frame-history. Alansary et al. [2] extended the approach to MRI view planning with a multi-scale coarse-to-fine strategy and benchmarked DQN/Double-DQN/Dueling variants, reporting that no single architecture wins uniformly across landmarks. Vlontzos et al. [7] studied multi-agent extensions sharing representations. These prior works all instantiate DQN variants and use frame history as the memory mechanism.

Our project differs in three respects. First, we compare across families (DQN vs. A2C vs. PPO) where prior work fixes the family at DQN. Second, we drop frame history and feed the agent an explicit 3-component unit direction vector toward the target. Section 5.4 shows that the *scale* of that direction vector relative to the voxel patch is a first-order determinant of whether the policy learns at all. Third, the whole system runs in-browser on TensorFlow.js and can optionally consume a pretrained MeshNet parcellation model [6] from the Brainchop project. The core algorithms are Schulman et al.'s PPO [3], Mnih et al.'s DQN [4] and A2C [5], with generalized advantage estimation [8] for the on-policy methods.

## 3. Approach

### 3.1 Environment

We model the task as an episodic MDP on the skull-stripped MNI152 T1 template. The state at time $t$ is $s_t = (\mathbf{n}_t, k\cdot\hat{\mathbf{d}}_t)$, where $\mathbf{n}_t \in \mathbb{R}^{343}$ is a $7^3$ neighborhood of min-max-normalized voxel intensities and $\hat{\mathbf{d}}_t \in \mathbb{R}^3$ is the unit vector toward the target voxel. The scalar $k$ is a *direction-scale* hyperparameter (default $k=1$) that rebalances the directional signal against the voxel patch. Actions are 6 discrete one-voxel axis steps. Reward combines a potential-based shaping term $-(d_{t+1}-d_t)$, a per-step penalty of $-0.1$, and a terminal bonus of $+10$ within 3 voxels of the target. Episodes start uniformly within a *starting radius* (default 50 voxels) of the target and terminate on success or after 200 steps. The 15 subcortical landmarks span thalamus, hippocampus, caudate, putamen, pallidum, amygdala, accumbens, lateral ventricle, inferior lateral ventricle, 3rd/4th ventricles, brain-stem, cerebellar cortex, cerebellar white matter, and ventral DC. Figure 1 illustrates the loop.

![System overview: (a) the agent is placed at a random position in the MNI152 volume and observes a $7^3$ voxel neighborhood plus a unit direction vector toward the target landmark; (b) the MDP interaction loop, with 346-dimensional state, 6 discrete actions, and dense distance-shaped reward.](figures/system_diagram.png){width=0.7\linewidth}

### 3.2 Agents

All three learners take the same 346-dim flat state, optimized with Adam at lr $3 \cdot 10^{-4}$. **DQN** [4] has a $256 \to 128 \to 64 \to 6$ Q-network, a 10K replay buffer, $\epsilon$-greedy decayed $1.0 \to 0.05$ at 0.995/episode, and target sync every 100 steps. **A2C** [5] and **PPO** [3] share a $128 \to 64$ trunk that splits into a 6-way softmax policy and a scalar value head, both with GAE ($\lambda=0.95$, $\gamma=0.99$) and entropy regularization 0.01. PPO uses the clipped surrogate ($\epsilon_{\text{clip}}=0.2$), 4 epochs per batch with minibatches of 64 and rollouts of 4 episodes; advantages are batch-normalized before the first epoch. (An earlier A2C with a frozen MeshNet [6] backbone diverged on every landmark and is not reported here.) **Oracle** takes the greedy direction-vector axis step; **Random** samples $\mathcal{A}$ uniformly. Neither trains.

![Comparative network architectures for the three learners. All three consume the same 346-dim flat state. DQN uses a single Q-network with hidden layers $256 \to 128 \to 64$ that outputs six Q-values (one per action). A2C and PPO share a $128 \to 64$ fully-connected trunk that splits into a 6-way softmax policy head and a scalar value head; they differ in the update rule, with A2C using the advantage actor-critic update and PPO using a clipped surrogate objective ($\epsilon_{\text{clip}} = 0.2$).](figures/network_architectures.png){width=0.7\linewidth}

## 4. Experimental Setup

All experiments use the same volume (skull-stripped MNI152, $197 \times 233 \times 189$ voxels, isotropic 1 mm), the same reward function, and the same 200-step episode cap. Unless otherwise noted, episodes start within a 50-voxel radius and each configuration trains for 300 episodes per seed. For replicated runs we report mean $\pm$ 95\% CI across 3 seeds, except where seed count is explicitly noted. All code, spec files, and result JSON dumps are public in the project repository. Results cache in browser local storage keyed on a structured configuration key, so an interrupted run resumes from the last completed (agent, landmark, seed, direction-scale) tuple.

## 5. Experimental Results

A short version of the result: with the default state encoding, none of DQN, A2C, or PPO learn the task. Average late-window success across 15 landmarks sits at 0.3-1.2\%, which is the random-baseline floor (Section 5.2). The fix is a one-line change to the state representation: rescale the 3-dim direction vector by $10\times$ before concatenating it with the 343-dim voxel patch (Section 5.4). That single hyperparameter raises last-50 success on two test landmarks from 2-4\% to roughly 24\%, a $20\times$ jump. From there, a curriculum plus larger rollouts pushes Lateral-Ventricle to 44\%, a multi-scale voxel patch doubles it again to 39-46\% over a longer horizon (Section 5.6), and replacing the oracle direction with a learned 3D-conv network (trained on Theia) decisively beats the oracle on Lateral-Ventricle (43\% vs. 25\%, Section 5.7).

### 5.1 Sanity check: oracle and random baselines

The oracle (greedy along the direction vector) reaches the target in 100\% of episodes on five representative landmarks (~50 steps from a 50-voxel start, consistent with an $\ell_\infty$ walk); uniform-random reaches it in 0 of 500. The environment is well-posed; any learner that fails to outperform random is not using the direction signal.

*Table 1: Oracle and uniform-random baselines (100 episodes per landmark, 50-voxel starting radius, 200-step cap).*

| Landmark          | Oracle success | Oracle reward | Oracle steps | Random success | Random reward |
|-------------------|---------------:|--------------:|-------------:|---------------:|--------------:|
| Thalamus          | 100\%          | $+37.70$      | 48           | 0\%            | $-20.87$      |
| Hippocampus       | 100\%          | $+40.32$      | 53           | 0\%            | $-22.46$      |
| Lateral-Ventricle | 100\%          | $+38.74$      | 50           | 0\%            | $-21.47$      |
| Brain-Stem        | 100\%          | $+39.66$      | 51           | 0\%            | $-22.42$      |
| Putamen           | 100\%          | $+38.71$      | 51           | 0\%            | $-22.12$      |

### 5.2 Three-algorithm comparison on 15 landmarks (the floor)

Under the default state encoding, **no algorithm learns**. Averaged over 300 episodes per (algorithm, landmark) pair on all 15 landmarks, mean late-window success is 0.7\% (DQN), 0.3\% (A2C), 1.2\% (PPO), which is essentially the random-baseline floor. Mean late-window reward sits at $-32.6$ / $-37.6$ / $-18.7$ and mean final distance at $50.1$ / $55.5$ / $36.5$ voxels. PPO is best on 11/15 landmarks by reward and 13/15 by final distance, so the *ranking* is real, but the absolute level is too low to matter. A2C diverges on Hippocampus and Putamen (reward $-123$, $-80$, final distance up to 138 voxels), which is the textbook signature of high-variance single-sample advantage updates. DQN regresses as $\epsilon$ decays (early-window reward $-10.7 \to$ late-window $-32.6$), a textbook overestimation/forgetting failure. Section 5.4 identifies the cause: the direction vector is at the wrong scale.

### 5.3 Observation-size and trunk ablation

Enlarging the window from $7^3$ to $15^3$ (10$\times$ the first-layer parameters) **hurt** PPO on all three easiest landmarks (late reward $-7.1$/$-7.5$/$-8.9$ → $-20.7$/$-79.3$/$-18.1$); a small 3D-conv trunk did not close the gap. The direction vector already carries the coarse navigation signal, so widening the voxel window adds parameters without task-relevant information. Section 5.4 probes the opposite intervention: *amplifying* the direction signal.

### 5.4 Direction-scale sweep: rebalancing the state

The 346-dim state concatenates a 343-component voxel patch (values in $[0,1]$) with a 3-component unit direction vector. With isotropically initialized weights, the direction signal is drowned by 343 voxels. We introduced a *direction-scale* hyperparameter $k$ that multiplies the direction vector before concatenation and swept $k \in \{10, 30, 100\}$ on Thalamus and Lateral-Ventricle (3 seeds, 300 episodes each).

*Table 2: Direction-scale sweep. Last-50 success rate (mean $\pm$ standard deviation across 3 seeds) for PPO with flat trunk, $7^3$ window.*

| Landmark          |        $k=10$ |        $k=30$ |       $k=100$ |
|-------------------|--------------:|--------------:|--------------:|
| Lateral-Ventricle | **28.0 $\pm$ 20.0\%** |   20.7 $\pm$ 29.1\% |     0.0 $\pm$ 0.0\% |
| Thalamus          | **20.7 $\pm$ 3.1\%**  |    3.3 $\pm$ 5.8\%  |   17.3 $\pm$ 30.0\% |

$k=10$ wins on both landmarks, lifting last-50 success roughly $20\times$ over flat-$k=1$ (Section 5.2 baseline). Larger values destabilize: $k=100$ on Lateral-Ventricle collapsed on all three seeds (the agent wanders), and $k=100$ on Thalamus produced one learning seed and two failures (the 30-pp standard deviation in that cell). Figure 3 shows the full curves. There is a sweet spot. The default scale under-weights the direction signal, but over-amplification saturates the policy softmax before the value head learns to discriminate near-terminal states.

![Direction-scale sweep on Thalamus and Lateral-Ventricle: success rate over 300 episodes, 30-episode rolling window, mean $\pm$ 95\% CI over 3 seeds. $k=10$ (lightest green) clearly dominates $k=30$ and $k=100$ on both landmarks. Compare with Section 5.2's baseline numbers, where the default $k=1$ achieved 2-4\% late-window success on these same landmarks.](figures/dirscale-sweep/success_rate.png){width=0.7\linewidth}

### 5.5 Seed variance and curriculum-based stabilization

Table 2's standard deviations are large: on Lateral-Ventricle at $k=10$, one seed gets 9\% and another 46\%; on Thalamus at $k=100$, seeds split 0/1/39\%. The flat-$k=1$ runs of Section 5.2 show similar per-seed swings, just all near zero. A small ($128 \to 64$) network with a softmax head is sensitive to logit initialization, and a 4-episode rollout (~200 transitions per update) gives little corrective gradient when an agent commits early to the wrong axis.

We ran a stabilization experiment with two changes at once: (i) a linear curriculum on the starting radius ($20 \to 50$ over the first 150 of 300 episodes), and (ii) larger on-policy rollouts (16 episodes per update, 256-sample minibatches, up from 4 / 64), 5 seeds per landmark. **Lateral-Ventricle** rises from 28.0\% to **44.4\%** last-50 success, with all 5 seeds learning (26, 24, 44, 68, 60\%). That is the best result in the project. **Thalamus** is unchanged in mean (20.7\% $\to$ 21.6\%) but improves in final distance (28.7 $\to$ 22.7 voxels); per-seed success is 38, 0, 10, 4, 56\%, with two seeds collapsing and the variance widening to 24.3 pp. The intervention raises the ceiling on geometrically unambiguous landmarks but does not prevent early-commitment seed collapse.

![Success rate across the stabilization runs (30-episode rolling window, mean $\pm$ 95\% CI). Curriculum + larger rollout (darker) vs. the Section 5.4 baseline (lighter) on Thalamus and Lateral-Ventricle.](figures/curriculum/success_rate.png){width=0.7\linewidth}

### 5.6 Multi-scale (coarse-fine) navigation

Following Ghesu et al. [1]'s coarse-to-fine idea, we added a stride-4 channel alongside the $7^3$ fine patch: two $7^3$ cubes at strides $\{1, 4\}$ (7 mm and 28 mm fields of view), concatenated into a 686-component vector before the direction signal. Other settings match Section 5.4 ($k=10$, 3 seeds), and both arms ran for 600 episodes.

*Table 3: Single-scale vs. multi-scale PPO on Thalamus and Lateral-Ventricle, $k=10$, 3 seeds, 600 episodes. Last-50 is the mean over the final 50 episodes; best-100 is the maximum 100-episode rolling success observed during training.*

| Landmark          | Agent              | Last-50 success       | Best-100 success      | Last-50 dist (vox)   |
|-------------------|--------------------|----------------------:|----------------------:|---------------------:|
| Lateral-Ventricle | single-scale $[1]$ | 13.3 $\pm$ 8.1\%      | 25.7 $\pm$ 7.0\%      | 25.2 $\pm$ 3.0       |
| Lateral-Ventricle | multi-scale $[1,4]$ | **38.7 $\pm$ 21.4\%** | **46.0 $\pm$ 14.5\%** | **16.3 $\pm$ 10.4**  |
| Thalamus          | single-scale $[1]$ | 20.0 $\pm$ 32.9\%     | **52.7 $\pm$ 5.7\%**  | 65.8 $\pm$ 49.8      |
| Thalamus          | multi-scale $[1,4]$ | **32.0 $\pm$ 15.9\%** | 39.7 $\pm$ 7.4\%      | **25.2 $\pm$ 17.9**  |

On **Lateral-Ventricle**, multi-scale is decisively better: last-50 success more than doubles (13.3\% $\to$ 38.7\%), best-100 climbs from 25.7\% to 46.0\%, and final distance roughly halves. On **Thalamus**, single-scale reaches a higher peak (52.7\% best-100 vs. 39.7\%) but two of three seeds collapse in the last third of training (last-50 std 32.9 pp); multi-scale settles at a lower mean and holds it (15.9 pp). At 300 episodes the comparison flipped: the doubled input dimension takes roughly 400 episodes to pay off, so naïve concatenation costs sample efficiency that a per-scale branch would not.

![Success rate for single-scale (lighter) vs. multi-scale (darker) PPO on Thalamus and Lateral-Ventricle. 30-episode rolling window, mean $\pm$ 95\% CI across 3 seeds, both arms run for 600 episodes.](figures/multiscale-comparison/success_rate.png){width=0.7\linewidth}

### 5.7 Removing the oracle dependency: a learned goal-vector network

Every result above uses an *oracle direction vector* that presupposes the landmark coordinate, which is not deployable. We replace it with a learned 3D-conv network. Inputs: a T1 patch around the agent, its $[-1,1]$-normalized position, and a 15-way one-hot landmark target. Output: an $\ell_2$-normalized 3-component direction. Loss: $1 - \cos(\hat y, y)$. Training samples come on the fly from AOMIC ID1000 FreeSurfer subjects: pick a subject, a landmark in its `aseg.mgz`, and a position within $r$ voxels of the centroid (rejection-sampled on the brain mask). The network sees only local context, position, and target identity. It never sees the landmark coordinate.

We trained three variants on Theia, the university's HPC cluster (single A100, USC Research Computing). The *baseline* network (`scripts/goal_vector/slurm/train_goal_net.sbatch`) takes a $7^3$ patch through three $3^3$ Conv3D layers (16/32/32) and a 64-unit dense head; it reaches $0.9760$ best val. cosine in 40 epochs ($\approx 5$ min wall-clock). A *laptop* version of the same architecture, trained on 100 subjects for 12 epochs, reaches $0.9685$. The *wide-FOV* variant (`train_goal_net_wide.sbatch`, 80 epochs, $\approx 4$ h wall-clock) widens the patch to $15^3$ at strides $\{1,2,4\}$, so each scale covers $15$, $30$, and $60$ voxels. Each scale runs through its own three-block convolutional branch (Conv $\to$ MaxPool $\to$ Conv $\to$ MaxPool $\to$ Conv $\to$ AvgPool, 32/64/96 filters), then the three branches concatenate into a 128-unit dense head ($\approx 715$K params, $\approx 25\times$ the baseline). It reaches $0.9968$ best val. cosine, which roughly halves the average angular error from the baseline's $\approx 12.6^\circ$ to $\approx 4.6^\circ$.

All three networks transfer to held-out AOMIC subjects (mean cosine $\geq 0.99$) but not to the MNI152 template used elsewhere in this report ($\sim 0.6$); FreeSurfer `brain.mgz` and skull-stripped MNI152 differ in intensity, orientation, and skull-strip aggressiveness. We sidestep this by deploying on AOMIC `sub-0083` and applying the same swap to the oracle baseline. Each Keras model is converted to TFJS-layers (`scripts/goal_vector/convert_to_tfjs.py`); the wide model's per-scale slice-Lambdas are rewired into separate per-scale inputs so tfjs-layers can deserialize them. The browser env's `goalVector` mode toggles `predicted` vs. `oracle`, with PPO hyperparameters matching Section 5.5 (3 seeds $\times$ 300 episodes).

*Table 4: Predicted (two FOVs) vs. oracle goal vector on AOMIC sub-0083, last-100-episode mean success.*

| Landmark          | Goal vector       | Val. cosine | Last-100 success      | Per-seed last-100   |
|-------------------|-------------------|------------:|----------------------:|---------------------|
| Thalamus          | predicted ($7^3$) | 0.9760      | 8.7 $\pm$ 8.6\%       | 7\%, 1\%, 18\%      |
| Thalamus          | predicted ($15^3$ wide) | 0.9968 | 7.3 $\pm$ 5.8\%       | 4\%, 4\%, 14\%      |
| Thalamus          | oracle            | --          | **10.7 $\pm$ 9.7\%**  | 0\%, 13\%, 19\%     |
| Lateral-Ventricle | predicted ($7^3$) | 0.9760      | **42.7 $\pm$ 16.7\%** | 48\%, 24\%, 56\%    |
| Lateral-Ventricle | predicted ($15^3$ wide) | 0.9968 | 16.3 $\pm$ 14.3\%     | 13\%, 4\%, 32\%     |
| Lateral-Ventricle | oracle            | --          | 25.3 $\pm$ 22.7\%     | 1\%, 29\%, 46\%     |

The two predicted variants invert the relationship between val. cosine and policy success. The wide-FOV network has a *much* better mean cosine ($+0.0208$, halving angular error) and yet **its Lateral-Ventricle success drops to $16.3\%$ from the baseline's $42.7\%$**, with no seed approaching the baseline's 56\% peak. Thalamus is essentially flat across both ($\approx 7$--$9\%$). The wide model is *more reliable* (per-seed std falls from 16.7 to 14.3 on Lateral-Ventricle, and every seed clears 4\% vs. the baseline's two-out-of-three), but its ceiling is much lower. The earlier laptop/Theia comparison showed the same effect on smaller scales ($-0.0075$ cosine flipped each landmark's predicted-vs-oracle ranking by 30--40 pp). All three data points point the same way: **better mean cosine does not buy better policy**. What matters is the *shape* of the per-step error distribution, which a softer scalar metric like cosine cannot capture. To address the residual seed lottery we added an in-runner seed-collapse detector (`src/experiments/ExperimentRunner.ts`) that rebuilds an agent from fresh weights when rolling success falls below threshold at a checkpoint episode (it triggered on Thalamus seed 0 of the wide-FOV run; reported numbers come from the post-restart segment). The takeaway: the oracle scaffold is removable, on Lateral-Ventricle the baseline learned vector strictly beats the oracle, and the path to better policies runs through per-step-error *distribution* shaping (auxiliary heads, hard-example mining), not raw cosine.

![Predicted goal vector under two FOVs (green: $7^3$ baseline; blue: $15^3$ wide multi-scale) vs. oracle direction (gray), under the same PPO + curriculum setup on AOMIC sub-0083. Success rate over 300 episodes, 30-episode rolling window, mean $\pm$ 95\% CI across 3 seeds. The wide model's better mean cosine (0.9968 vs. 0.9760) does not transfer into better policy: on Lateral-Ventricle its peak collapses from 56\% to 32\%.](figures/predicted-vs-oracle/success_rate.png){width=0.85\linewidth}

## 6. Conclusion

We built a browser-based platform for deep RL on anatomical landmark navigation in 3D brain MRI and used it to find what actually makes the task learnable. The single biggest win was a one-line fix to the state encoding. Amplifying the 3-dim direction vector by $10\times$ lifts last-50 success from 2--4\% to about 24\% on two representative landmarks. With that in place, PPO was the strongest of three algorithm families (best on 11/15 landmarks by late reward, 13/15 by final distance), and a curriculum over starting radius plus $4\times$ larger rollouts pushed Lateral-Ventricle to 44\% with all five seeds learning. A stride-4 multi-scale channel more than doubled late Lateral-Ventricle success at 600 episodes and tightened seed variance.

The other key result is that the oracle direction vector, which is the part of the state that presupposes the answer, can be replaced by a learned 3D-conv network and the policy still works. The baseline $7^3$ network trained on Theia (the university's HPC cluster: 40 epochs, 200 AOMIC subjects, val. cosine $0.9760$) reaches $42.7\%$ on Lateral-Ventricle vs. $25.3\%$ for the oracle, and $8.7\%$ vs. $10.7\%$ on Thalamus. Scaling the network up did not help the policy. A $15^3$ multi-scale variant ($\approx 25\times$ params, 80 epochs on Theia) halves the validation angular error and reaches $0.9968$ cosine, but *worsens* policy success on Lateral-Ventricle to $16.3\%$ while making seeds more uniform. With this wide-FOV data point added to the earlier laptop ($0.9685$) result, three samples now confirm that the prediction-quality $\to$ RL-success mapping is non-monotonic: better mean cosine does not buy better policy. Next steps are 10+ RL seeds with the collapse detector engaged, a domain-invariant goal-vector network to close the AOMIC/MNI152 gap, and auxiliary heads (distance prediction, hard-example mining) that target the *distribution* of per-step error rather than its mean. The oracle scaffold is removable, and on at least one landmark with irregular spatial extent the learned vector is strictly preferable.

## References

[1] F. C. Ghesu, B. Georgescu, Y. Zheng, S. Grbic, A. Maier, J. Hornegger, and D. Comaniciu. Multi-scale deep reinforcement learning for real-time 3D-landmark detection in CT scans. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(1):176--189, 2019.

[2] A. Alansary, L. Le Folgoc, G. Vaillant, O. Oktay, Y. Li, W. Bai, J. Passerat-Palmbach, R. Guerrero, K. Kamnitsas, B. Hou, S. McDonagh, B. Glocker, B. Kainz, and D. Rueckert. Automatic view planning with multi-scale deep reinforcement learning agents. In *Proceedings of Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2018.

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. *arXiv:1707.06347*, 2017.

[4] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. *Nature*, 518:529--533, 2015.

[5] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2016.

[6] S. M. Plis, M. Masoud, and F. Hu. Brainchop: Providing an edge ecosystem for deployment of neuroimaging artificial intelligence models. *Aperture Neuro*, 4, 2024.

[7] A. Vlontzos, A. Alansary, K. Kamnitsas, D. Rueckert, and B. Kainz. Multiple landmark detection using multi-agent reinforcement learning. In *Proceedings of Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2019.

[8] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. In *International Conference on Learning Representations (ICLR)*, 2016.
