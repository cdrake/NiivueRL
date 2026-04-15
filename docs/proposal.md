**CSCE 775: Deep Reinforcement Learning and Search**

# Deep Reinforcement Learning for Anatomical Landmark Navigation in 3D Brain MRI

**Chris Drake**
**February 24, 2026**

## Introduction

Finding anatomical landmarks in brain MRI is important for many neuroimaging tasks, including atlas registration, surgical planning, and volumetric analysis. Traditional approaches use template-based atlas registration or supervised CNNs trained on large, manually annotated datasets. Both require a lot of labeling effort and can generalize poorly to atypical or pathological brains.

This project frames landmark detection as a **navigation problem**: a deep RL agent starts at a random location inside a 3D brain volume and has to learn to navigate to a specified anatomical target. This is appealing because it does not require voxel-level segmentation labels, only target coordinates, and the agent learns an implicit spatial model of brain anatomy through interaction. We target 15 subcortical structures (e.g., thalamus, hippocampus, ventricles, brain-stem) that span a wide range of sizes, depths, and MRI contrast levels within a standard MNI152 T1-weighted volume. The central question is: **how do different deep RL algorithms and environment design choices affect the agent's ability to reliably locate these landmarks?**

## Approach

We model the problem as an episodic MDP with a discrete action space (Figure 1). At each step, the agent observes a local neighborhood of voxel intensities (e.g., 7x7x7 = 343 values) around its current position, plus a 3-component direction vector toward the target. It picks one of 6 actions (move ±1 voxel along each spatial axis). An episode ends when the agent gets within 3 voxels of the target (success) or exceeds 200 steps (failure). Rewards combine a dense distance-based signal, a step penalty, and a success bonus.

We compare three algorithms: **DQN** [1], using a feedforward Q-network with experience replay and a target network; **A2C** [2], an actor-critic method trained with generalized advantage estimation (GAE) and entropy regularization; and **PPO** [3], which extends A2C with a clipped surrogate objective for more stable policy updates. The entire system runs in-browser using TensorFlow.js, and we use a pretrained MeshNet model from Brainchop [6] for subject-specific brain parcellation to define landmark regions.

The closest prior work is Ghesu et al. [4] and Alansary et al. [5], both of which formulate anatomical landmark detection as RL navigation with 6 discrete movement actions and a distance-based reward. Ghesu et al. [4] use DQN on 3D CT scans with a fixed-scale approach: the agent observes a fixed-size image patch centered at its position and a frame history buffer of the last 4 steps to reduce oscillation. Alansary et al. [5] extend this with a multi-scale coarse-to-fine strategy for MRI view planning, where the agent starts with large step sizes (9 voxels) and coarse image spacing, then progressively refines to single-voxel steps across 3 scales. Both works evaluate only DQN variants (DQN, Double DQN, Dueling DQN) and find that no single architecture consistently wins across landmarks.

Our approach shares the same basic formulation (6 actions, distance-based reward) but differs in several ways. First, we compare across algorithm families: off-policy value-based (DQN) versus on-policy actor-critic (A2C, PPO), which neither prior work does. Second, rather than using a frame history buffer, we provide the agent with an explicit direction-to-target vector as part of the state, and we ablate this choice to measure its effect. Third, we use a single-scale approach but systematically vary the neighborhood size (3x3x3 to 7x7x7), whereas Ghesu et al. and Alansary et al. fix the patch size and vary the image spacing. Finally, we ablate the reward function itself (sparse vs. dense vs. hybrid), which prior work treats as fixed.

## Evaluation

We evaluate each configuration on all 15 landmarks using 5 random seeds per condition. The main metrics are **success rate** (fraction of episodes reaching within 3 voxels of the target), **path efficiency** (ratio of straight-line distance to actual steps taken), and **sample efficiency** (episodes needed to reach 80% success rate). We also report mean episode reward and final Euclidean distance to the target.

The experiments are structured in stages to keep the total number of runs manageable. First, we compare the three algorithms with a fixed hybrid reward and 7x7x7 state (3 × 15 × 5 = 225 runs). Next, we fix the best-performing algorithm and ablate reward formulation (sparse, dense, hybrid) and state representation (3x3x3, 5x5x5, 7x7x7, with and without direction vector). Finally, we aggregate results to rank landmarks by difficulty and look at correlations with anatomical properties like structure volume and MRI contrast. Baselines include a random-walk agent and a greedy agent that always steps toward the target.

## Conclusion

We propose a systematic study of deep RL for anatomical landmark navigation in 3D brain MRI. By comparing DQN, A2C, and PPO under controlled variations in reward shaping and state representation across 15 subcortical landmarks, we want to find out which algorithmic and design choices matter most for navigation performance. The project builds on an existing browser-based environment (NiivueRL) with working DQN and A2C agents. The remaining work is to implement PPO, build an experiment runner for batch evaluation, and run the proposed ablation studies over the next seven weeks.

## References

[1] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.

[2] V. Mnih et al., "Asynchronous methods for deep reinforcement learning," in *Proc. ICML*, 2016.

[3] J. Schulman et al., "Proximal policy optimization algorithms," *arXiv:1707.06347*, 2017.

[4] F. C. Ghesu et al., "Multi-scale deep reinforcement learning for real-time 3D-landmark detection in CT scans," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 41, no. 1, pp. 176–189, 2019.

[5] A. Alansary et al., "Automatic view planning with multi-scale deep reinforcement learning agents," in *Proc. MICCAI*, 2018.

[6] S. M. Plis, M. Masoud, and F. Hu, "Brainchop: Providing an edge ecosystem for deployment of neuroimaging artificial intelligence models," *Aperture Neuro*, vol. 4, 2024.
