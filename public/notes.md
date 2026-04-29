# NiivueRL — Experiment Notes

A running log of experiments. Each entry: what we tried, what happened, what we learned.

## Oracle baseline

Sanity check that the environment is sound. Oracle policy (move toward target each step) hits roughly 100% success; a uniform-random policy hits roughly 0%. Confirms reward shaping, target-reach detection, and start-radius sampling all behave as intended.

→ [run](#/run/oracle-baseline)

## DQN (the original agent)

DQN with a flat MLP over the 7×7×7 patch + position + direction-to-target was the project's first agent. It learned the toy 2-landmark setup but generalized poorly to a wider landmark set: replay-buffer staleness and Q-value overestimation made training unstable across seeds.

## A2C (synchronous, single-worker)

We swapped DQN for A2C as a stepping stone toward PPO. Convergence was faster but variance across seeds was huge — most seeds either took off or collapsed early. The agent was also brittle to the direction signal magnitude.

## Direction-scale diagnostic

Because PPO was performing near random, we hypothesized the goal-vector signal was being washed out by the patch features. We added a `dirScale` knob that multiplies the unit direction-to-target. Scaling 1× → 10× turned random-looking learning curves into clean monotonic improvement.

→ [diagnostic](#/run/dirscale-diag) · [sweep 10/30/100](#/run/dirscale-sweep)

## Multi-scale observation

We tried giving PPO two nested patches (stride 1 and stride 4) so it could see both fine local structure and coarse context. At matched compute budget (600 episodes), single-scale `strides=[1]` matched or beat multi-scale on the landmarks we tested — the extra channels increased input size faster than they paid back in sample efficiency.

→ [multi-scale](#/run/multiscale) · [single-scale control](#/run/singlescale-600)

## Curriculum over starting radius

The big stability win. Starting episodes near the target (radius 20) and annealing out to 50 over 150 episodes essentially eliminated the "dead seed" failure mode. Late-stage performance also improves because earlier success episodes have populated the rollout buffer with informative trajectories before harder starts begin.

→ [PPO + curriculum (5 seeds)](#/run/curriculum-5seed)

## Goal-vector network (deploy-time signal)

The clinically realistic version of the agent: the env's exact direction-to-target is replaced with the prediction of a small CNN that takes a local T1 patch, position, and target one-hot and outputs a unit vector. We trained the CNN on AOMIC ID1000 FreeSurfer parcellations on Theia HPC; held-out validation cosine similarity is 0.9760.

End-to-end: PPO with the same curriculum that learned with the oracle signal also learns with the predicted signal, with a modest gap in final success rate but otherwise comparable training dynamics.

→ [predicted](#/run/predicted-vs-oracle) · [oracle baseline (matched config)](#/run/oracle-aomic)

## Long-run asymptote

A 1000-episode run at `dirScale=10` to see where PPO plateaus. Useful sanity check that we hadn't been calling things "converged" too early at 300 episodes.

→ [long run](#/run/long-run)
