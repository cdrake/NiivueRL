# Presentation speaker notes

**Target: 10 min presentation + 3 min Q&A. 8 slides, ~75s per slide.**
Watch the clock at slide 5 (halfway).

---

## Slide 1 — Title (≈15s)

- Name, "Deep RL for Anatomical Landmark Navigation in 3D Brain MRI".
- "I'll cover the problem, three algorithms across 15 landmarks, one
  surprising practical finding around the state representation, and a
  follow-up experiment on training stabilization."

**Transition:** "Let's start with the problem."

---

## Slide 2 — Problem & MDP (≈90s)

**Lede.** "Automated landmark localization underlies atlas registration,
surgical planning, ROI volumetry. Conventional approach is either template
registration or supervised CNNs — both need dense voxel labels and
generalize poorly to atypical brains."

**The RL reframing.**
- Drop an agent at a random voxel, learn to step toward a *named* target.
- Weak supervision: only the target coordinate, no dense labels.
- Interpretable: the trajectory itself is the output.
- Open-vocabulary over labeled targets (one trained model generalizes).

**Point to figure.**
- Panel (a): agent (green), target (red star), yellow direction arrow, 7³
  neighborhood box.
- Panel (b): state = 343 voxels + 3-dim direction vector = **346 dims**.
  Actions ±x, ±y, ±z (6 discrete). Reward `−Δd − 0.1` with `+10` on success.
  200-step cap, 50-voxel starting radius.

**Platform note.** "Everything runs in the browser — TF.js + Niivue. Fully
client-side, reproducible from a static page."

**Transition:** "Three learners, plus two non-learning baselines."

---

## Slide 3 — Agents + sanity baselines (≈90s)

**Agents (each in one sentence).**
- **DQN** — flat 346-d input through dense 256→128→64, ε-greedy, replay
  buffer, target network.
- **A2C** — dense 128→64 trunk, policy + value heads, GAE(λ=0.95), entropy
  bonus.
- **PPO** — same architecture, clipped surrogate objective
  (ε_clip=0.2), 4 epochs × mb 64, rollout 4 episodes.
- Same optimizer (Adam 3e-4) across all three.

**Non-learners.**
- **Oracle** — greedy argmax on direction components. Never trains.
- **Random** — uniform over 6 actions.

**Sanity result.**
- Oracle: 100% success in ~50 steps on all 15 landmarks (matches a greedy
  ℓ∞ walk from 50 voxels away).
- Random: **0% in 500 episodes**.
- **Point:** any learner that doesn't clear random is ignoring the
  direction signal entirely. Sets the floor.

**Training.** 300 episodes × 3 seeds per config; bands in curves are
95% CI over seeds.

**Transition:** "So how do the three learners stack up?"

---

## Slide 4 — Three-algorithm comparison (≈90s)

**Headline.** "**PPO wins on 11 of 15 landmarks by late reward, 13 of 15 by
final distance.**"

**Point to curves.**
- PPO (green): stable or improving on most landmarks.
- A2C (orange): learns, but **diverges on Hippocampus and Putamen** —
  small curved structures where single-sample advantages are too noisy.
- DQN (blue): actually **regresses** as ε decays. Classic Q-value
  overestimation + catastrophic forgetting as the replay buffer fills with
  the drifting policy's own trajectories.

**Important caveat.** "Late-window success rates are still in the low
single digits. Oracle is at 100%. Where's the gap? That's the next slide."

**Transition:** "The biggest practical result of the project is a one-line
change to the state representation."

---

## Slide 5 — KEY: direction-scale k (≈120s)  ⚠ *key slide — take your time*

**Set up the problem.** "The 346-d state is 343 voxel values concatenated
with a 3-dim unit direction vector. Under L2 norm, the direction vector's
magnitude is bounded by 1. Voxel intensities occupy [0, 1] over 343
components. At initialization, the direction signal is quite literally
drowned in the voxel patch."

**The fix.** "One-line change: multiply the direction vector by a scalar k
before concatenation. That's it. I swept k ∈ {10, 30, 100} on the two
landmarks with reliable learning in the main sweep (Thalamus and
Lateral-Ventricle), 3 seeds each."

**Point to table.**
- **k=10 wins on both landmarks.** Mean last-50 success roughly 24%,
  averaged across the two — up from 2–4% with default k=1. **That's a 20×
  improvement.**
- Over-amplifying destabilizes: k=100 on Lateral-Ventricle collapses on
  all 3 seeds (the softmax saturates on one axis before the value head
  learns which states are close to success).

**One quick bullet.** "For contrast — enlarging the voxel window in the
opposite direction, from 7³ to 15³, both with flat MLP and a trainable 3D
conv trunk, also hurt performance. The direction vector, not the voxel
context, is doing the heavy lifting."

**General lesson.** "For RL practitioners: **when one component of a
concatenated state has much lower dimensionality than the rest, the default
encoding under-weights it, and the fix at init time is explicit rescaling.**"

**Transition:** "Here's the learning-curve picture."

---

## Slide 6 — Direction-scale curves (≈60s)

- Visual confirmation of the table.
- k=10 (lightest green) clearly dominates k=30 and k=100 on both
  landmarks; k=30 is second.
- Point out the k=100 Lateral-Ventricle panel (flat zero — all 3 seeds
  collapsed) vs. k=100 Thalamus (one lucky seed pulled the mean up,
  explaining the huge error band in the table).
- **But even the winner is ~25–30%, not oracle-like 100%.** There's
  headroom, and that headroom is limited by seed variance, which is the
  next thing we tried to fix.

**Transition:** "We ran a stabilization experiment."

---

## Slide 7 — Stabilization (≈90s)

**The problem.** "Across-seed standard deviation in last-50 success is
**20–30 percentage points**. This is the dominant source of uncertainty
in everything I've shown so far — it hides effect sizes on the order of
the direction-scale win."

**Hypothesis.** "Small network (128→64), softmax policy sensitive to logit
init, plus a thin rollout (4 episodes ≈ 200 transitions per update). A
seed that commits to the wrong axis early has no gradient signal to escape."

**Intervention (both at once).**
1. **Curriculum** over starting radius — begin at 20 voxels, anneal to 50
   over the first 150 episodes.
2. **Larger rollouts** — 4 → 16 episodes per update, minibatch 64 → 256.
- 5 seeds per landmark.

**Result (table).**
- **Lateral-Ventricle: 28% → 44%.** All 5 seeds learn (26, 24, 44, **68**,
  **60**). The 68% seed is the best single-seed result in the project.
- **Thalamus: unchanged** (20.7% → 21.6% mean) — but **2 of 5 seeds
  collapsed** to 0–4% success. Variance actually got worse.

**The read.** "Curriculum + larger rollouts is a **partial** stabilization.
It raises the ceiling for seeds that can bootstrap. It does **not** prevent
early wrong-axis commitment. The right fix is an online seed-collapse
detector with restart — not more hyperparameter search."

**Transition:** "To wrap up."

---

## Slide 8 — Summary + future (≈45s)

**Three takeaways (say these verbatim if you want).**
1. PPO wins on most landmarks; DQN is unstable on navigation-length
   episodes.
2. The **direction-scale trick** (k=10) is a one-line, order-of-magnitude
   fix — and suggests a general lesson for concatenative state
   representations.
3. Curriculum + bigger rollouts **partially** stabilizes training;
   wrong-axis collapse is the remaining wall.

**Future (brisk, one line each).**
1. 10+ seeds with online collapse detection and restart.
2. Cross-subject training on OpenNeuro cohorts (template → real data).
3. Architectural direction-conditioning (FiLM-style) as the principled
   replacement for the scalar-k trick.
4. Multi-agent shared representations (Vlontzos 2019) across all 15
   landmarks jointly.

**Closer.** "Happy to take questions."

---

## Q&A prep — likely questions

**Q: Why not just use supervised learning with dense labels?**
A: Requires dense annotation; doesn't transfer to new target lists. RL
needs only a target coordinate per landmark. Trajectories are also an
interpretable artifact.

**Q: Why does DQN regress?**
A: Two compounding issues. Q-value overestimation with a small target-sync
interval (100 steps) gets stale fast on navigation-length episodes (up to
200 steps). And once ε decays, the replay buffer fills with the drifting
policy's own trajectories — catastrophic forgetting. The 4% best-seed
result shows it *can* learn; it just doesn't stay learned.

**Q: Isn't multiplying by k just sensitivity to initialization?**
A: Partially — yes. But it's a *reproducible, monotone* axis: performance
is much better at k=10 than k=1 across seeds, consistently. And k=100
reliably destabilizes (not noise). So it's not just init variance, it's a
real imbalance in the default encoding.

**Q: Why 15 landmarks, not more?**
A: 15 subcortical MNI152 structures cover the representative difficulty
spectrum — large easy (Cerebellum-Cortex), small curved (Hippocampus),
tiny medial (Pallidum, 3rd Ventricle). Expanding to cortical parcels is
future work; they're structurally more ambiguous without multi-subject
data.

**Q: Does this generalize across subjects?**
A: Present work is single-template (MNI152). Infrastructure supports
multi-subject, and cross-subject training on OpenNeuro is the clearest
next experiment.

**Q: Why browser / TF.js?**
A: Reproducibility (anyone can run the full pipeline from a static URL, no
local setup), and Niivue is a first-class browser viewer. Cost is slower
training than native GPU, which limits seeds and episode counts — a known
trade-off that shows up in the 20–30 pp seed variance.

**Q: Is the reward shaping problematic?**
A: It's dense and potential-based (`−Δd` is the difference in distance),
so the optimal policy under the shaped reward is the same as under a
sparse success-only reward. The `−0.1` step penalty only discourages
wandering. Oracle achieving 100% confirms the shaping doesn't create
distracting local optima.
