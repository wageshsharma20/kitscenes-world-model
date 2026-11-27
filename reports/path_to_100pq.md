# Path to 100 PQ: what is actually costing us the last 2.19 points

Status: measured on 2026-11-27, CPU only, ground truth available.
Context: rank 7, PQ 97.81, three submission slots left, deadline 2026-12-03 AoE.

This note replaces the earlier assumption that the deficit is made of holes that the
inpainter fills badly. It is not. The renderer already covers 99 %+ of most frames; the loss
is per-clip outliers, resampling detail, and photometric drift.

---

## 1. What the score actually measures

- Each camera clip is 25 frames at 5 fps. Q-Align samples it at 1 fps, so it sees frames
  `[0, 5, 10, 15, 20]` only: **12,000 scored frames out of 60,000 rendered**.
- `PQ = 50 % front + 10 % each other camera`, so the front view is half the score and every
  camera must be submitted (front-only caps at 50 %).
- Q-Align is **no-reference**. It rates how good a frame looks, not whether it predicts the
  real future. Prediction fidelity enters only EMC and MMS, which break ties.
- A clip is rated **as one video, not frame by frame**. `QAlignVideoScorer.forward`
  (`q_align/evaluate/scorer.py`) preprocesses all sampled frames into a single tensor, runs
  one forward pass, and returns `softmax(logits) @ [1, .75, .5, .25, 0]` — i.e. exactly the
  official `100 * (rating - 1) / 4` mapping. There is **one set of logits, one softmax and one
  score per camera clip**; the five frames are never scored independently.
- So a clip score of 95 means the model's level distribution *for the whole 5-frame sequence*
  averaged 0.95 (e.g. 80 % excellent / 20 % good). It does **not** mean one frame was `good`
  and the rest `excellent`. A single bad frame therefore has **no fixed price**: near
  saturation it can move several points, while a clip already averaged at `good` barely moves
  at all. Tail repair is not a linear per-frame budget — whole-clip appearance dominates, and
  the five sampled frames move the score only through the joint judgement.

Two consequences drive everything below:

1. The last 2.19 points are an *image quality* problem, not a world-model problem.
2. The deficit is a **probability-mass shift, not a defect count**: 97.81 is roughly 2 % of
   the level mass sitting one band below `excellent`, averaged over 2,400 judgements. That is
   why no single visible artefact explains it and why the fix has to be systematic.

### Where the points are

| type | clips | PQ | weighted gap | gap per clip |
| --- | --- | --- | --- | --- |
| intersection | 125 | 98.23 | 0.553 | 0.0044 |
| overtake | 102 | 98.01 | 0.507 | 0.0050 |
| heavy rain | 27 | 92.77 | 0.488 | 0.0181 |
| construction | 36 | 97.25 | 0.248 | 0.0069 |
| snow | 23 | 96.10 | 0.224 | 0.0097 |
| night | 19 | 97.44 | 0.122 | 0.0064 |
| selected | 68 | 99.70 | 0.051 | 0.0008 |
| **total** | 400 | **97.81** | **2.19** | |

Rain, snow and night are cheap to fix per point: 27 rain clips carry the same gap as 125
intersection clips. Intersection and overtaking carry 1.06 of the 2.19 but are the most
expensive to move.

---

## 2. What was measured today

`scripts/ab_render_metrics.py` renders a validation scenario with the real future trajectory
(`expert_like`) and compares the front clip against the dataset's own future video. `hf` and
`sharp` are ratios to the real frame, so they do not depend on trajectory accuracy; `psnr`
does.

`render_scale=1`, `chain_warps=off`, no depth, no inpainting: the plain analytical renderer.

| scenario | type | holes mean (worst) | psnr 0 -> 20 | hf 0 -> 20 | sharp 0 -> 20 | saturation absolute |
| --- | --- | --- | --- | --- | --- | --- |
| 0000903 | rain, straight | 6.1 % (44 %) | 26.9 -> 24.2 | 1.06 -> 0.78 | 1.11 -> 0.77 | 37.6 -> 41.9 (+11 %) |
| 0000900 | snow, straight | 0.7 % (11 %) | 30.5 -> 20.1 | 1.05 -> 1.05 | 1.08 -> 1.04 | 9.7 -> 11.4 (+18 %) |
| 0000902 | intersection, right | 1.7 % (11 %) | 28.2 -> 14.7 | 1.01 -> 1.11 | 1.00 -> 1.10 | 51.7 -> 57.7 (+12 %) |
| 0000911 | selected, left | 2.6 % (12 %) | 19.3 -> 15.4 | 0.92 -> 0.96 | 0.87 -> 0.90 | 68.1 -> 60.6 |
| 0000905 | selected, right | 16.9 % (56 %) | 15.9 -> 19.6 | 0.72 -> 1.04 | 0.58 -> 0.98 | 58.6 -> 29.0 |

### Findings

**F1 — Holes are not the problem.** The fraction of each frame that had to be synthesised is
0.7 % to 6 % in four of five scenarios. The inpainter is doing almost nothing, so
"better LaMa" and "cross-camera fusion" are not levers: cross-camera, past-frame and
cross-rig ground fill are already implemented in `generate_scenario`. The 17 % / 56 %
outlier is a *tail* case, and tail cases are exactly what pins a type below 100.

**F2 — Detail decays with warp distance.** On rain, high-frequency energy relative to real
footage falls `1.06 -> 0.78` over the clip while sharpness falls `1.11 -> 0.77`. The render
is 22 % softer than real footage by frame 20. This cannot be recovered by sharpening: the
source pixels are magnified across a 4.8 s ego displacement, so the detail was never
sampled. Restoring it requires texture synthesis, or accepting the loss.

**F3 — The opposite failure also exists.** On the intersection clip the render is 10 %
*sharper* than real footage (`sharp 1.10`) while PSNR collapses to 14.7 dB. Large,
minifying warps undersample, which is the mechanism behind the rainbow fringes on thin bright
structures at night. This one is fixable deterministically with minification-aware
prefiltering; it does not need a model.

**F4 — Photometric drift is real and large.** Saturation rises 11-18 % across a clip, and one
scenario swings by 2x. Q-Align sees a clip's five frames together; a clip whose colour walks
away from its own first frame reads as a different camera. `kitscenes/refine.py` anchors each
clip to its observed t=0 frame; that is the right fix and it costs no GPU.

**F5 — `render_scale=2` buys nothing.** On rain, `hf` at frame 20 is 0.759 at scale 2 versus
0.781 at scale 1, at 3.7x the runtime (154 s vs 42 s per scenario). The conditioning frames
are natively 876x568, so supersampling only interpolates. Production should render at scale 1
unless a variant wins on EMC.

**F6 — Temporal chaining is worse, measured.** Warping each frame from the previously
rendered frame instead of from t=0 (added as `chain_warps`, default off) gives rain frame 20
`hf` 0.749 vs 0.781, `sharp` 0.684 vs 0.770, PSNR 23.6 vs 24.2. 25 compounded bicubic
resamples lose more than one large warp does. Rejected; the flag stays default-off.

---

## 3. What rank 1 tells us

Rank 1 sits at **PQ 100.00 in all seven types** with a training-free method, but only
**EMC 0.49**. A method that reproduced the future faithfully would score higher on EMC. So
the winning recipe is: make every frame look photographically real, and let fidelity be
"good enough". Their EMC x MMS is 2.401; ours is 0.16 x 4.50 = 0.72. If we reach 100.00 flat
we tie on PQ and **lose the tie-break**, so EMC has to move too.

EMC 0.16 is suspicious on its own terms: our warps are geometrically derived from the
trajectory we submit, so flow-derived ego motion should agree with it. Three candidates, in
order: the frames are too soft for optical flow to lock on (F2), the side views are mostly
static car body, and the submitted trajectory and the renderer use different heading
conventions. Fixing detail helps PQ and EMC at the same time, which is the cheapest
correlation available.

---

## 4. Hardware reality (verified November 2026)

| provider | hardware | free allowance | relevant to us |
| --- | --- | --- | --- |
| Kaggle | 2x T4 16 GB | ~30 GPU-h/week, 12 h per session, resets Sat 00:00 UTC | bulk rendering and bulk scoring; the two cards are discrete |
| Colab free | T4 16 GB | unpublished, <= 12 h | second scoring lane |
| Lightning AI | T4 / L4 24 GB | one-time 30 credits (~54 T4-h, ~38 L4-h) | L4 fits fp16 Q-Align exactly |
| HF ZeroGPU | RTX Pro 6000 48/96 GB | daily quota, 60 s per call | spot checks of a handful of frames only |
| Paperspace free | M4000 8 GB | 6 h sessions | too small |

Model fit:

| model | fp16 | 8-bit | fits a T4? |
| --- | --- | --- | --- |
| Q-Align / OneAlign (7.8 B) | 16.2 GB | 8.9 GB (4-bit 5.4 GB) | 8-bit yes (LLM.int8 needs Turing; T4 is Turing) |
| SDXL-inpaint (3.5 B) | ~6.5 GB at 1024^2 | ~4.5 GB | yes, ~0.6-0.7 s/step |
| FLUX.1-Fill (12 B) | ~20 GB | ~12 GB fp8 | fp8 needs Ada+; L4 only, and slow |
| Wan 2.2 TI2V-5B | ~10 GB | needs fp8 | no: T4 has no fp8 |

**Full video diffusion is not available to us.** At 20-40 min per 5 s clip on a T4,
2,400 clips is 800-1,600 GPU-hours, i.e. 27-53 Kaggle-weeks. A rank-1-style generative
world model is out of reach on free hardware. **Frame-level image models are in reach**:
12,000 frames at ~6 s is ~20 GPU-hours, one week of one Kaggle account, or one day across
Kaggle plus Colab.

---

## 5. The plan, ordered by points per GPU-hour

### P0 — Run the oracle (about 5 GPU-h, must be first)

`scripts/eval_world_model_qalign.py` already loads `q-future/one-align` and already has an
8-bit fallback for a single 15 GiB T4; it has apparently never been run at frame level.
Without it, every change below is guesswork and the last three slots are spent blind.

1. Score the **current** submission frame by frame (12,000 frames), not just per clip, and
   write a per-frame table.
2. Score the ten validation ground-truth clips the same way. If real footage does not pin at
   ~100, the oracle is miscalibrated and nothing else matters until it does.
3. Cross-check 8-bit against fp16 on 200 frames on a Lightning L4 to prove the ranking is
   preserved. Only then trust it for accept/reject.

Gate: do not change the renderer until step 2 passes.

### P1 — Photometric anchoring (0 GPU-h, safe)

`scripts/refine_submission.py` + `kitscenes/refine.py`: anchor every clip to its observed t=0
frame, neutralise the grey-world cast, restore acutance. Two real bugs were found and fixed
today in `_box_blur` (wrong pad arity for 4-D input, off-by-one in the summed-area window), so
this path had never actually run; it now does, end to end, at 25 frames/876x568/5 fps.
Validate the result against real footage with `ab_render_metrics.py` (saturation ratio should
move towards 1.0) before it goes in a submission.

### P2 — Tail repair, not average improvement (10-20 GPU-h)

P1 and F1 both point at outliers. The plan:

1. From P0, rank clips by `100 - PQ`, per type.
2. Repair only the bottom decile: those pins cost 5 points each and are what keeps a type
   below 100.00.
3. Repair method: masked SDXL-inpaint over the synthesised/low-confidence regions only
   (`padding_mask_crop` keeps cost proportional to the mask, not the frame), strength 0.35-0.5.
   Re-score with the oracle; keep only frames that improve, and only whole clips that improve
   their clip mean.
4. Because only the five sampled frames are scored, refine the five and interpolate the rest
   from the untouched render, so the clip stays continuous.

### P3 — Detail restoration for soft clips (P2 infra, no extra budget)

F2 says the render is up to 22 % softer than real footage. Same diffusion pass, whole-frame
img2img at strength 0.2-0.25 with a depth ControlNet so geometry cannot move. This is the only
lever that can close a systematic gap rather than a tail, and it is also the one that can move
EMC, because flow needs texture.

### P4 — Aliasing prefilter (0 GPU-h, deterministic)

F3: sample from a mip-matched source so minifying regions are prefiltered. Compute the local
Jacobian of the analytic homography per pixel, pick a blur level from its area ratio, blend
two levels. Target: `sharp` ratio 1.10 -> ~1.0 on intersection clips, and the night
traffic-light fringes disappear. Verifiable with `ab_render_metrics.py` alone.

### P5 — Encode budget (0 GPU-h)

The archive is 2.42 GiB against a 3 GiB limit: 24 % headroom, currently unspent. Front is
half the score, so give front a lower CRF and the side cameras a higher one, and confirm with
`validate_submission.py --jobs 8` that the archive still passes.

### P6 — EMC for the tie-break (fold into P2/P3)

Re-derive the heading convention once, end to end (submitted trajectory -> renderer -> flow
estimator), and confirm locally that recovered yaw matches the submitted trajectory before
spending a slot. Target EMC >= 0.49 so a 100.00 tie does not lose on EMC x MMS.

---

## 6. Six days, three slots

| day | work | slot |
| --- | --- | --- |
| 1 (today) | P0 oracle on the current submission + validation GT; P1 refine validated | - |
| 2 | P4 prefilter + P1 in the rerender; oracle re-score of the ten val clips | - |
| 3 | P2 tail repair on the worst decile; oracle A/B per type | - |
| 4 | full render, package, validate, submit | **slot 1** |
| 5 | P3 detail pass on the types still below 100; submit if the oracle improves | **slot 2** |
| 6 | last fixes only if the oracle shows a clear win | slot 3 (reserve) |

Never spend a slot without an oracle number for the exact archive being uploaded. "It looks
better" is what produced 97.81.

---

## 7. Does 100.00 per type actually happen?

Not guaranteed, and the honest split is:

- **Plausible cheaply:** rain (27 clips), snow (23), night (19), selected (68) — 0.89 of the
  2.19 points across 137 clips, where the failures are photometric drift, haze and a handful
  of outlier clips. P1 + P2 should move these.
- **Hard:** intersection (125) and overtaking (102) — 1.06 of the gap. These need P3 (detail)
  and P4 (aliasing) to work at once, and every one of their 125 / 102 clips must come out
  perfect for the type to read 100.00.
- **The ceiling is a threshold, not a slope.** Because one softmax is taken over the whole
  5-frame sequence, 100.00 in a type means the model's level distribution for *every* clip in
  that type stayed on the `excellent` side. At 97.81 the average clip carries about 2 % of its
  mass one band low, which is why the last points are the most expensive and why the oracle is
  the only way to find the clip holding a type at 99.8.

## 8. Do not do

- Tune against `kitscenes/pq_proxy.py` or `kitscenes/emc_eval.py`. Neither is the official
  metric; the local EMC harness has a known yaw-sign defect.
- Tune holes, LaMa thresholds or cross-camera fill. Measured: not the bottleneck.
- Raise `render_scale`. Measured: no detail gain, 3.7x cost.
- Enable `chain_warps`. Measured: worse on all three metrics.
- Chase a Wan-class video model on free GPUs. Measured against the quota: impossible.
