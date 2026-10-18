# Dual-person data pipeline with legacy tuned G1 23-DoF

Branch is based on official Holosoma d1e8499c10670f44c49d161b9876a4c17ac305f7. The legacy user G1 23-DoF configuration/mappings and paired-data adapter are retained. Core costs/constraints are unchanged; numerical compatibility, acceleration and whole-trajectory direction selection are described below. New semantic region tracking is not used.

`config_types/robot.py` and `data_type.py` retain the previous local `g1-23dof` configuration (c1a289f), with the model path made repository-relative. `models/g1_23` contains the matching existing Unitree rev_1_0 model. No 29-DoF fallback.

Input NPZ: `actor_joints`, `partner_joints` each (T,22,3), `source_frame` (T,), `fps=30`. Both are already resized/grounded Z-up world metres. Actor is the small person. Preserve partner unmodified; it is not supplied to the solver objective, collision handling or interaction graph. No additional inter-person contact optimization is introduced. The upstream ground/foot/joint-limit behavior remains active.

```bash
python scripts/dual_person_tracking.py --pair /path/pair.npz --output outputs/robot.npz
```

Use `--prepare-only` for model/shape validation and `--max-frames 3` for a short solver smoke run. Upstream dependencies are installed from `src/holosoma_retargeting/pyproject.toml`. The workspace `hri/scripts/export_pair_for_omni.py` exports the latest human-remap result without repeating size or ground normalization.

Validation: G002T003A016R009 full 109-frame run completed (~8.6 s solver loop); output `(109,30)` finite qpos, exact partner-joint and source-frame preservation. At that initial branch validation, solver source was byte-identical to the official baseline. This is an execution/alignment test, not a visual or physical acceptance claim.

## PAL: joint palm position and normal refinement

The baseline includes the legacy 23-DoF free-joint limit indexing and hinge
Jacobian compatibility fixes. The optional PAL stage jointly optimizes source
palm position, outward normal, reference-arm preservation and correction
continuity. Only shoulder/elbow/wrist joints change; the partner and robot base
and legs stay fixed. The position target preserves the current SMPL palm's world
position, including its relationship to the partner; it does not force zero gap.
This hand-specific stage is not a general region-contact or collision solver.

`configs/palm_frames_v1.json` contains the user-reviewed left/right human
triangle+barycentric points and robot body-local points/normals. Human surface
normals follow triangle winding and the saved sign. No proximity-based palm
surface selection is used for the orientation target. Activation still uses
5-cm hand proximity with a 15-frame transition envelope.

Run the existing paired-input CLI with `--palm-motion surface.npz` to enable PAL.
The surface NPZ must contain synchronized `vertices (T,2,V,3)`, `joints` and
`faces` using the same SMPL-X topology as the approved configuration. Output
includes `qpos_reference`, `palm_angles_deg` and `palm_seconds`.

Acceleration switches (enabled by the paired CLI; set to 0 to disable):
- `OMNI_COMPACT_LAPLACIAN=1`: algebraically eliminates auxiliary Laplacian variables and their equality constraints; retains all original objective terms, limits and SOC constraints.
- `OMNI_PAL_GPU=1`: batches PAL hand-proximity distances in CUDA float64, using direct Euclidean differences. MuJoCo kinematics and Clarabel/SciPy optimization remain CPU. CUDA initialization can dominate short clips; benchmark before enabling for a workload.

The paired CLI enables `OMNI_REUSE_PROBLEM=1`: direct Clarabel quadratic/SOC
assembly bypasses repeated CVXPY canonicalization. All affine inequalities,
quadratic terms, SOC radius and iteration counts are retained. Failed direct
solves fall back to the original CVXPY path. Set the flag to 0 for comparison.
Validated on 109- and 115-frame clips; maximum qpos differences against saved
references were 3.6e-6 and 3.9e-5 respectively, not bitwise equivalence.

## Dataset batches

Run `python scripts/batch_dual_tracking.py --manifest jobs.json --output results --workers 2 --threads 1` in the Omni Python environment.
Manifest paths are relative to the manifest file:
```json
{"clips":[{"id":"clip-id","pair":"pairs/clip-id.npz","palm_motion":"surfaces/clip-id.npz"}]}
```
Omit `palm_motion` for baseline-only tracking. Every surface must use synchronized
approved SMPL-X topology. Independent clips run concurrently; the temporal solve
order inside each clip is unchanged. PAL distance computation uses CUDA when
available; the constrained tracking solver remains CPU. More workers increase
both CPU usage and per-process CUDA memory, so benchmark before large runs.
Each clip has a log and atomic published output. Restarting skips completed
outputs only if input hashes and relevant code/config hashes match. Failed clips
are recorded in `batch_report.json` and do not stop the remaining jobs. Limits on
BLAS/OpenMP threads prevent workers from oversubscribing CPU cores.

## Complete multi-repository pipeline (starts from raw Inter-X)

Place `hri-dataset` at sibling `dataset`, `smpl_hri_retarget` at sibling
`human-retarget`, and this checkout at `omni-retarget`. Install each repository's
requirements/bootstrap assets. Dataset and SMPL-X model files stay external.

```bash
HUMAN_PYTHON tools/workspace_pipeline.py \
  --dataset /path/Inter-X_Dataset --models /path/body_models \
  --config ../human-retarget/configs/laplacian.json \
  --output /path/results --actions highfive handshake --limit 10 --workers 4 \
  --human-python HUMAN_PYTHON --omni-python OMNI_PYTHON
OMNI_PYTHON tools/catalog_server.py --root /path/results --port 8771
```

Default path: original Inter-X P1/P2 -> dataset shared grounding and semantic
contact preprocessing -> released human local-Laplacian remap -> synchronized
pair/surface export -> accelerated robot tracking + PAL -> catalog. Use
`--clip ID --actions ACTION --limit 1` for a smoke test. `--limit 0` supports all
clips of the selected actions. `--from-human-manifest` is an explicit partial-pipeline
shortcut, not the default. Failures retain logs and nonzero exit status.

The browser catalog filters clip ID, action and completion status, and launches
the existing MuJoCo style of three-panel synchronized native review. This is a
browser management page plus a native interactive viewer, not a browser renderer.
Models can rotate/pan/zoom, and robot panels overlay the small-human source.
The local catalog binds loopback only. No dataset or model assets are uploaded.

## Whole-trajectory solve direction (2026-10-18)

The paired CLI now defaults to `--trajectory-solve bidirectional`. The forward
SQP solve can enter a shoulder branch that subsequently saturates the joint
limits, even when a much better feasible trajectory exists. This was reproduced
with both the original CVXPY and accelerated paths; no incorrect Jacobian or
input export was found. It is a history-dependent local solution, not a proven
arithmetic/indexing defect.

Run the unchanged solver independently in both time directions, align both
outputs to source order, and evaluate the existing nonlinear Laplacian, posture,
and temporal smoothness objective over the complete clip. A backward candidate
is eligible only if joint limits, original forward foot-sticking edges and ground
penetration are no worse than the forward candidate within numerical tolerances.
Choose one whole trajectory, never splice frames. No torso target, additional
penalty, joint restriction or contact rule is added. This comparison is for the
current robot-only paired path (no nominal target or moving object).

`--trajectory-solve forward` reproduces the prior single-direction path.
`qpos_forward` preserves that result alongside the chosen trajectory;
`motion.json` records candidate scores, constraint checks and selection. PAL
runs on the selected trajectory afterwards. Native review adds an original
forward column when this diagnostic array is present. Cost: roughly two tracking
passes plus objective/constraint evaluation, not the previous single-pass timing.

Regression: G002T003A016R009 frame78 right-wrist position error 197.4 -> 20.1mm,
whole-clip maximum44.3mm, objective60.77 ->27.28, locked foot XY ~1mm and no
joint-limit violation. Maximum adjacent joint change0.353rad. The original
nonaccelerated path selects the same candidate (max qpos difference2.8e-5).
G001T003A016R005 retains the forward candidate (objective22.32 vs22.67).
These are two-clip numeric checks, not acceptance of the whole dataset.
