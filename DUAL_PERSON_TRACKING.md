# Dual-person data pipeline with legacy tuned G1 23-DoF

Branch is based on official Holosoma d1e8499c10670f44c49d161b9876a4c17ac305f7. Only legacy user G1 23-DoF configuration/mappings and a paired-data adapter are added. The upstream solver and its default costs/constraints are unchanged. New semantic region tracking is not used.

`config_types/robot.py` and `data_type.py` retain the previous local `g1-23dof` configuration (c1a289f), with the model path made repository-relative. `models/g1_23` contains the matching existing Unitree rev_1_0 model. No 29-DoF fallback.

Input NPZ: `actor_joints`, `partner_joints` each (T,22,3), `source_frame` (T,), `fps=30`. Both are already resized/grounded Z-up world metres. Actor is the small person. Preserve partner unmodified; it is not supplied to the solver objective, collision handling or interaction graph. No additional inter-person contact optimization is introduced. The upstream ground/foot/joint-limit behavior remains active.

```bash
python scripts/dual_person_tracking.py --pair /path/pair.npz --output outputs/robot.npz
```

Use `--prepare-only` for model/shape validation and `--max-frames 3` for a short solver smoke run. Upstream dependencies are installed from `src/holosoma_retargeting/pyproject.toml`. The workspace `hri/scripts/export_pair_for_omni.py` exports the latest human-remap result without repeating size or ground normalization.

Validation: G002T003A016R009 full 109-frame run completed (~8.6 s solver loop); output `(109,30)` finite qpos, exact partner-joint and source-frame preservation. Solver source is byte-identical to official baseline. This is an execution/alignment test, not a visual or physical acceptance claim.

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
