# Latest usable release: G1 23DoF bidirectional + PAL

Approved 2026-10-18. Default supported scheme: resized human input → accelerated forward and reverse tracking → select one complete trajectory → reviewed joint PAL position/normal/reference refinement. Keep the existing tuned 23DoF configuration, contact activation and weights. This is the user-approved pipeline, not the separate adaptive-direction research branch. No Qwen component.

PAL is hand-specific; general person/robot contact constraints are not in base tracking. Bidirectional selection alleviates some local arm branches but is not a global solution. Acceleration is numerically close, not bitwise equivalent. PAL may be inactive when the existing activation condition is absent. See DUAL_PERSON_TRACKING.md for scope and validation.

## Full raw-dataset run

```bash
HUMAN_PYTHON scripts/run_full_dataset.py \
 --dataset /path/Inter-X_Dataset --models /path/Inter-X/body_models \
 --output /data/interx_23dof_release \
 --human-python HUMAN_PYTHON --omni-python OMNI_PYTHON --workers 2
```

Always selects all raw A001 handshake, A000 hugging, A016 highfive clips (`--limit 0`), including clips never previously remapped. Requires sibling dataset and human-retarget repositories. Action pipelines default to serial GPU preprocessing/remap; robot conversion uses bounded thread-pool dispatch of independent subprocesses. `--action-workers` can overlap action pipelines when GPU memory allows. Inner numerical threads are bounded by the batch runner.

Each action logs its raw contact preparation, human Laplacian remap, pair/surface export and robot+PAL conversion. Resume reuses only compatible stage caches, not archived robot results. Final acceptance.json enumerates every expected raw clip and verifies finite (T,30) 23DoF output, source-frame alignment and presence of both direction/PAL references. Missing/failed clips produce a nonzero exit. This structural audit is not visual/physical acceptance; inspect final catalog and representative motions after completion. Per-action logs survive failure; rerun the same command to resume.
