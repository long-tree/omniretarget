"""Track only the resized actor with upstream solver and legacy 23-DoF config."""

import argparse, json, sys, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = Path(os.environ["OMNI_LEGACY_PACKAGE"]) if os.environ.get("OMNI_LEGACY_PACKAGE") else ROOT / "src/holosoma_retargeting/holosoma_retargeting"
sys.path.insert(0, str(PKG.parent))


def load_pair(path):
    with np.load(path, allow_pickle=False) as z:
        actor = np.asarray(z["actor_joints"], float)
        partner = np.asarray(z["partner_joints"], float)
        fps = float(z["fps"])
        frames = np.asarray(z["source_frame"])
    if actor.ndim != 3 or actor.shape[1:] != (22, 3) or partner.shape != actor.shape or len(frames) != len(actor):
        raise ValueError("Expected synchronized (T,22,3) paired joints")
    if not np.isfinite(actor).all() or not np.isfinite(partner).all() or fps != 30:
        raise ValueError("Finite Z-up metres at 30 fps required; resample explicitly")
    return actor, partner, fps, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--palm-motion", type=Path, help="Current SMPL surface NPZ: vertices, joints, faces; enables joint PAL solve")
    ap.add_argument("--trajectory-solve", choices=["forward", "bidirectional"], default="bidirectional",
                    help="Compare complete forward/backward solutions under the unchanged objective")
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    # Validated equivalent fast path; users can explicitly set either flag to 0.
    os.environ.setdefault('OMNI_COMPACT_LAPLACIAN','1')
    os.environ.setdefault('OMNI_REUSE_PROBLEM','1')
    if a.palm_motion:
        import torch
        os.environ.setdefault('OMNI_PAL_GPU','1' if torch.cuda.is_available() else '0')
    actor, partner, fps, frames = load_pair(a.pair)
    if a.max_frames:
        actor, partner, frames = actor[: a.max_frames], partner[: a.max_frames], frames[: a.max_frames]
    a.output = a.output.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.data_type import MotionDataConfig

    robot = RobotConfig()
    motion = MotionDataConfig(data_format="smplx", robot_type="g1")
    import mujoco

    model = mujoco.MjModel.from_xml_path(robot.ROBOT_URDF_FILE.replace(".urdf", ".xml"))
    assert model.nq == 30 and robot.ROBOT_DOF == 23
    assert all(model.body(b).id >= 0 for b in motion.resolved_joints_mapping.values())
    if a.prepare_only:
        np.savez_compressed(a.output, actor_joints=actor, partner_joints=partner, fps=fps, source_frame=frames)
        print("23-DoF mapping and paired input validated")
        return
    from holosoma_retargeting.examples.robot_retarget import (
        create_task_constants,
        setup_object_data,
        build_retargeter_kwargs_from_config,
        initialize_robot_pose,
    )
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity

    # Upstream relative assets are resolved in package directory; no shared file is modified.
    os.chdir(PKG)
    task = TaskConfig()
    constants = create_task_constants(robot, motion, task, "robot_only")
    pts, demo, urdf = setup_object_data("robot_only", constants, None, 1.0, task, False)
    retargeter = InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(RetargeterConfig(), constants, urdf, "robot_only")
    )
    # Already sized/grounded upstream. Do not scale or independently ground either person again.
    object_poses = np.tile([1.0, 0, 0, 0, 0, 0, 0], (len(actor), 1))
    q, nom, aug, actor, object_poses = initialize_robot_pose(
        "robot_only",
        "smplx",
        actor,
        object_poses,
        constants,
        retargeter,
        task,
        False,
        a.output.parent,
        a.pair.stem,
        np.array([0.2, 0, 0]),
    )
    feet = extract_foot_sticking_sequence_velocity(actor, retargeter.demo_joints, motion.toe_names)
    retargeter.retarget_motion(
        human_joint_motions=actor,
        object_poses=object_poses,
        object_poses_augmented=aug,
        object_points_local_demo=demo,
        object_points_local=pts,
        foot_sticking_sequences=feet,
        q_a_init=q,
        q_nominal_list=nom,
        original=True,
        dest_res_path=str(a.output),
    )
    with np.load(a.output) as z:
        payload = {k: z[k] for k in z.files}
    trajectory_report = {"selected": "forward"}
    if a.trajectory_solve == "bidirectional" and len(actor) > 1:
        import tempfile
        from trajectory_solver import select_trajectory
        # An independent solve from the other end escapes history-dependent IK branches.
        reverse_actor = actor[::-1].copy()
        reverse_objects = np.tile([1.0, 0, 0, 0, 0, 0, 0], (len(actor), 1))
        rq, rnom, raug, reverse_actor, reverse_objects = initialize_robot_pose(
            "robot_only", "smplx", reverse_actor, reverse_objects, constants,
            retargeter, task, False, a.output.parent, a.pair.stem,
            np.array([0.2, 0, 0]))
        rfeet = extract_foot_sticking_sequence_velocity(
            reverse_actor, retargeter.demo_joints, motion.toe_names)
        with tempfile.TemporaryDirectory(prefix="omni-backward-") as temp:
            reverse_output = Path(temp) / "motion.npz"
            retargeter.retarget_motion(
                human_joint_motions=reverse_actor, object_poses=reverse_objects,
                object_poses_augmented=raug, object_points_local_demo=demo,
                object_points_local=pts, foot_sticking_sequences=rfeet,
                q_a_init=rq, q_nominal_list=rnom, original=True,
                dest_res_path=str(reverse_output))
            with np.load(reverse_output) as saved:
                backward = saved['qpos'][::-1].copy()
        payload['qpos_forward'] = payload['qpos'].copy()
        payload['qpos'], trajectory_report = select_trajectory(
            retargeter, payload['qpos'], backward, actor, pts, feet)
        if 'cost' in payload:
            payload['cost_forward'] = payload['cost'].copy()
        payload['cost'] = np.asarray(trajectory_report.pop('frame_cost'))
        print('Trajectory selection: ' + json.dumps(trajectory_report), flush=True)
    if len(payload["qpos"]) != len(partner):
        raise ValueError("Solver output frame alignment mismatch")
    payload.update(partner_joints=partner, source_frame=frames, fps=fps, actor_joints=actor)
    if a.palm_motion:
        from palm_orientation import optimize
        import time
        with np.load(a.palm_motion) as saved:
            surface={k:saved[k] for k in saved.files}
        if len(surface['vertices'])!=len(payload['qpos']):
            raise ValueError('PAL surface and robot frame counts differ')
        started=time.perf_counter()
        payload['qpos_reference']=payload['qpos'].copy()
        payload['qpos'],payload['palm_angles_deg']=optimize(surface,payload['qpos'])
        payload['palm_seconds']=np.array(time.perf_counter()-started)
    np.savez_compressed(a.output, **payload)
    a.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                upstream="d1e8499c10670f44c49d161b9876a4c17ac305f7",
                robot_dof=23,
                configuration="legacy-user-tuned",
                trajectory_solve=a.trajectory_solve,
                trajectory_report=trajectory_report,
                partner_in_objective=False,
                new_region_tracking=False,
                additional_contact_optimization=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
