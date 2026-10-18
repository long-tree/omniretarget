"""Evaluate whole trajectories using the existing tracking objective and limits."""
import numpy as np
from holosoma_retargeting.src.utils import (
    create_interaction_mesh, get_adjacency_list,
    calculate_laplacian_coordinates, calculate_laplacian_matrix,
)


def trajectory_metrics(solver, q, actor, ground, feet):
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite trajectory')
    laplacian = 0.0
    lap_frames = []
    penetration = 0.0
    foot_positions = []
    for frame, pose in enumerate(q):
        points = ground[frame] if isinstance(ground, list) else ground
        target = actor[frame, solver.smplh_mapped_joint_indices]
        vertices, tetrahedra = create_interaction_mesh(np.vstack([target, points]))
        adjacency = get_adjacency_list(tetrahedra, len(vertices))
        goal = calculate_laplacian_coordinates(vertices, adjacency)
        _, positions, _ = solver._calc_manipulator_jacobians(
            pose, solver.laplacian_match_links, obj_frame=False)
        robot = np.vstack([list(positions.values()), points])
        matrix = calculate_laplacian_matrix(robot, adjacency)
        value = float(solver.laplacian_weights * np.sum((matrix @ robot - goal) ** 2))
        laplacian += value
        lap_frames.append(value)
        _, positions, _ = solver._calc_manipulator_jacobians(pose, solver.foot_links)
        foot_positions.append(positions)
        _, distances = solver._update_jacobians_and_phis_from_q(pose)
        penetration = max(penetration, max((-float(x) for x in distances.values()), default=0.0))
    selected = q[:, solver.q_a_indices]
    differences = np.diff(selected, axis=0)
    weight = np.asarray(solver.smooth_weight)
    smooth = (np.sum(differences * (differences @ weight)) if weight.ndim == 2
              else np.sum(weight * differences ** 2))
    posture_frames = np.sum(selected ** 2 * solver.Q_diag, axis=1)
    posture = np.sum(posture_frames)
    smooth_frames = (np.sum(differences * (differences @ weight), axis=1) if weight.ndim == 2
                     else np.sum(weight * differences ** 2, axis=1))
    frame_cost = np.asarray(lap_frames) + posture_frames + np.r_[0.0, smooth_frames]
    locked_xy = 0.0
    for frame in range(1, len(q)):
        for name in solver.foot_links:
            side = 'l' if 'left' in name else 'r'
            active = any(str(key).lower().startswith(side) and value for key, value in feet[frame].items())
            if active and solver.activate_foot_sticking:
                locked_xy = max(locked_xy, float(np.max(np.abs(
                    foot_positions[frame][name][:2] - foot_positions[frame-1][name][:2]))))
    limits = max(0.0, float(np.max(solver.q_a_lb - selected)),
                 float(np.max(selected - solver.q_a_ub)))
    return dict(objective=float(laplacian + smooth + posture), laplacian=float(laplacian),
                smooth=float(smooth), posture=float(posture), locked_xy=locked_xy,
                penetration=penetration, joint_violation=limits, frame_cost=frame_cost.tolist())


def select_trajectory(solver, forward, backward, actor, ground, feet):
    metrics = {name: trajectory_metrics(solver, q, actor, ground, feet)
               for name, q in [('forward', forward), ('backward', backward)]}
    f, b = metrics['forward'], metrics['backward']
    # Compare feasible whole trajectories; never splice poses from separate branches.
    eligible = (b['joint_violation'] <= max(1e-7, f['joint_violation']) + 1e-7
                and b['locked_xy'] <= max(solver.foot_sticking_tolerance, f['locked_xy']) + 2e-6
                and b['penetration'] <= max(solver.penetration_tolerance, f['penetration']) + 2e-6)
    selected = 'backward' if eligible and b['objective'] < f['objective'] else 'forward'
    frame_cost = metrics[selected]['frame_cost']
    for metric in metrics.values():
        metric.pop('frame_cost')
    return (backward if selected == 'backward' else forward), dict(
        selected=selected, metrics=metrics, frame_cost=frame_cost)
