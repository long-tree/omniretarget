"""Joint local solve of source palm position, normal and motion preservation."""
from pathlib import Path
import json
import numpy as np
import mujoco,trimesh
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'src/holosoma_retargeting/holosoma_retargeting/models/g1_23'
def palm_patch(side):
 mesh=trimesh.load(MODEL/f'meshes/{side}_wrist_roll_rubber_hand.STL')
 sign=-1 if side=='left' else 1
 keep=(mesh.triangles_center[:,0]>.16)&(mesh.triangles_center[:,0]<.225)&(mesh.face_normals[:,1]*sign>.8)
 normal=np.average(mesh.face_normals[keep],axis=0,weights=mesh.area_faces[keep]);normal/=np.linalg.norm(normal)
 center=np.average(mesh.triangles_center[keep],axis=0,weights=mesh.area_faces[keep])
 return mesh,keep,center,normal

def optimize(motion,q):
 model=mujoco.MjModel.from_xml_path(str(MODEL/'g1_23dof_rev_1_0.xml'))
 data=mujoco.MjData(model);out=q.copy();faces=motion['faces'];v=motion['vertices'];j=motion['joints'];errors=[];position_errors=[]
 cfg=json.loads((ROOT/'configs/palm_frames_v1.json').read_text())
 for side,wrist in [('left',20),('right',21)]:
  point=next(p for p in cfg['points'] if p['side']==side)
  if point['human'] is None:raise ValueError('Confirmed human palm required')
  local_normal=np.asarray(point['robot']['normal'],float);local_normal/=np.linalg.norm(local_normal)
  # Approved material point: position and normal are evaluated on this same surface.
  pairdist=np.linalg.norm(j[:,0,wrist,None]-j[:,1,[20,21]],axis=-1)
  t,other=np.unravel_index(pairdist.argmin(),pairdist.shape);other+=20
  ids=np.flatnonzero(np.linalg.norm(v[t,0]-j[t,0,wrist],axis=-1)<.09)
  partner_ids=np.flatnonzero(np.linalg.norm(v[t,1]-j[t,1,other],axis=-1)<.09)
  selected=np.array([point['human']['face']],dtype=int)
  columns=np.array([int(model.jnt_qposadr[k]) for k in range(model.njnt) if (model.joint(k).name or '').startswith(side+'_') and any(x in model.joint(k).name for x in ('shoulder','elbow','wrist'))])
  joints=[k for k in range(model.njnt) if int(model.jnt_qposadr[k]) in columns];low,high=model.jnt_range[joints].T
  body=model.body(side+'_rubber_hand_link').id
  offset=np.asarray(point['robot']['position'],float)
  distance=np.array([np.linalg.norm(v[k,0,ids,None]-v[k,1,partner_ids][None],axis=-1).min() for k in range(len(q))])
  active=distance<.05
  weight=active.astype(float)
  for k in np.flatnonzero(active):
   for f in range(max(0,k-15),min(len(q),k+16)):
    x=max(0,1-abs(f-k)/15);weight[f]=max(weight[f],x*x*(3-2*x))
  previous=np.zeros(len(columns))
  for f in range(len(q)):
   if weight[f]==0:previous*=0;continue
   tri=v[f,0,faces[selected]];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).sum(0);n/=max(np.linalg.norm(n),1e-12);n*=point['human'].get('sign',1)
   data.qpos[:]=q[f];mujoco.mj_forward(model,data);position=data.xpos[body]+data.xmat[body].reshape(3,3)@offset
   before=data.xmat[body].reshape(3,3)@local_normal
   target=(v[f,0,faces[point['human']['face']]]*np.asarray(point['human']['barycentric'])[:,None]).sum(0)
   # Preserve the source palm's position relative to the untouched partner.
   # This retains its original gap rather than forcing palm-to-palm coincidence.
   blend=weight[f]
   position_target=(1-blend)*position+blend*target
   reference_points=[]
   for joint_name in (side+'_elbow_link',side+'_rubber_hand_link'):
    bid=model.body(joint_name).id
    reference_points.append((bid,data.xpos[bid].copy()))
   def residual(x):
    data.qpos[:]=q[f];data.qpos[columns]=x;mujoco.mj_forward(model,data)
    rot=data.xmat[body].reshape(3,3);delta=x-q[f,columns]
    return np.r_[(data.xpos[body]+rot@offset-position_target)/.015,
                 np.sqrt(10*weight[f])*(rot@local_normal-n),
                 *[(data.xpos[bid]-ref)/.08 for bid,ref in reference_points],
                 delta/.5,(delta-previous)/.2]
   solution=least_squares(residual,np.clip(q[f,columns],low,high),bounds=(low,high),max_nfev=40)
   out[f,columns]=solution.x;previous=solution.x-q[f,columns]
   data.qpos[:]=out[f];mujoco.mj_forward(model,data);after=data.xmat[body].reshape(3,3)@local_normal
   if active[f]:
    position_errors.append([float(np.linalg.norm(position-target)),float(np.linalg.norm(data.xpos[body]+data.xmat[body].reshape(3,3)@offset-target))])
   if active[f]:errors.append([float(np.degrees(np.arccos(np.clip(before@n,-1,1)))),float(np.degrees(np.arccos(np.clip(after@n,-1,1))))])
 print('Joint solve palm position mean before/after (m):',np.mean(position_errors,axis=0) if position_errors else [],flush=True)
 return out,np.asarray(errors)
