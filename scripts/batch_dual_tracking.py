"""Bounded independent-clip conversion with resumable, provenance-checked outputs."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=Path,required=True,help='JSON list or {clips:[{id,pair,palm_motion?}]}')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
 p.add_argument('--threads',type=int,default=1);p.add_argument('--overwrite',action='store_true')
 a=p.parse_args()
 if a.workers<1 or a.threads<1:raise ValueError('Positive workers/threads required')
 rows=json.loads(a.manifest.read_text());rows=rows['clips'] if isinstance(rows,dict) else rows
 base=a.manifest.resolve().parent;ids=[r['id'] for r in rows]
 if len(set(ids))!=len(ids) or any(Path(i).name!=i or i in ('.','..') for i in ids):raise ValueError('Unique plain clip IDs required')
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
 code_hash=hashlib.sha256(b''.join((ROOT/x).read_bytes() for x in ['scripts/dual_person_tracking.py','scripts/palm_orientation.py','scripts/trajectory_solver.py','configs/palm_frames_v1.json','src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py'])).hexdigest()
 def worker(row):
  started=time.perf_counter();clip=row['id'];folder=a.output/clip;folder.mkdir(exist_ok=True)
  pair=(base/row['pair']).resolve();surface=(base/row['palm_motion']).resolve() if row.get('palm_motion') else None
  provenance=dict(pair_sha256=digest(pair),surface_sha256=digest(surface) if surface else None,code_sha256=code_hash)
  dest=folder/'motion.npz';stamp=folder/'complete.json'
  if not a.overwrite and dest.exists() and stamp.exists() and json.loads(stamp.read_text()).get('provenance')==provenance:
   return dict(id=clip,status='cached',seconds=0)
  temp=folder/'pending.npz';cmd=[sys.executable,str(ROOT/'scripts/dual_person_tracking.py'),'--pair',str(pair),'--output',str(temp)]
  if surface:cmd+=['--palm-motion',str(surface)]
  env=os.environ.copy()
  for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]=str(a.threads)
  env['MUJOCO_GL']='egl'
  with (folder/'run.log').open('w') as log:
   result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  elapsed=time.perf_counter()-started
  if result.returncode:return dict(id=clip,status='failed',seconds=elapsed,returncode=result.returncode,log=str(folder/'run.log'))
  import numpy as np
  with np.load(temp) as z:
   if not np.isfinite(z['qpos']).all():raise ValueError(f'{clip}: nonfinite output')
   frames=len(z['qpos'])
  temp.replace(dest)
  if temp.with_suffix('.json').exists():temp.with_suffix('.json').replace(dest.with_suffix('.json'))
  record=dict(id=clip,status='ok',seconds=elapsed,frames=frames,provenance=provenance)
  stamp.write_text(json.dumps(record,indent=2));return record
 started=time.perf_counter();records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(worker,r):r['id'] for r in rows}
  for f in concurrent.futures.as_completed(futures):
   try:r=f.result()
   except Exception as e:r=dict(id=futures[f],status='failed',error=str(e))
   records.append(r);print(json.dumps(r),flush=True)
   (a.output/'batch_report.json').write_text(json.dumps(dict(workers=a.workers,threads=a.threads,wall_seconds=time.perf_counter()-started,clips=records),indent=2))
 print('wall_seconds',time.perf_counter()-started,flush=True)
 if any(r['status']=='failed' for r in records):sys.exit(1)
if __name__=='__main__':main()
