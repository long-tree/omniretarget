"""Run all three raw Inter-X actions through the latest usable 23DoF scheme."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];H=ROOT.parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--human-python',default=str(H/'human-retarget/interx_contact/.venv/bin/python'))
 p.add_argument('--omni-python',default=str(H.parent/'inter/omniretarget/src/holosoma_retargeting/.venv/bin/python'))
 p.add_argument('--action-workers',type=int,default=1,help='Concurrent GPU preprocessing/remap pipelines; 1 avoids GPU memory contention')
 p.add_argument('--workers',type=int,default=2,help='Independent robot clips per action')
 a=p.parse_args()
 if a.action_workers<1 or a.workers<1:raise ValueError('Positive worker counts required')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 actions={'handshake':'A001','hugging':'A000','highfive':'A016'}
 expected={k:sorted(x.name for x in (a.dataset/'motions').iterdir() if x.is_dir() and code in x.name) for k,code in actions.items()}
 if any(not x for x in expected.values()):raise ValueError('One or more raw actions empty')
 (out/'expected.json').write_text(json.dumps(expected,indent=2))
 def worker(action):
  start=time.perf_counter();dest=out/action;dest.mkdir(exist_ok=True)
  cmd=[a.human_python,str(ROOT/'tools/workspace_pipeline.py'),'--dataset',str(a.dataset.resolve()),'--models',str(a.models.resolve()),'--output',str(dest),'--actions',action,'--limit','0','--workers',str(a.workers),'--human-python',a.human_python,'--omni-python',a.omni_python]
  with (dest/'pipeline.log').open('a') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  return dict(action=action,returncode=result.returncode,seconds=time.perf_counter()-start)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.action_workers) as pool:records=list(pool.map(worker,actions))
 audit=[]
 import numpy as np
 for action,ids in expected.items():
  for clip in ids:
   d=out/action/'results'/clip
   try:
    with np.load(d/'motion.npz') as z, np.load(d/'pair.npz') as pair:
     q=z['qpos'];assert q.shape==(len(pair['source_frame']),30) and np.isfinite(q).all()
     np.testing.assert_array_equal(z['source_frame'],pair['source_frame']);assert 'qpos_reference' in z and 'qpos_forward' in z
    assert (d/'complete.json').exists()
    audit.append(dict(action=action,id=clip,status='ok'))
   except Exception as e:audit.append(dict(action=action,id=clip,status='failed',error=str(e)))
 catalog=[]
 for action in actions:
  file=out/action/'catalog.json'
  if file.exists():catalog.extend(json.loads(file.read_text()))
 (out/'catalog.json').write_text(json.dumps(catalog,indent=2))
 report=dict(pipelines=records,expected=sum(map(len,expected.values())),passed=sum(x['status']=='ok' for x in audit),clips=audit)
 (out/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='clips'}),flush=True)
 if report['passed']!=report['expected'] or any(x['returncode'] for x in records):sys.exit(1)
if __name__=='__main__':main()
