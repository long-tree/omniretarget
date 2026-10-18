"""Inter-X -> contacts -> human Laplacian remap -> paired robot PAL outputs."""
import argparse,json,os,subprocess,sys,time,concurrent.futures
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];H=REPO.parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=H/'human-retarget/configs/laplacian.json')
 p.add_argument('--actions',nargs='+',default=['highfive','handshake']);p.add_argument('--limit',type=int,default=10,help='Per action; 0 means all')
 p.add_argument('--clip',help='One raw Inter-X clip ID, with --limit 1 and matching --actions')
 p.add_argument('--preprocess-workers',type=int,default=2)
 p.add_argument('--workers',type=int,default=2);p.add_argument('--human-python',default=str(H/'human-retarget/interx_contact/.venv/bin/python'))
 p.add_argument('--omni-python',default=str(H.parent/'inter/omniretarget/src/holosoma_retargeting/.venv/bin/python'))
 p.add_argument('--from-human-manifest',type=Path,help='Reuse existing remap manifest; skip dataset/remap stages')
 a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(H/'dataset'),str(H/'human-retarget')]);env['MUJOCO_GL']='egl'
 def run(cmd):subprocess.run(list(map(str,cmd)),env=env,check=True)
 started=time.perf_counter()
 if a.from_human_manifest:manifest=a.from_human_manifest.resolve()
 else:
  def prepare(index):
   target=out/'preprocess_shards'/str(index);target.mkdir(parents=True,exist_ok=True)
   # Reuse verified completed clips, with each shard owning a separate manifest.
   for action in a.actions:
    parent=target/action;parent.mkdir(exist_ok=True)
    for existing in (out/'source'/action).glob('*'):
     link=parent/existing.name
     if existing.is_dir() and not link.exists():link.symlink_to(existing.resolve(),target_is_directory=True)
   run([a.human_python,'-m','hri_dataset.pipeline','--dataset',a.dataset,'--models',a.models,'--output',target,'--actions',*a.actions,'--limit',a.limit,'--shards',a.preprocess_workers,'--shard-index',index,'--threads','1','--resume',*(['--clip',a.clip] if a.clip else [])])
   return json.loads((target/'manifest.json').read_text())
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.preprocess_workers) as pool:parts=list(pool.map(prepare,range(a.preprocess_workers)))
  merged=parts[0];merged['clips']=sorted([row for part in parts for row in part['clips']],key=lambda row:row['clip'])
  source=out/'source';source.mkdir(exist_ok=True);(source/'manifest.json').write_text(json.dumps(merged,indent=2))
  run([a.human_python,'-m','interx_contact.remap','--config',a.config,'--manifest',out/'source/manifest.json','--output',out/'human','--resume'])
  manifest=out/'human/manifest.json'
 sys.path.insert(0,str(H/'human-retarget'))
 from interx_contact.size_render import load_variants
 rows=json.loads(manifest.read_text())['clips'];jobs=[];catalog=[]
 for row in rows:
  folder=Path(row['directory']);clip=row['clip'];target=out/'results'/clip;target.mkdir(parents=True,exist_ok=True)
  try:
   variants,_,_,_=load_variants(folder);motion=variants[-1]
   np.savez_compressed(target/'surface.npz',**{k:motion[k] for k in ['vertices','joints','faces','source_frame','fps','shared_dz']})
   np.savez_compressed(target/'pair.npz',actor_joints=motion['joints'][:,0,:22],partner_joints=motion['joints'][:,1,:22],source_frame=motion['source_frame'],fps=motion['fps'])
   jobs.append(dict(id=clip,pair=str(target/'pair.npz'),palm_motion=str(target/'surface.npz')))
   catalog.append(dict(id=clip,action=row.get('action',folder.parent.name),directory=str(target),human_directory=str(folder)))
  except Exception as error:
   catalog.append(dict(id=clip,action=folder.parent.name,directory=str(target),error=str(error)))
  (out/'catalog.json').write_text(json.dumps(catalog,indent=2))
 (out/'jobs.json').write_text(json.dumps(dict(clips=jobs),indent=2))
 run([a.omni_python,REPO/'scripts/batch_dual_tracking.py','--manifest',out/'jobs.json','--output',out/'results','--workers',a.workers])
 print('COMPLETE seconds',time.perf_counter()-started,'catalog',out/'catalog.json')
if __name__=='__main__':main()
