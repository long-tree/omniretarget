"""Local searchable result catalog launching synchronized MuJoCo comparisons."""
import argparse,json,subprocess,sys,threading,os
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8771);a=p.parse_args();root=a.root.resolve();viewer=None;lock=threading.Lock()
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/api/clips':
   rows=json.loads((root/'catalog.json').read_text())
   for r in rows:
    d=Path(r['directory']);r['status']='ready' if (d/'complete.json').exists() else 'failed' if (d/'run.log').exists() or r.get('error') else 'pending'
    if (d/'complete.json').exists():r['seconds']=json.loads((d/'complete.json').read_text()).get('seconds')
   data=json.dumps(rows).encode();mime='application/json'
  elif self.path=='/':data=(P/'catalog.html').read_bytes();mime='text/html; charset=utf-8'
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type',mime);self.end_headers();self.wfile.write(data)
 def do_POST(self):
  global viewer
  try:
   if self.path!='/api/open':raise ValueError('Unknown endpoint')
   key=json.loads(self.rfile.read(int(self.headers['Content-Length'])))['id']
   row=next(r for r in json.loads((root/'catalog.json').read_text()) if r['id']==key)
   folder=Path(row['directory'])
   if not (folder/'complete.json').exists():raise ValueError('Result not complete')
   with lock:
    if viewer is not None and viewer.poll() is None:viewer.terminate()
    env=os.environ.copy();env['MUJOCO_GL']='glfw'
    log=(root/'viewer.log').open('a')
    viewer=subprocess.Popen([sys.executable,str(P/'review_motion.py'),'--clip',str(folder)],env=env,stdout=log,stderr=log);log.close()
   self.send_response(200);self.end_headers();self.wfile.write(b'{"ok":true}')
  except Exception as e:self.send_error(400,str(e))
print(f'Catalog http://127.0.0.1:{a.port}',flush=True)
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
