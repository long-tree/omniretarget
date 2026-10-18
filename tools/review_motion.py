"""Synchronized native comparison; robot tracks exactly the displayed small human."""
import sys, json, time, argparse, subprocess, tempfile, io
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mujoco
REPO=Path(__file__).resolve().parents[1]
HRI=REPO.parent
sys.path.insert(0,str(HRI/"scripts"))
sys.path.insert(0,str(HRI/'human-retarget'))
from interx_contact.size_render import load_variants as load_humans
from interx_contact.render import Scene
from interx_contact.view_controls import ClipNavigation
from interx_contact.pipeline import atomic_json

def load_variants(folder):
    with np.load(folder/'surface.npz') as z: motion={k:z[k] for k in z.files}
    with np.load(folder/'motion.npz') as z:
        q=z['qpos'].copy();reference=z['qpos_reference'].copy() if 'qpos_reference' in z else q.copy()
        forward=z['qpos_forward'].copy() if 'qpos_forward' in z else None
    variants=[{**motion,'label':'SMPL | resized interaction'}]
    if forward is not None:
        variants.append({**motion,'robot_qpos':forward,'label':'ROBOT | original forward solve'})
    variants.extend([{**motion,'robot_qpos':reference,'label':'ROBOT | selected whole trajectory'},
                     {**motion,'robot_qpos':q,'label':'ROBOT | PAL joint solve'}])
    return variants,[None]*len(variants),{'frames':len(q)},None


class ClipNavigation:
    def __init__(self,folder):
        self.folders=[folder];self.index=0;self.pending=0
    def request(self,d):pass
    def take(self):return None
    def click(self,*args):return False
    def draw(self,*args):pass

class ComparisonScene(Scene):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'humans.xml'
            mujoco.mj_saveLastXML(str(path),self.model)
            spec=mujoco.MjSpec.from_file(str(path))
            region=json.loads((HRI/'human-retarget/interx_contact/third_party/chi3d/info/contact_regions.json').read_text())
            tex='\n'.join(f'vt {(r+.5)/75:.9f} 0.5' for r in range(75))
            faces='\n'.join('f '+' '.join(f'{v+1}/{rid+1}' for v in tri) for tri,rid in zip(self.faces,self.face_region))
            for person in range(2):
                vertices='\n'.join('v '+' '.join(map(str,v)) for v in self.motion['vertices'][0,person])
                spec.assets[f'p{person}.obj']=(vertices+'\n'+tex+'\n'+faces+'\n').encode()
            buf=io.BytesIO();Image.fromarray(np.uint8(np.tile(np.asarray(region['rid_to_color'])[None],(8,1,1)))).save(buf,format='PNG')
            spec.assets['regions.png']=buf.getvalue()
            robot=mujoco.MjSpec.from_file(str(HRI/'omni-retarget/src/holosoma_retargeting/holosoma_retargeting/models/g1_23/g1_23dof_rev_1_0.xml'))
            spec.attach(robot,prefix='robot/',frame=spec.worldbody.add_frame())
            self.model=spec.compile()
        # One matte floor; remove the robot XML's checkerboard ground.
        planes=np.flatnonzero(self.model.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE))
        for k,gid in enumerate(planes):
            self.model.geom_matid[gid]=-1
            self.model.geom_rgba[gid]=[.16,.18,.21,1. if k==0 else 0.]
        self.data=mujoco.MjData(self.model)
        for gid in self.gids:
            self.model.geom_pos[gid]=0;self.model.geom_quat[gid]=[1,0,0,0]
        self.robot_geoms=[i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('robot/')]
        # Unnamed robot geoms are identified by their body, too.
        self.robot_geoms=[i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[i])) or '').startswith('robot/')]
        self.robot_colors=self.model.geom_rgba[self.robot_geoms].copy()
    def set_mesh(self,frame,raw,context):
        super().set_mesh(frame,raw,context)
        if self.motion.get('unavailable'):
            self.model.geom_rgba[self.gids]=0
            self.model.geom_rgba[self.robot_geoms]=0
            return
        robot='robot_qpos' in self.motion
        self.model.geom_rgba[self.robot_geoms]=self.robot_colors if robot else 0
        if robot:
            self.data.qpos[:]=self.motion['robot_qpos'][frame]
            self.model.geom_matid[self.gids[0]]=-1
            self.model.geom_rgba[self.gids[0]]=[.3,.8,1.,.22]
            mujoco.mj_forward(self.model,self.data)

def interactive(folder, max_frames=0, screenshot=None, start_frame=0, paused=False):
    import glfw
    import mujoco
    navigation = ClipNavigation(folder)
    variants, mappings, report, signature = load_variants(folder)
    if not glfw.init(): raise RuntimeError('Desktop display required')
    window = glfw.create_window(1800, 650, folder.name+' | size transfer: Space pause, arrows step, drag orbit, wheel zoom', None, None)
    if not window: raise RuntimeError('Cannot create GLFW window')
    glfw.make_context_current(window); glfw.swap_interval(1)
    scene = ComparisonScene(variants[0], width=360, height=520)
    # Centre on the displayed frame, including original and resized pair;
    # narrow six-panel viewports need more horizontal camera clearance.
    displayed=np.concatenate([x['vertices'][start_frame%report['frames']].reshape(-1,3) for x in variants[:2]])
    lo,hi=displayed.min(0),displayed.max(0)
    scene.camera.lookat[:]=(lo+hi)/2
    scene.camera.distance=max(4.,float(np.linalg.norm(hi-lo))*2.)
    context = mujoco.MjrContext(scene.model, mujoco.mjtFontScale.mjFONTSCALE_100)
    scn = mujoco.MjvScene(scene.model, maxgeom=10000)
    n = report['frames']
    fps = float(variants[0]['fps'])
    state = {'t': start_frame%n, 'play': not (bool(max_frames) or paused), 'last': time.monotonic(), 'drag': None, 'mouse': None}
    def key(win, k, scan, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT): return
        if k in (glfw.KEY_N, glfw.KEY_PAGE_DOWN): navigation.request(1)
        if k in (glfw.KEY_P, glfw.KEY_PAGE_UP): navigation.request(-1)
        if k == glfw.KEY_SPACE: state['play'] = not state['play']
        if k == glfw.KEY_H: scene.highlight = not scene.highlight
        if k in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            state['play'] = False; state['t'] = (state['t']+(1 if k==glfw.KEY_RIGHT else -1))%n
        if k == glfw.KEY_ESCAPE: glfw.set_window_should_close(win, True)
    def scroll(win, x, y): scene.camera.distance *= np.exp(-y*.1)
    def button(win, b, action, mods):
        if b not in (glfw.MOUSE_BUTTON_LEFT,glfw.MOUSE_BUTTON_RIGHT,glfw.MOUSE_BUTTON_MIDDLE):return
        if b==glfw.MOUSE_BUTTON_LEFT and action==glfw.PRESS and navigation.click(win,glfw):return
        state['drag']=b if action==glfw.PRESS else None
        state['mouse']=glfw.get_cursor_pos(win)
    def cursor(win,x,y):
        if state['drag'] is not None and state['mouse'] is not None:
            px,py=state['mouse'];height=max(1,glfw.get_window_size(win)[1])
            action=mujoco.mjtMouse.mjMOUSE_ROTATE_V if state['drag']==glfw.MOUSE_BUTTON_LEFT else mujoco.mjtMouse.mjMOUSE_MOVE_V
            mujoco.mjv_moveCamera(scene.model,action,(x-px)/height,(y-py)/height,scene.camera)
        state['mouse']=x,y
    glfw.set_key_callback(window, key); glfw.set_scroll_callback(window, scroll)
    glfw.set_mouse_button_callback(window, button); glfw.set_cursor_pos_callback(window, cursor)
    draws = 0
    last_diagnostic_state=None
    loader=ThreadPoolExecutor(max_workers=1)
    pending=None
    load_message=""
    try:
        while not glfw.window_should_close(window):
            if pending is None:
                next_folder=navigation.take()
                if next_folder is not None:
                    load_message=f'Loading {next_folder.name}: robot optimization in background'
                    pending=(next_folder,loader.submit(load_variants,next_folder))
            if pending is not None and pending[1].done():
                next_folder,future=pending
                pending=None
                try:
                    new_variants,new_mappings,new_report,new_signature=future.result()
                except Exception as error:
                    import traceback
                    traceback.print_exc()
                    load_message=f'Cannot load {next_folder.name}: {type(error).__name__}: {error}'
                    navigation.index=navigation.folders.index(folder)
                    navigation.pending=0
                    continue
                load_message=''
                camera = (scene.camera.lookat.copy(), scene.camera.distance, scene.camera.azimuth, scene.camera.elevation)
                new_scene = ComparisonScene(new_variants[0], width=360, height=520)
                new_scene.highlight = scene.highlight
                new_scene.camera.lookat[:] = camera[0]
                new_scene.camera.distance, new_scene.camera.azimuth, new_scene.camera.elevation = camera[1:]
                context.free()
                scene.close()
                scene = new_scene
                context = mujoco.MjrContext(scene.model, mujoco.mjtFontScale.mjFONTSCALE_100)
                scn = mujoco.MjvScene(scene.model, maxgeom=10000)
                variants, mappings, report, signature = new_variants, new_mappings, new_report, new_signature
                folder = next_folder
                n, fps = report['frames'], float(variants[0]['fps'])
                state.update(t=0, last=time.monotonic(), drag=None)
            now = time.monotonic(); step = int((now-state['last'])*fps)
            if state['play'] and step:
                state['t'] = (state['t']+1)%n; state['last'] = now
            if not state['play']: state['last'] = now
            diagnostic_state=(str(folder),state['t'])
            if diagnostic_state!=last_diagnostic_state:
                atomic_json(folder.parents[1]/'viewer_state.json',{'clip':folder.name,'directory':str(folder),'frame':state['t'],'source_frame':int(variants[0]['source_frame'][state['t']])})
                last_diagnostic_state=diagnostic_state
            width, height = glfw.get_framebuffer_size(window)
            for col, (motion, mapping) in enumerate(zip(variants, mappings)):
                scene.motion = motion; scene.set_mesh(state['t'], False, context)
                mujoco.mjv_updateScene(scene.model, scene.data, mujoco.MjvOption(), None, scene.camera, mujoco.mjtCatBit.mjCAT_ALL, scn)
                scene.markers(scn, mapping, state['t'])
                rect = mujoco.MjrRect(col*width//len(variants), 0, width//len(variants), max(1, height-int(48*height/max(glfw.get_window_size(window)[1], 1))))
                mujoco.mjr_render(rect, scn, context)
                detail=motion.get('label','ORIGINAL | source body' if col==0 else f'Variant {col}')+f'\nframe {state["t"]} | {"PLAY" if state["play"] else "PAUSED"} | Space pause; arrows step'
                if mapping is not None:
                    lo,hi=mapping['offsets'][state['t']:state['t']+2]
                    if hi>lo:
                        gap=np.linalg.norm(mapping['points1'][lo:hi]-mapping['points2'][lo:hi],axis=-1).mean()*1000
                        detail+=f'\nsample gap {gap:.1f}mm | {hi-lo} pairs'
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, rect,
                                   detail, '', context)
            navigation.draw(window, glfw, mujoco, context, width, height)
            if load_message:
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,mujoco.MjrRect(0,40,width,height-40),load_message,'',context)
            if 'region_states' in variants[-1]:
                legend='ORANGE: retained contact anchors | PURPLE: contact-chain deformation | CYAN: transition deformation | GREY: <2mm change | H: colors'
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                                   mujoco.MjrRect(0,28,width,height-28),legend,'',context)
            spacing = report.get('spacing')
            if spacing:
                info = f'P1 approach {spacing["approach_translation_m"]:.3f}m | k={spacing["initial_distance_coefficient"]:.3f} | P2 trajectory preserved'
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                   mujoco.MjrRect(0,0,width,height), info, '', context)
            draws += 1
            if max_frames and draws>=max_frames:
                if screenshot:
                    rgb = np.zeros((height, width, 3), np.uint8)
                    mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), context)
                    Image.fromarray(np.flipud(rgb)).save(screenshot)
                glfw.set_window_should_close(window, True)
            glfw.swap_buffers(window); glfw.poll_events()
    finally:
        loader.shutdown(wait=False,cancel_futures=True)
        context.free(); glfw.destroy_window(window); glfw.terminate()



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clip',type=Path,required=True);p.add_argument('--max-viewer-frames',type=int,default=0)
    a=p.parse_args();interactive(a.clip.resolve(),max_frames=a.max_viewer_frames)
