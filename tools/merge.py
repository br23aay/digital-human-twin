import sys,json,numpy as np,trimesh,mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image
SC,P,OUT=sys.argv[1:4]
meta=json.load(open(OUT+'/body_meta.json'))
det=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=SC+'mp/face_landmarker.task')))
im=np.asarray(Image.open(P+'Front Body.jpeg').convert('RGB'));H,W=im.shape[:2]
r=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=im))
lm=r.face_landmarks[0]; chin_px=lm[152].y*H
mpp=meta['mpp']; top=meta['height_m']
chin_y=top-(chin_px-meta['top_f'])*mpp
L_b=top-chin_y
body=trimesh.load(OUT+'/body.glb',force='mesh'); head=trimesh.load(OUT+'/head.glb',force='mesh')
# head: top at its max y; chin is 0.235 m below top by construction
hv=head.vertices; htop=hv[:,1].max()
sc=L_b/0.235
head.vertices=(hv-[0,htop,0])*sc+[0,top,0]
# align x/z by centroids of the region between chin and top
def cen(m,y0,y1):
    v=m.vertices; s=(v[:,1]>y0)&(v[:,1]<y1); return v[s][:,[0,2]].mean(0)
cb=cen(body,chin_y+0.03,top-0.03); ch=cen(head,chin_y+0.03,top-0.03)
head.vertices[:,[0,2]]+=cb-ch
def cut(m,keep):
    f=m.faces; k=keep(m.vertices)[f].all(1); m.update_faces(k); m.remove_unreferenced_vertices(); return m
body=cut(body,lambda v:v[:,1]<chin_y-0.045)
head=cut(head,lambda v:v[:,1]>chin_y-0.12)
sc_=trimesh.Scene(); sc_.add_geometry(body,node_name='body'); sc_.add_geometry(head,node_name='head')
sc_.export(OUT+'/digital_human_v1.glb')
print('chin y',round(chin_y,3),'head scale',round(sc,3),'offset',(cb-ch).round(3))
# web version: decimate and keep colours
import fast_simplification as fs
from scipy.spatial import cKDTree
web=trimesh.Scene()
for name,g,frac in (('body',body,0.7),('head',head,0.5)):
    v,f=fs.simplify(g.vertices.astype(np.float32),g.faces.astype(np.int32),target_reduction=frac)
    _,idx=cKDTree(g.vertices).query(v); m=trimesh.Trimesh(v,f,process=False)
    m.visual=trimesh.visual.ColorVisuals(m,vertex_colors=g.visual.vertex_colors[idx]); web.add_geometry(m,node_name=name)
web.export(OUT+'/digital_human_v1_web.glb')
