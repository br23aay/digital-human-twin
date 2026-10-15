import sys, json, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image
img_path, model, canon, out = sys.argv[1:5]
tris=[]
for line in open(canon):
    if line.startswith('f '):
        tris.append([int(t.split('/')[0])-1 for t in line.split()[1:4]])
tris=np.array(tris)
opts=vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model),output_facial_transformation_matrixes=True,output_face_blendshapes=True,num_faces=1)
det=vision.FaceLandmarker.create_from_options(opts)
im=Image.open(img_path).convert('RGB'); W,H=im.size
res=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(im)))
lm=np.array([[p.x,p.y,p.z] for p in res.face_landmarks[0]])
M=np.array(res.facial_transformation_matrixes[0])
yaw=np.degrees(np.arctan2(M[0,2],M[2,2])); pitch=np.degrees(np.arcsin(-M[1,2]))
print('landmarks',lm.shape,'tris',tris.shape,'yaw %.1f pitch %.1f'%(yaw,pitch))
# crop texture to face bbox with margin
x0,y0=lm[:,0].min(),lm[:,1].min(); x1,y1=lm[:,0].max(),lm[:,1].max()
mx,my=(x1-x0)*.08,(y1-y0)*.08
cx0,cy0,cx1,cy1=[int(v) for v in ((x0-mx)*W,(y0-my)*H,(x1+mx)*W,(y1+my)*H)]
crop=im.crop((cx0,cy0,cx1,cy1)); crop.save(out+'/face_texture.jpg',quality=92)
cw,ch=crop.size
uv=np.stack([(lm[:,0]*W-cx0)/cw, 1-(lm[:,1]*H-cy0)/ch],1)
# metric-ish coords: scale so face width (234-454) ~ 0.14 m
P=np.stack([lm[:,0]*W, -lm[:,1]*H, -lm[:,2]*W],1)
P-=P[1]  # nose tip origin
s=0.14/np.linalg.norm(P[234]-P[454]); P*=s
V=P[:468]; UV=uv[:468]
with open(out+'/face.obj','w') as f:
    f.write('mtllib face.mtl\nusemtl face\n')
    for v in V: f.write('v %.5f %.5f %.5f\n'%tuple(v))
    for t in UV: f.write('vt %.5f %.5f\n'%tuple(t))
    for a,b,c in tris: f.write('f %d/%d %d/%d %d/%d\n'%(a+1,a+1,b+1,b+1,c+1,c+1))
open(out+'/face.mtl','w').write('newmtl face\nKd 1 1 1\nmap_Kd face_texture.jpg\n')
json.dump({'landmarks':lm.tolist(),'yaw':float(yaw),'pitch':float(pitch),'blendshapes':{c.category_name:round(float(c.score),3) for c in res.face_blendshapes[0]}},open(out+'/landmarks.json','w'))
json.dump(tris.flatten().tolist(),open(out+'/tris.json','w'))
