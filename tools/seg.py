import sys,numpy as np,mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image
SC='/tmp/claude-0/-home-claude/2e0e4172-1eed-530e-8da6-98bfe85d24d4/scratchpad/'
P='/mnt/user-data/uploads/Digital Human Twin/Pictures/Bharadwaj/'
seg=vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=SC+'mp/selfie_multiclass.tflite'),output_category_mask=True,output_confidence_masks=True))
names=['Front Head','Left Head','Right Head','Back Head','Front Body','Left Side Body','Right SIde Body','Back Side Body']
tiles=[]
for n in names:
    im=np.asarray(Image.open(P+n+'.jpeg').convert('RGB'))
    r=seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=im))
    cat=np.squeeze(r.category_mask.numpy_view()).copy()   # 0 bg,1 hair,2 body-skin,3 face-skin,4 clothes,5 others
    bgconf=np.squeeze(r.confidence_masks[0].numpy_view())
    np.save(SC+'seg/'+n+'.npy',cat); np.save(SC+'seg/'+n+'_bg.npy',bgconf)
    pal=np.array([[0,0,0],[200,60,200],[240,180,120],[255,220,160],[60,140,255],[120,255,120]],np.uint8)
    vis=(im*0.4+pal[cat]*0.6).astype(np.uint8)
    tiles.append(Image.fromarray(vis).resize((240,320)))
    print(n, {k:int((cat==k).sum()) for k in range(6)})
c=Image.new('RGB',(960,640))
for i,t in enumerate(tiles): c.paste(t,((i%4)*240,(i//4)*320))
c.save(SC+'seg/overview.jpg')
