import cv2
import dlib
import numpy as np 
detector = dlib.get_frontal_face_detector()
# cap=cv2.VideoCapture(-1)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def one_face(im):
	dets=detector(im);
	#print len(dets)
	if not len(dets)==0:
		d=dets[0]
		if len(im.shape)==2:
			return im[d.top():d.bottom(),d.left():d.right()]
		if len(im.shape)==3:
			return im[d.top():d.bottom(),d.left():d.right(),:]
		# d.left(),d.top()),(d.right(),d.bottom()
	else:
		return None
def one_big_face(im):
	dets=detector(im);
	#print len(dets)
	if not len(dets)==0:
		d=dets[0]
		h=d.bottom()-d.top()
		w=d.right()-d.left()
		# print h,w
		d1=(d.top()-h/4)
		if d1<0:d1=0
		d2=(d.bottom()+h/4)
		if d2>im.shape[0]:
			d2=im.shape[0]
		d3=(d.left()-w/4)
		if d3<0:d3=0
		d4=(d.right()+w/4)
		if d4>im.shape[1]:
			d4=im.shape[1]
		if len(im.shape)==2:
			return im[d1:d2,d3:d4]
		if len(im.shape)==3:
			return im[d1:d2,d3:d4,:]
		# d.left(),d.top()),(d.right(),d.bottom()
	else:
		return im
def get_shape(im):
	dets=detector(im);
	# print dets
	face_pts=np.zeros((68,2))
	if not len(dets)==0:
		shape = predictor(im, dets[0])
		for i in range(68):
			face_pts[i,0]=shape.part(i).x
			face_pts[i,1]=shape.part(i).y
	return face_pts
def get_faceim_shape(im):
	rect=dlib.rectangle(0,0,im.shape[1],im.shape[0])
	face_pts=np.ndarray((68,2))
	shape = predictor(im, rect)
	for i in range(68):
		face_pts[i,0]=shape.part(i).x
		face_pts[i,1]=shape.part(i).y
	return face_pts
def list_faces(imgs_path,dst_folder):
	for pth in imgs_path:
		if '.jpg' or '.png' in pth:
			imi=cv2.imread(pth)
			try:
				imf=one_face(imi)
				pre_pth=imgs.split('/')[-1]
				cv2.imwrite(dst_folder+'/'+pre_pth,imf)
			except Exception as e:
				print pth, 'no face found...'
