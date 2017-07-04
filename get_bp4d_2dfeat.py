# python get_bp4d_2dfeat.py

import os
import scipy.io as sio
import h5py
import numpy as np
def get_shape(img):
	info_str=img.split('.')[0]
	info=info_str.split('_')
	# print info

	mat_path='/home/wei/caffedata/BUFE/BP4D/2DFeatures'
	mat_name=info[0]+'_'+info[1]+'.mat'
	feat_mat=h5py.File(mat_path+'/'+mat_name)
	np_array=feat_mat['fit/pred'][int(info[2])-1][0]
	return np.array(feat_mat[np_array]).reshape((-1))




# root='/media/wei/Seagate Backup Plus Drive1'
# fp=open(root+'/'+'BP4D_tr_fold2.txt','r')
# fw=open(root+'/'+'BP4D_SAD_tr_fold2.txt','w')
# fp_info=fp.readlines()
# print len(fp_info)
# for i,s in enumerate(fp_info):
# 	if i%5000==0:print i
# 	buff=s.split('->')
# 	fname=root+'/'+'BP4D_IMG/'+buff[0]
# 	if os.path.isfile(fname):
# 		im_shape=get_shape(buff[0])
# 		# print im_shape
# 		fw.write(s.replace('\n','->'+str(list(im_shape))+'\n'))

def get_au_tg_map(array):
	try:
		arr2d=np.array(array).reshape((2,49))
		arr2d[0,:]=arr2d[0,:]/1040*100
		arr2d[1,:]=arr2d[1,:]/1392*100
		region_bbox=[]
		ruler=abs(arr2d[0,22]-arr2d[1,25])
		# print ruler
		region_bbox+=[[arr2d[0,4],arr2d[1,4]-ruler/2,arr2d[0,5],arr2d[1,5]-ruler/2]]
		# region_bbox+=[[arr2d[0,25]/2+arr2d[0,22]/2,arr2d[1,25]/2+arr2d[1,22]/2,arr2d[0,5]/2+arr2d[0,4]/2,arr2d[1,5]/2\
		# +arr2d[1,4]/2-5]]

		region_bbox+=[[arr2d[0,1],arr2d[1,1]-ruler/3,arr2d[0,8],arr2d[1,8]]-ruler/3]		
		#replace layers		
		
		region_bbox+=[[arr2d[0,2],arr2d[1,2]+ruler/3,arr2d[0,7],arr2d[1,7]+ruler/3]]#au4
		region_bbox+=[[arr2d[0,24],arr2d[1,24]+ruler,arr2d[0,29],arr2d[1,29]+ruler]]#au6
		region_bbox+=[[arr2d[0,21],arr2d[1,21],arr2d[0,26],arr2d[1,26]]]#au7
		region_bbox+=[[arr2d[0,33],arr2d[1,33],arr2d[0,35],arr2d[1,35]]]#au10
		region_bbox+=[[arr2d[0,31],arr2d[1,31],arr2d[0,37],arr2d[1,37]]]# au12 au14 
		region_bbox+=[[arr2d[0,34],arr2d[1,34],arr2d[0,40],arr2d[1,40]]]# au17
		region_bbox+=[[arr2d[0,43],arr2d[1,43],arr2d[0,45],arr2d[1,45]]]#au 23 24
		region_bbox+=[[arr2d[0,39],arr2d[1,39]+ruler,arr2d[0,41],arr2d[1,41]+ruler]]#au23

		region_array=np.array(region_bbox)
	except Exception as e:
		region_array=np.zeros((10,4))

	return region_array

# img='M008_T5_1346.jpg'
# arr=get_shape(img).reshape((-1))
# print get_au_tg_map(arr).shape



