# python get_atten_map.py

import numpy as np 
import cv2

def get_au_tg_map(array_str):
	array_str=array_str[1:-2]
	str_d=array_str.split(',')
	str_dt=[float(i) for i in str_d]
	region_array=np.zeros((10,4))
	# print str_dt
	try:
	# print len(str_dt)
		W=1024
		H=768
		arr2d=np.array(str_dt).reshape((2,68))
		# print arr2d
		arr2d[0,:]=arr2d[0,:]/W*100
		arr2d[1,:]=arr2d[1,:]/H*100
		region_bbox=[]
		ruler=abs(arr2d[0,39]-arr2d[0,42])
		# print ruler
		region_bbox+=[[arr2d[0,21],arr2d[1,21]-ruler/2,arr2d[0,22],arr2d[1,22]-ruler/2]]
		region_bbox+=[[arr2d[0,21]/2+arr2d[0,22]/2,arr2d[1,21]/2+arr2d[1,22]/2,arr2d[0,39]/2+arr2d[0,42]/2,arr2d[1,39]/2+arr2d[1,42]/2]]

		region_bbox+=[[arr2d[0,18],arr2d[1,18]-ruler/3,arr2d[0,25],arr2d[1,25]-ruler/3]]		
		#replace layers		
		
		region_bbox+=[[arr2d[0,19],arr2d[1,19]+ruler/3,arr2d[0,24],arr2d[1,24]+ruler/3]]#au4
		region_bbox+=[[arr2d[0,41],arr2d[1,41]+ruler,arr2d[0,46],arr2d[1,46]+ruler]]#au6
		region_bbox+=[[arr2d[0,38],arr2d[1,38],arr2d[0,43],arr2d[1,43]]]#au7
		region_bbox+=[[arr2d[0,49],arr2d[1,49],arr2d[0,53],arr2d[1,53]]]#au10
		region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]# au12 au14 
		region_bbox+=[[arr2d[0,51],arr2d[1,51],arr2d[0,57],arr2d[1,57]]]# au17
		region_bbox+=[[arr2d[0,61],arr2d[1,61],arr2d[0,63],arr2d[1,63]]]#au 23 24
		region_bbox+=[[arr2d[0,56],arr2d[1,56]+ruler/2,arr2d[0,58],arr2d[1,58]+ruler/2]]#au23

		region_array=np.array(region_bbox)
	except Exception as e:
		i=0;
	return region_array
def get_au_tg_dlib(array_68,h,w):
	
	str_dt=list(array_68[:,0])+list(array_68[:,1])
	region_array=np.zeros((11,4))
	# print str_dt
	try:
	# print len(str_dt)
		W=w
		H=h
		arr2d=np.array(str_dt).reshape((2,68))
		# print arr2d
		arr2d[0,:]=arr2d[0,:]/W*100
		arr2d[1,:]=arr2d[1,:]/H*100
		region_bbox=[]
		ruler=abs(arr2d[0,39]-arr2d[0,42])
		# print ruler
		region_bbox+=[[arr2d[0,21],arr2d[1,21]-ruler/2,arr2d[0,22],arr2d[1,22]-ruler/2]]#0
		#region_bbox+=[[arr2d[0,21]/2+arr2d[0,22]/2,arr2d[1,21]/2+arr2d[1,22]/2,arr2d[0,39]/2+arr2d[0,42]/2,arr2d[1,39]/2+arr2d[1,42]/2]]
		region_bbox+=[[arr2d[0,18],arr2d[1,18]-ruler/3,arr2d[0,25],arr2d[1,25]-ruler/3]]#2		
		#replace layers		
		
		region_bbox+=[[arr2d[0,19],arr2d[1,19]+ruler/3,arr2d[0,24],arr2d[1,24]+ruler/3]]#3: au4
		region_bbox+=[[arr2d[0,41],arr2d[1,41]+ruler,arr2d[0,46],arr2d[1,46]+ruler]]#4: au6
		region_bbox+=[[arr2d[0,38],arr2d[1,38],arr2d[0,43],arr2d[1,43]]]#5: au7
		region_bbox+=[[arr2d[0,49],arr2d[1,49],arr2d[0,53],arr2d[1,53]]]#6: au10
		region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]#7: au12 au14 lip corner
		region_bbox+=[[arr2d[0,51],arr2d[1,51],arr2d[0,57],arr2d[1,57]]]#8: au17
		region_bbox+=[[arr2d[0,61],arr2d[1,61],arr2d[0,63],arr2d[1,63]]]#9: au 23 24
		region_bbox+=[[arr2d[0,56],arr2d[1,56]+ruler/2,arr2d[0,58],arr2d[1,58]+ruler/2]]#10: #au23

		region_array=np.array(region_bbox)
	except Exception as e:
		i=0;
	return region_array
def get_map(array_68,h,w):
	feat_map=np.zeros((100,100))
	tg_array=get_au_tg_dlib(array_68,h,w)
	# print tg_array
	for i in range(tg_array.shape[0]):
		for j in range(2):
			pt=tg_array[i,j*2:(j+1)*2]
			pt=pt.astype('uint8')
			# print pt
			for px in range(pt[0]-7,pt[0]+7):
				if px<0 or px>99:
					break
				for py in range(pt[1]-7,pt[1]+7):
					if py <0 or py>99:
						break
					d1=abs(px-pt[0])
					d2=abs(py-pt[1])
					value=1-(d1+d2)*0.07
					feat_map[py][px]=max(feat_map[py][px],value)
					# print feat_map[py][px]
	# feat_map=cv2.resize(feat_map,(224,224))
					# print value	
	return feat_map
def get_map_single_au(array_68,h,w,auid=13):
	feat_map=np.zeros((100,100))
	tg_array=get_au_tg_dlib(array_68,h,w)
	# print tg_array
	# print tg_array
	# for i in range(10):
	map_dict={0:0,1:1,2:2,13:8,}
	i=map_dict[auid]
	for j in range(2):
		pt=tg_array[i,j*2:(j+1)*2]
		pt=pt.astype('uint8')
		# print pt
		for px in range(pt[0]-5,pt[0]+5):
			if px<0 or px>99:
				break
			for py in range(pt[1]-5,pt[1]+5):
				if py <0 or py>99:
					break
				d1=abs(px-pt[0])
				d2=abs(py-pt[1])
				value=1-(d1+d2)*0.095
				feat_map[py][px]=max(feat_map[py][px],value)
	# feat_map=cv2.resize(feat_map,(224,224))
					# print value	
	return feat_map


# pt_arr='[455.0, 454.0, 457.0, 461.0, 475.0, 495.0, 521.0, 550.0, 585.0, 617.0, 643.0, 665.0, 684.0, 695.0, 702.0, 708.0, 710.0, 488.0, 508.0, 535.0, 562.0, 586.0, 628.0, 650.0, 672.0, 693.0, 704.0, 607.0, 607.0, 607.0, 608.0, 574.0, 588.0, 602.0, 615.0, 627.0, 520.0, 537.0, 555.0, 568.0, 553.0, 534.0, 632.0, 647.0, 665.0, 678.0, 666.0, 648.0, 540.0, 564.0, 585.0, 598.0, 612.0, 626.0, 642.0, 623.0, 607.0, 593.0, 579.0, 560.0, 549.0, 583.0, 597.0, 611.0, 633.0, 608.0, 595.0, 581.0, 474.0, 511.0, 549.0, 586.0, 621.0, 652.0, 678.0, 699.0, 706.0, 701.0, 679.0, 653.0, 624.0, 593.0, 559.0, 527.0, 494.0, 459.0, 441.0, 435.0, 439.0, 451.0, 453.0, 446.0, 443.0, 450.0, 469.0, 478.0, 502.0, 527.0, 553.0, 570.0, 574.0, 579.0, 576.0, 573.0, 477.0, 469.0, 470.0, 483.0, 486.0, 485.0, 487.0, 475.0, 476.0, 486.0, 493.0, 492.0, 606.0, 609.0, 609.0, 615.0, 611.0, 615.0, 616.0, 633.0, 640.0, 641.0, 638.0, 627.0, 611.0, 621.0, 625.0, 623.0, 620.0, 623.0, 625.0, 621.0]'
# im=get_map(pt_arr)
# im*=255
# im.astype('uint8')
# cv2.imwrite('/home/wei/Desktop/heat.png',im)


