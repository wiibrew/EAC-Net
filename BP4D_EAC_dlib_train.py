#run  THEANO_FLAGS="device=gpu0,floatX=float32" python BP4D_EAC_dlib_train.py

import numpy as np
import theano
import theano.tensor as T
import lasagne


import skimage.transform
import sklearn.cross_validation
import pickle
import os
import re

##build the vgg model

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,DropoutLayer,ROI_SliceLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax,sigmoid
from lasagne.utils import floatX
from lasagne.layers import SliceLayer, concat,BatchNormLayer,ElemwiseSumLayer,ElemwiseMergeLayer,ReshapeLayer
from lasagne.layers import LocalResponseNormalization2DLayer,BatchNormLayer, ROI_CropLayer,ROI_GotLayer,Upscale2DLayer
from lasagne.objectives import squared_error

import get_bp4d_2dfeat
import get_attention_map_dlib
import get_face

IM_SIZE=224
X_sym = T.tensor4()
y_sym = T.tensor4()


def build_model():
    net = {}
    net['input'] = InputLayer((None, 512*20, 3, 3))

    au_fc_layers=[]
    for i in range(20):
        net['roi_AU_N_'+str(i)]=SliceLayer(net['input'],indices=slice(i*512,(i+1)*512),axis=1)

        #try to adding upsampling here for more conv

        net['Roi_upsample_'+str(i)]=Upscale2DLayer(net['roi_AU_N_'+str(i)],scale_factor=2)

        net['conv_roi_'+str(i)]=ConvLayer(net['Roi_upsample_'+str(i)],512,3)

        net['au_fc_'+str(i)]=DenseLayer(net['conv_roi_'+str(i)],num_units=150)

        au_fc_layers+=[net['au_fc_'+str(i)]]

    #
    net['local_fc']=concat(au_fc_layers)
    net['local_fc2']=DenseLayer(net['local_fc'],num_units=2048)

    net['local_fc_dp']=DropoutLayer(net['local_fc2'],p=0.5)


    # net['fc_comb']=concat([net['au_fc_layer'],net['local_fc_dp']])


    # net['fc_dense']=DenseLayer(net['fc_comb'],num_units=1024)

    # net['fc_dense_dp']=DropoutLayer(net['fc_dense'],p=0.3)

    net['real_out']=DenseLayer(net['local_fc_dp'],num_units=12,nonlinearity=sigmoid)


    # net['final']=concat([net['pred_pos_layer'],net['output_layer']])

    return net


BATCH_SIZE = 50

net = build_model()


prediction = lasagne.layers.get_output(net['real_out'], X_sym)

print 'successfully...'

# with np.load('data/model_BP4D_part_Roi_conv_V2_450.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]

# lasagne.layers.set_all_param_values(net['real_out'], param_values)




def get_f1_acc(outputs,y_labels):

    outputs_i=outputs+0.5
    outputs_i=outputs_i.astype('int32')
    y_ilab=y_labels.astype('int32')
    gd_num=T.sum(y_ilab,axis=0)
    pr_num=T.sum(outputs_i,axis=0)
    # pr_rtm=T.eq(outputs_i,y_ilab)
    # pr_rt=T.sum(pr_rtm,axis=0)
    
    sum_ones=y_ilab+outputs_i
    pr_rtm=sum_ones/2

    # pr_rtm=T.eq(outputs_i,y_ilab)
    pr_rt=T.sum(pr_rtm,axis=0)

    #prevent nan to destroy the f1
    pr_rt=pr_rt.astype('float32')
    gd_num=gd_num.astype('float32')
    pr_num=pr_num.astype('float32')

    acc=pr_rt/outputs.shape[0]

    zero_scale=T.zeros_like(T.min(pr_rt))
    if T.eq(zero_scale,T.min(gd_num)):
        gd_num+=1
    if T.eq(zero_scale,T.min(pr_num)):
        pr_num+=1
    if T.eq(zero_scale,T.min(pr_rt)):
        pr_rt+=0.01

    recall=pr_rt/gd_num
    precision=pr_rt/pr_num
    f1=2*recall*precision/(recall+precision)
    # return T.min(pr_rt)
    return acc,f1

def multi_label_ACE(outputs,y_labels):
    data_shape=outputs.shape
    loss_buff=0
    # num=T.iscalar(data_shape[0]) #theano int to get value from tensor
    # for i in range(int(num)):
    #     for j in range(12):
    #         y_exp=outputs[i,j]
    #         y_tru=y_labels[i,0,0,j]
    #         if y_tru==0:
    #             loss_ij=math.log(1-outputs[i,j])
    #             loss_buff-=loss_ij
    #         if y_tru>0:
    #             loss_ij=math.log(outputs[i,j])
    #             loss_buff-=loss_ij
    
    # wts=[ 0.24331649,  0.18382575,  0.23082499,  0.44545567,  0.52901483,  0.58482504, \
    # 0.57321465,  0.43411294,  0.15502839,  0.36377019,  0.19050646,  0.16083916]
    # for i in [3,4,5,6,7,9]:

    for i in range(12):
        target=y_labels[:,i]
        output=outputs[:,i]
        loss_au=T.sum(-(target * T.log((output+0.05)/1.05) + (1.0 - target) * T.log((1.05 - output)/1.05)))
        loss_buff+=loss_au
    return loss_buff/(12*BATCH_SIZE)


y_lb=y_sym.reshape((y_sym.shape[0],-1))



# loss=lasagne.objectives.categorical_crossentropy(prediction,y_lb)
# loss = loss.mean()

#define loss func
loss=multi_label_ACE(prediction,y_lb)

acc_scr,f1_score=get_f1_acc(prediction,y_lb)



params = lasagne.layers.get_all_params(net['real_out'], trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)


# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
print  'compile train'
val_fn = theano.function([X_sym, y_sym], loss)
print 'complie test'
f1_fn=theano.function([X_sym,y_sym],f1_score)
print 'compile F1'

import math


# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk)==N:
            rst=chunk
            chunk=[]
            yield rst
    if chunk:
        yield chunk

# We need a fairly small batch size to fit a large network like this in GPU memory



def train_batch():
    trdata,trlb=prep_model_input(imglist)
    # trdata=trdata-MEAN_IMAGE
    return train_fn(trdata,trlb)

def test_batch():
    tsdata,tslb=prep_model_input(ixx)
    # trdata=trdata-MEAN_IMAGE
    loss=val_fn(tsdata,tslb)
    # batch_error=error_fn(tsdata,tslb)
    return loss

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])



import random
import cv2
patt=re.compile('\d+')



#adding extracting part



def build_test_model():
    T_net = {}
    T_net['input'] = InputLayer((None, 4, 224, 224))

    #slice the input to get image and feat map part
    T_net['input_map']=SliceLayer(T_net['input'],indices=slice(3,4),axis=1)
    T_net['map112']=PoolLayer(T_net['input_map'],2)
    T_net['map56']=PoolLayer(T_net['map112'],2)
    T_net['map28']=PoolLayer(T_net['map56'],2)
    T_net_buff56=[T_net['map56'] for i in range(256)]
    T_net['map56x256']=concat(T_net_buff56)
    T_net_buff28=[T_net['map28'] for i in range(512)]
    T_net['map28x512']=concat(T_net_buff28)





    T_net['input_im']=SliceLayer(T_net['input'],indices=slice(0,3),axis=1)
    T_net['conv1_1'] = ConvLayer(
        T_net['input_im'], 64, 3, pad=1, flip_filters=False)
    T_net['conv1_2'] = ConvLayer(
        T_net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    T_net['pool1'] = PoolLayer(T_net['conv1_2'], 2)
    T_net['conv2_1'] = ConvLayer(
        T_net['pool1'], 128, 3, pad=1, flip_filters=False)
    T_net['conv2_2'] = ConvLayer(
        T_net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    T_net['pool2'] = PoolLayer(T_net['conv2_2'], 2)
    T_net['conv3_1'] = ConvLayer(
        T_net['pool2'], 256, 3, pad=1, flip_filters=False)
    T_net['conv3_2'] = ConvLayer(
        T_net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    T_net['conv3_3'] = ConvLayer(
        T_net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    T_net['conv3_4'] = ConvLayer(
        T_net['conv3_3'], 256, 3, pad=1, flip_filters=False)

    T_net['conv3_map']=ElemwiseMergeLayer([T_net['conv3_1'],T_net['map56x256']],merge_function=T.mul)
    T_net['conv3_all']=ElemwiseSumLayer([T_net['conv3_4'],T_net['conv3_map']])

    T_net['pool3'] = PoolLayer(T_net['conv3_all'], 2)
    T_net['conv4_1'] = ConvLayer(
        T_net['pool3'], 512, 3, pad=1, flip_filters=False)
    T_net['conv4_2'] = ConvLayer(
        T_net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    T_net['conv4_3'] = ConvLayer(
        T_net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    T_net['conv4_4'] = ConvLayer(
        T_net['conv4_3'], 512, 3, pad=1, flip_filters=False)

    T_net['conv4_map']=ElemwiseMergeLayer([T_net['conv4_1'],T_net['map28x512']],merge_function=T.mul)
    T_net['conv4_all']=ElemwiseSumLayer([T_net['conv4_4'],T_net['conv4_map']])

    T_net['pool4'] = PoolLayer(T_net['conv4_all'], 2)
    T_net['conv5_1'] = ConvLayer(
        T_net['pool4'], 512, 3, pad=1, flip_filters=False)
    T_net['conv5_2'] = ConvLayer(
        T_net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    T_net['conv5_3'] = ConvLayer(
        T_net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    T_net['conv5_4'] = ConvLayer(
        T_net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    T_net['pool5'] = PoolLayer(T_net['conv5_4'], 2)
    T_net['fc6'] = DenseLayer(T_net['pool5'], num_units=4096)
    T_net['fc6_dropout'] = DropoutLayer(T_net['fc6'], p=0.)
    T_net['fc7'] = DenseLayer(T_net['fc6_dropout'], num_units=4096)
    T_net['fc7_dropout'] = DropoutLayer(T_net['fc7'], p=0.5)
    T_net['fc8'] = DenseLayer(T_net['fc7_dropout'], num_units=1000, nonlinearity=None)
    T_net['prob'] = NonlinearityLayer(T_net['fc8'], softmax)

    # T_net['pos_fc_layer']=DenseLayer(T_net['fc6_dropout'],num_units=2048)
    # T_net['pos_drop']=DropoutLayer(T_net['pos_fc_layer'],p=0.)
    # T_net['pred_pos_layer']=DenseLayer(T_net['pos_drop'],num_units=40,nonlinearity=sigmoid)

    #AU detection part
    T_net['au_fc_layer']=DenseLayer(T_net['fc6_dropout'],num_units=2048)
    T_net['au_drop']=DropoutLayer(T_net['au_fc_layer'],p=0.)
    T_net['output_layer']=DenseLayer(T_net['au_drop'],num_units=12,nonlinearity=sigmoid)
    # T_net['final']=concat([T_net['pred_pos_layer'],T_net['output_layer']])

    return T_net


T_net = build_test_model()

with np.load('data/ENet_dlib_300.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(T_net['output_layer'], param_values)

T_X_sym = T.tensor4()

T_feat_layer=LocalResponseNormalization2DLayer(T_net['conv4_all'],alpha=0.002, k=2, beta=0.75,)

T_prediction = lasagne.layers.get_output(T_feat_layer, T_X_sym)
T_pred = theano.function([T_X_sym], T_prediction)

conv4shape=lasagne.layers.get_output_shape(T_net['conv4_all'])
print 'conv4 shape:',conv4shape

def imdata(fls,data_size=BATCH_SIZE):
    datablob=np.ndarray((data_size,4,IM_SIZE,IM_SIZE))
    datalb=np.zeros((data_size,1,1,12))
    dataps=np.zeros((data_size,10,4))
    n=len(fls)
    random.shuffle(fls)
    fls=fls[:data_size]

    im224=np.zeros((4,IM_SIZE,IM_SIZE))
    for i,f in enumerate(fls):
        fname,flabel,fpos=f.split('->')
        pre_path='/home/wei/DATA/BP4D_FACE/'
        imi=cv2.imread(pre_path+fname)
        if imi==None:
            # print fname
            fname,flabel,fpos=fls[0].split('->')
            imi=cv2.imread(pre_path+fname)
        #cv2 read img as 3xNxN and with BGR
        if imi==None:continue
        # imi=get_face.one_big_face(imi)
        for t in range(3):
            im224[t,:,:]=cv2.resize(imi[:,:,t],(IM_SIZE,IM_SIZE))
        shape_str=fpos[1:-2]
        np_shape=np.array([float(t) for t in shape_str.split(',')])
        imshape=np.reshape(np_shape,(68,2))
        feat_map=get_attention_map_dlib.get_map(imshape,imi.shape[0],imi.shape[1])
        feat_map224=cv2.resize(feat_map,(224,224))
        im224[3,:,:]=feat_map224
        datablob[i,:,:,:]=im224

        #then the label 
        datalb[i,0,0,:]=np.array(patt.findall(flabel))
        for t in range(12):
            datalb[i,0,0,t]=min(datalb[i,0,0,t],1)

        dataps[i,:,:]=get_attention_map_dlib.get_au_tg_dlib(imshape,imi.shape[0],imi.shape[1])

    datablob=datablob.astype('float32')
    datalb=datalb.astype('float32')
    dataps=dataps.astype('float32')
    dataps/=100
    dataps*=28
    dataps=dataps.astype('int32')
    # print dataps[0,:,:]
    return datablob,datalb,dataps

def prep_model_input(fls,data_size=BATCH_SIZE):
    data,lb,pos_para=imdata(fls,data_size)
    feat_data=T_pred(data)
    new_input=np.zeros((data_size,20*512,3,3))

    for i in range(data_size):
        for j in range(10):
            p=pos_para[i,j,:]
            for t in range(4):
                if p[t]<=0:
                    p[t]=1
                if p[t]>=27:
                    p[t]=26
            try:
                new_input[i,512*2*j:512*(2*j+1),:,:]=feat_data[i,:,p[1]-1:p[1]+2,p[0]-1:p[0]+2]
                new_input[i,512*(2*j+1):512*(2*j+2),:,:]=feat_data[i,:,p[3]-1:p[3]+2,p[2]-1:p[2]+2]
            except Exception as e:
                print p
    new_input=new_input.astype('float32')
    return new_input,lb

listtrainpath='/home/wei/DATA/BP4D_faceim_tr_ag_shp.txt'
listtestpath='/home/wei/DATA/BP4D_faceim_ts_shp.txt'

fp=open(listtrainpath)
imglist=fp.readlines()

#reading test list,ixx contain all the test image names
ft=open(listtestpath)
ixx=ft.readlines()


trdata,trlb=prep_model_input(imglist)

print trdata[0,1:10,:,:],trlb[0,0,0,:]


##things to change

print 'begin training'

for epoch in range(1000):
    for batch in range(20):
        loss = train_batch()
        # print loss
    print 'epoch ',epoch, ',train loss is ', loss
    # ix = range(len(ts_lb))
    # np.random.shuffle(ix)
    if epoch%10==0:
        d1,d2=prep_model_input(ixx,250)
        f1_print=f1_fn(d1,d2)
        print 'Testing: \n', f1_print,f1_print.mean()
        # pred=pred_fn(d1,d3)
        # pred_o=pred_old_fn(d1)
        # check_feat=extract_fn(d1,d3)
        # old_feat=extract_fn2(d1)
        # print 'test sample: ',pred[0,:],d2[0,:]
        # print 'old sample: ',pred_o[0,:]
        # print check_feat[0,100:200]
        # print old_feat[0,100:200]
    loss_tot = 0.
    acc_tot = 0.
    # for chunk in batches(ixx, BATCH_SIZE):
    #     #got all the data based on index
    #     # tsdata,tslb=testbatch(chunk)
    #     loss, acc = val_fn(tsdata, tslb)
    #     # loss_tot += loss * len(chunk)
    #     # acc_tot += acc * len(chunk)
    loss = test_batch()  
    print epoch,'TEST Loss :', loss
    if (epoch+1)%200==0:
        np.savez('data/EAC_dlib_'+str(epoch+1)+'.npz', *lasagne.layers.get_all_param_values(net['real_out']))
# np.savez('model_vgg_fine.npz', *lasagne.layers.get_all_param_values(output_layer))



# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)
