#run  THEANO_FLAGS="device=gpu0,floatX=float32" python BP4D_ENet_dlib_train.py

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

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax,sigmoid
from lasagne.utils import floatX
from lasagne.layers import SliceLayer, concat,BatchNormLayer,ElemwiseSumLayer,ElemwiseMergeLayer
from lasagne.objectives import squared_error

import get_bp4d_2dfeat
import get_attention_map_dlib
import get_face

IM_SIZE=224

def build_model():
    net = {}
    net['input'] = InputLayer((None, 4, 224, 224))

    #slice the input to get image and feat map part
    net['input_map']=SliceLayer(net['input'],indices=slice(3,4),axis=1)
    net['map112']=PoolLayer(net['input_map'],2)
    net['map56']=PoolLayer(net['map112'],2)
    net['map28']=PoolLayer(net['map56'],2)
    net_buff56=[net['map56'] for i in range(256)]
    net['map56x256']=concat(net_buff56)
    net_buff28=[net['map28'] for i in range(512)]
    net['map28x512']=concat(net_buff28)





    net['input_im']=SliceLayer(net['input'],indices=slice(0,3),axis=1)
    net['conv1_1'] = ConvLayer(
        net['input_im'], 64, 3, pad=1, flip_filters=False,trainable=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False,trainable=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False,trainable=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False,trainable=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(
        net['conv3_3'], 256, 3, pad=1, flip_filters=False)

    net['conv3_map']=ElemwiseMergeLayer([net['conv3_1'],net['map56x256']],merge_function=T.mul)
    net['conv3_all']=ElemwiseSumLayer([net['conv3_4'],net['conv3_map']])

    net['pool3'] = PoolLayer(net['conv3_all'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(
        net['conv4_3'], 512, 3, pad=1, flip_filters=False)

    net['conv4_map']=ElemwiseMergeLayer([net['conv4_1'],net['map28x512']],merge_function=T.mul)
    net['conv4_all']=ElemwiseSumLayer([net['conv4_4'],net['conv4_map']])

    net['pool4'] = PoolLayer(net['conv4_all'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(
        net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)


    return net




# Load model weights and metadata
d = pickle.load(open('vgg19.pkl'))

#setting batch size for the whole training
BATCH_SIZE = 50
# MEAN_IMAGE=np.load('vgg_mean.npy')


# Build the network and fill with pretrained weights
net = build_model()

lasagne.layers.set_all_param_values(net['conv3_4'], d['param values'][0:8],trainable=False)
lasagne.layers.set_all_param_values(net['prob'], d['param values'][8:],trainable=True)

#regression part
# pos_fc_layer=DenseLayer(net['fc6_dropout'],num_units=2048)
# pos_drop=DropoutLayer(pos_fc_layer,p=0.25)
# pred_pos_layer=DenseLayer(pos_drop,num_units=40,nonlinearity=sigmoid)

#AU detection part
au_fc_layer=DenseLayer(net['fc6_dropout'],num_units=2048)
au_drop=DropoutLayer(au_fc_layer,p=0.5)
output_layer=DenseLayer(au_drop,num_units=12,nonlinearity=sigmoid)
# final_concat=concat([pred_pos_layer,output_layer])



# BATCH_SIZE = 50
# MEAN_IMAGE=np.load('vgg_mean.npy')

# with np.load('data/model_vgg16_CIFE_test_300.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]



# lasagne.layers.set_all_param_values(net['conv5_1'], param_values[0:22],trainable=False)
# lasagne.layers.set_all_param_values(net['prob'], param_values[22:32],trainable=True)
print 'successfully...'




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
    return loss_buff/(600)
# def regress_landmarks_pts(preds,pts):



# def get_f1_acc(outputs,y_labels):
#     outputs+=0.5
#     outputs=outputs.astype('int8')
#     y_labels=y_labels.astype('int8')
#     acc=np.array((12,))
#     for i in range(12):
#         cnt=0
#         for j in range(outputs.shape[0]):
#             if outputs[j][i]==y_labels[j][i]:
#                 cnt+=1
#         acc[i]=cnt/outputs.shape[0]
#     return acc

# import h5py
# f=h5py.File('test.h5','r')
# ##get the dataset
# tr_data=f['tr_data']
# tr_lb=f['tr_lb']
# ts_data=f['ts_data']
# ts_lb=f['ts_lb']

# MEAN_IMAGE=np.load('vgg_mean.npy')
# #error occur they need to be dtype float32
# #error occur they need to be dtype float32
# tr_data=tr_data[:,:,16:240,16:240].astype('float32')
# ts_data=ts_data[:,:,16:240,16:240].astype('float32')
# tr_data=tr_data-MEAN_IMAGE
# ts_data=ts_data-MEAN_IMAGE
# tr_lb=tr_lb[:].astype('int32')
# ts_lb=ts_lb[:].astype('int32')
# print tr_data.shape

##begin to train the model 

# ##first just configure
# half_feature_layer=DenseLayer(net['fc6'],num_units=2048)
# feat_dropout=DropoutLayer(half_feature_layer,p=0.5,rescale=False)
# output_layer=DenseLayer(feat_dropout,num_units=12,nonlinearity=sigmoid)
# final_prob=NonlinearityLayer(output_layer, softmax)

# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.tensor4()
p_sym=T.tensor3()

prediction = lasagne.layers.get_output(output_layer, X_sym)
# loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
# 
# pred_pos_raw=lasagne.layers.get_output(pred_pos_layer, X_sym)
# pred_pos=pred_pos_raw.reshape(((pred_pos_raw.shape[0],10,4))) #same shape to target

y_lb=y_sym.reshape((y_sym.shape[0],-1))
# loss=lasagne.objectives.categorical_crossentropy(prediction,y_lb)
# loss = loss.mean()
loss=multi_label_ACE(prediction,y_lb)
# loss2=squared_error(pred_pos,p_sym)
# loss2=loss2.mean()
# loss=(loss1+loss2*5)/2
# loss_print=[loss1,loss2,loss]
# error=get_f1_acc(prediction,y_lb)
# acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
#                       dtype=theano.config.floatX)
last_conv_shape=lasagne.layers.get_output_shape(net['conv5_4'])
print last_conv_shape
params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)


# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
val_fn = theano.function([X_sym, y_sym], loss)
# error_fn=theano.function([X_sym, y_sym],error)
pred_fn = theano.function([X_sym], prediction)
# local_fn=theano.function([X_sym], pred_pos)
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
    trdata,trlb,trpos=imdata(imglist)
    # trdata=trdata-MEAN_IMAGE
    return train_fn(trdata,trlb)

def test_batch():
    tsdata,tslb,tspos=imdata(ixx)
    # trdata=trdata-MEAN_IMAGE
    loss=val_fn(tsdata,tslb)
    # batch_error=error_fn(tsdata,tslb)
    return loss

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])


###read the emotion dataset make them as numpy array


#load the files first then deliver thedata when needed.

import random
import cv2
patt=re.compile('\d+')
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



listtrainpath='/home/wei/DATA/BP4D_faceim_tr_ag_shp.txt'
listtestpath='/home/wei/DATA/BP4D_faceim_ts_shp.txt'
# impath='/home/wei/caffedata/webim/webemo_trag'
# testpath='/home/wei/caffedata/webim/CIFEv2.0/webemo_ts'

fp=open(listtrainpath)
imglist=fp.readlines()
print len(imglist)
#reading test list,ixx contain all the test image names
ft=open(listtestpath)
ixx=ft.readlines()


trdata,trlb,trps=imdata(imglist)

print trdata[10,:,:,:],trlb[10,0,0,:],trps[10,:,:]


##things to change

print 'begin training'

for epoch in range(300):
    for batch in range(20):
        loss = train_batch()
        # print loss
    print 'epoch ',epoch, ',train loss is ', loss

    # ix = range(len(ts_lb))
    # np.random.shuffle(ix)
    
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
    if (epoch+1)%100==0:
        np.savez('data/ENet_dlib_'+str(epoch+1)+'.npz', *lasagne.layers.get_all_param_values(output_layer))
# np.savez('model_vgg_fine.npz', *lasagne.layers.get_all_param_values(output_layer))



# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)
