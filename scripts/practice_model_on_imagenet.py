#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse, time
import numpy as np
import os
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.io import ImageRecordIter

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
mx.random.seed(1)
print("Imports Successful!")


# # Wrapper Class for DataIter and ImageRecordIter
# Because ImageRecordIter works with DataIter types and is not well suited for Gluon training loops, create this wrapper class so that a data loader can be used properly with the training loop...
# We'll see if this works or not.

# In[ ]:


class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter
    
    def __iter__(self):
        self.data_iter.reset()
        return self
    
    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label
    
    def next(self):
        return self.__next__() # For Python 2


# In[ ]:


num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]

# Get the model ResNet50_v2 with 10 output classes
net = get_model('ResNet50_v2', classes=1000)
#net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
net.initialize(mx.init.Xavier(), ctx=ctx)
print("Net Init Complete.")


# In[ ]:


jitter_param = 0.4
lighting_param = 0.1
mean_rgb = [123.68, 116.779, 103.939]
std_rgb = [58.393, 57.12, 57.375]

train_data = mx.io.ImageRecordIter(
    path_imgrec = '/newvolume/ImageNet-ILSVRC2012/training/train_rec.rec',
    path_imgidx = '/newvolume/ImageNet-ILSVRC2012/training/train_rec.idx',
    preprocess_threads  = 32,
    shuffle             = True,
    batch_size          = 64,

    data_shape          = (3, 224, 224),
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
    rand_mirror         = True,
    random_resized_crop = True,
    max_aspect_ratio    = 4. / 3.,
    min_aspect_ratio    = 3. / 4.,
    max_random_area     = 1,
    min_random_area     = 0.08,
    brightness          = jitter_param,
    saturation          = jitter_param,
    contrast            = jitter_param,
    pca_noise           = lighting_param,
)

val_data = mx.io.ImageRecordIter(
    path_imgrec = '/newvolume/ImageNet-ILSVRC2012/testing/val_rec.rec',
    path_imgidx = '/newvolume/ImageNet-ILSVRC2012/testing/val_rec.idx',
    preprocess_threads  = 32,
    shuffle             = False,
    batch_size          = 64,

    resize              = 256,
    data_shape          = (3, 224, 224),
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
)
print("RecordIter Setup Done.")


# ## Optimizer, Loss, and Metric

# In[ ]:


lr_decay = 0.10 # Learning rate should be divided by 10 (1/10).
lr_decay_epoch = [30, 60, 90, np.inf]
optimizer = 'nag'
optimizer_params = {'learning_rate': 0.1, 'wd':0.0001, 'momentum':0.9}
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)


# In[ ]:


loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


# With 1000 classes the model may not always rate the correct answer with the highest rank. Besides top-1 accuracy, we want top-5 accuracy for how well the model is doing. At the end of every epoch, we reocrd and print the metric scores.

# In[ ]:


acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_history = TrainingHistory(['training-top1-err', 'training-top5-err',
                                 'validation-top1-err', 'validation-top5-err'])


# In[ ]:


def test(ctx, val_data_loader):
    acc_top1_val = mx.metric.Accuracy()
    acc_top5_val = mx.metric.TopKAccuracy(5)
    for i, batch in enumerate(val_data_loader):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        acc_top1_val.update(label, outputs)
        acc_top5_val.update(label, outputs)

    _, top1 = acc_top1_val.get()
    _, top5 = acc_top5_val.get()
    return (1 - top1, 1 - top5)


# In[ ]:


epochs = 120
lr_decay_count = 0
log_interval = 50
batch_size = 64

print(type(train_data))
print(type(val_data))
print("Creating Data Loaders")
train_data_loader = DataIterLoader(train_data)
val_data_loader = DataIterLoader(val_data)
print(type(train_data_loader))
print(type(val_data_loader))

print("Training Loop Started:")
for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    acc_top1.reset()
    acc_top5.reset()
    train_loss = 0

    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    for i, batch in enumerate(train_data_loader):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        
#         if i == 0:
#             print(type(data))
#             print(type(label))
        
        # Autograd
        with ag.record():
            outputs = [net(X) for X in data]
            loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]
        
        # Backpropagation
        #ag.backward(loss)
        for l in loss:
            l.backward()
        
        # Optimize
        trainer.step(batch_size)
        
        # Update metrics
        train_loss += sum([l.sum().asscalar() for l in loss])
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
        
        if log_interval and not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            err_top1, err_top5 = (1-top1, 1-top5)
            print('Epoch[%d] Batch [%d]:\tSpeed: %f samples/sec\ttop1-err=%f\ttop5-err=%f\ttrain_loss=%f' %
                  (epoch, i, batch_size*log_interval/(time.time()-btic), err_top1, err_top5, train_loss))
            btic = time.time()

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    err_top1, err_top5 = (1-top1, 1-top5)

    err_top1_val, err_top5_val = test(ctx, val_data_loader)
    train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])

    print('[Epoch %d] train_top5=%f train_top1=%f val_top5=%f val_top1=%f loss=%f time: %f' % 
             (epoch, top5, top1, (1-err_top5_val), (1-err_top1_val), train_loss, time.time()-tic))
    #print('[Epoch %d] training: err-top1=%f err-top5=%f'%(epoch, err_top1, err_top5))
    #print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
    #print('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

print("Training Phase Done.")


# In[ ]:


# https://cv.gluon.ai/api/utils.html
train_history.plot(['training-top1-err', 'validation-top1-err'], save_path="/home/ubuntu/")
print("Image Plot Saved!")

