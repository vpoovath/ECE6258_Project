#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory, LRSequential, LRScheduler
from gluoncv.data import transforms as gcv_transforms
print("Imports successful")


# In[ ]:


# number of GPUs to use
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]

net = get_model('cifar_resnet56_v1', classes=100)
net.initialize(mx.init.Xavier(), ctx = ctx)
print("Model Init Done.")


# ## Smoothing

# In[ ]:


label_smoothing = True
def smooth(label, num_classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        res = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        smoothed.append(res)
    return smoothed


# ## Mixup

# In[ ]:


mixup = True
def mixup_transform(label, num_classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(num_classes, on_value = 1 - eta + eta/num_classes, off_value = eta/classes)
        y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/num_classes, off_values = eta/classes)
        res.append(lam*y1 + (1-lam)*y2)
    return res


# In[ ]:


resize = 32
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
max_aspect_ratio = 4.0 / 3.0
min_aspect_ratio = 3.0 / 4.0
max_random_area = 1
min_random_area = 0.08
jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([    
#     transforms.RandomResizedCrop(resize,
#                                  scale=(min_random_area, max_random_area), 
#                                  ratio=(min_aspect_ratio, max_aspect_ratio)),
    
        # Randomly flip the image horizontally
    transforms.RandomFlipLeftRight(),
    
    transforms.RandomBrightness(brightness=jitter_param),
    transforms.RandomSaturation(saturation=jitter_param),
    transforms.RandomHue(hue=jitter_param),
    
    transforms.RandomLighting(lighting_param),
    
    # Randomly crop an area and resize it to be 32x32, then pad it to be 40x40
    gcv_transforms.RandomCrop(32, pad=4),
        
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
    
    # Normalize the image with mean and standard deviation calculated across all images
    transforms.Normalize(mean_rgb, std_rgb),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_rgb, std_rgb),
])
print("Preprocessing Step Successful.")


# In[ ]:


# Batch Size for Each GPU
per_device_batch_size = 128

# Number of data loader workers
num_workers = 2

# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

# For CIFAR100 Dataset:
num_classes = 100
num_images_per_class = 500
num_training_samples = num_classes * num_images_per_class
num_batches = num_training_samples // batch_size

# Set train=True for training data
# Set shuffle=True to shuffle the training data
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR100(train=True).transform_first(transform_train),
    batch_size=batch_size, 
    shuffle=True, 
    last_batch='discard', 
    num_workers=num_workers)

# Set train=False for validation data
# Set shuffle=False to shuffle the testing data
val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR100(train=False).transform_first(transform_test),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)
print("Initialization of train_data and val_data successful.")


# In[ ]:


epochs = 120
warmup_epochs = 10

# Learning rate decay factor
lr_decay = 0.1

# Epochs where learning rate decays
lr_decay_epoch = [30, 60, 90, np.inf]

# Sets up a linear warmup scheduler, followed by a cosine rate decay.
# Consult the paper for the proper parameters (base_lr, target_lr, warmup_epochs, etc.)
lr_scheduler = LRSequential([
    LRScheduler('linear',
                base_lr = 0,
                target_lr = 0.1,
                nepochs = warmup_epochs,
                iters_per_epoch = num_batches),
    
    LRScheduler('cosine',
                base_lr = 0.1,
                target_lr = 0,
                nepochs = epochs - warmup_epochs,
                iters_per_epoch = num_batches,
                step_epoch = lr_decay_epoch,
                step_factor = lr_decay,
                power = 2)
])

# Nesterov accelerated gradient descent and set parameters (based of off 
# reference papers and default values):
optimizer = 'nag'
optimizer_params = {'lr_scheduler': lr_scheduler, 'wd': 0.0001, 'momentum': 0.9}

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-error', 'validation-error'])
train_history2 = TrainingHistory(['training-acc', 'val-acc-top1', 'val-acc-top5'])


# In[ ]:


acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy()

def test(ctx, val_data):
    #metric = mx.metric.Accuracy()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        #metric.update(label, outputs)
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    #return metric.get()
    return (top1, top5)


# In[ ]:


lr_decay_count = 0
epochs = 120
train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-error', 'validation-error'])
train_history2 = TrainingHistory(['training-acc', 'val-acc-top1', 'val-acc-top5'])

print("Training loop started:")
for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0

    # Learning rate decay
    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        
        if mixup:
            # What is parameters of the beta distribution used?
            lam = np.random.beta(-inf, inf)
            if epoch >= epochs - mixup_off_epochs:
                lam = 1
            data = [lam*X + (1-lam)*X[::-1] for X in data]
            
            if label_smoothing:
                eta = 0.1
            else:
                eta = 0.0
            label = mixup_transform(label, num_classes, lam, eta)
        
        if label_smoothing:
            hard_label = label
            label = smooth(label, num_classes)

        # AutoGrad
        with ag.record():
            output = [net(X) for X in data]
            
            # TODO: Add distillation condition here
            
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()
        
        train_loss += sum([l.sum().asscalar() for l in loss])
            
        # Optimize
        trainer.step(batch_size)
        
        # Update metrics
        if mixup:
            output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False))                               for out in outputs]
            train_metric.update(label, output_softmax)
        else:
            if label_smoothing:
                train_metric.update(hard_label, outputs)
            else:
                train_metric.update(label, output)

    name, acc = train_metric.get()
    
    # Evaluate on Validation data
    #name, val_acc = test(ctx, val_data)
    val_acc_top1, val_acc_top5 = test(ctx, val_data)

    # Update history and print metrics
    train_history.update([1-acc, 1-val_acc_top1])
    train_history2.update([acc, val_acc_top1, val_acc_top5])
    
    print('[Epoch %d] train=%f val_top1=%f val_top5=%f loss=%f time: %f' %
        (epoch, acc, val_acc_top1, val_acc_top5, train_loss, time.time()-tic))

# We can plot the metric scores with:
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_history.plot(['training-error', 'validation-error'], 
                    save_path="./cifar100_resnet56_v1_nag_errors_{}.png".format(timestamp))
train_history2.plot(['training-acc', 'val-acc-top1', 'val-acc-top5'],
                     save_path="./cifar100_resnet56_v1_nag_accuracies_{}.png".format(timestamp))
print("Done.")
