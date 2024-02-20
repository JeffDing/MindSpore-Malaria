# The MIT license:

# Copyright <2023> <lisenlingood、热带鱼>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#  and associated documentation files (the “Software”), to deal in the Software without restric
# tion, including without limitation the rights to use, copy, modify, merge, publish, distribute, sub
# license, and/or sell copies of the Software, and to permit persons to whom the Software is furni
# shed to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantia
# l portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED
# , INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PART
# ICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT H
# OLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF C
# ONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWA
# RE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os

import mindspore as ms
from mindspore.dataset import ImageFolderDataset
from mindspore import nn

from Vit import *

data_path = './data/'
dataset_train = ImageFolderDataset(dataset_dir=os.path.join(data_path, "train"),
                                class_indexing={"falciparum":0, "uninfected":1,"vivax":2},
                                extensions=[".tiff", ".jpg"],
                                shuffle=True)
dataset_eval = ImageFolderDataset(dataset_dir=os.path.join(data_path, "test"),
                                class_indexing={"falciparum":0, "uninfected":1,"vivax":2},
                                extensions=[".tiff", ".jpg"],
                                shuffle=True)

dataset_train = datapipe(dataset_train,1)
dataset_eval = datapipe(dataset_eval,1)

# define super parameter
epoch_size = 10
momentum = 0.9
num_classes = 1000
step_size = dataset_train.get_dataset_size()

# construct model
network = ViT(image_size=IMAGESIZE)

lr = nn.cosine_decay_lr(min_lr=float(0),
                        max_lr=0.00005,
                        total_step=epoch_size * step_size,
                        step_per_epoch=step_size,
                        decay_epoch=10)

# define optimizer
network_opt = nn.Adam(network.trainable_params(), lr, momentum)

network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=num_classes)
# set checkpoint
ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=100)
ckpt_callback = ModelCheckpoint(prefix='vit_b_16', directory='./ViT', config=ckpt_config)

# initialize model
# "Ascend + mixed precision" can improve performance
ascend_target = (ms.get_context("device_target") == "Ascend")
if ascend_target:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O2")
else:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O0")

if __name__ == '__main__':
    # train model
    model.fit(epoch_size,dataset_train,dataset_eval
              ,callbacks=[TimeMonitor(), LossMonitor()])