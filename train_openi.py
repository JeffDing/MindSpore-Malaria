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
import argparse

import mindspore as ms
from mindspore.dataset import ImageFolderDataset
from mindspore import nn

from Vit import *

parser = argparse.ArgumentParser()
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--data_path', default="/cache/data", type=str, help='data path')
parser.add_argument("--epochs", default=10, type=int, help="epochs")

parser.add_argument('--data_url', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--train_url', metavar='DIR', default='', help='save output')
parser.add_argument('--result_url', metavar='DIR', default='', help='save result url')
parser.add_argument('--multi_data_url',help='path to multi dataset', default= '/cache/data/')
parser.add_argument('--ckpt_url', type=str, default=None,help='load ckpt file path')
parser.add_argument('--ckpt_path', type=str, default='/cache/pretrain/',help='load ckpt file path')
parser.add_argument('--pretrain_url', type=str, default=None, help='load ckpt file path')
parser.add_argument('--model_url', type=str, default=None, help='load ckpt file path')
parser.add_argument('--use_qizhi', type=bool, default=False,help='use qizhi')
parser.add_argument('--use_zhisuan', type=bool, default=True, help='use zhisuan')

args = parser.parse_args()

data_path = args.data_path

if args.use_qizhi:
    from openi import openi_multidataset_to_env as DatasetToEnv  
    from openi import env_to_openi as EnvToOpeni

    data_dir = '/cache/data'  
    train_dir = '/cache/output'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    DatasetToEnv(args.multi_data_url,data_dir)


if args.use_zhisuan:
    from openi import c2net_multidataset_to_env as DatasetToEnv  
    from openi import env_to_openi as EnvToOpeni

    data_dir = '/cache/data'  
    train_dir = '/cache/output'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    DatasetToEnv(args.multi_data_url,data_dir)

ms.set_context(device_target=args.device_target) 

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
epoch_size = args.epochs
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
ckpt_callback = ModelCheckpoint(prefix='vit_b_16', directory=train_dir, config=ckpt_config)
callback=[TimeMonitor(), LossMonitor()]
callback.append(ckpt_callback) 

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
              ,callbacks=callback)

    if args.use_qizhi:
        EnvToOpeni(train_dir,args.train_url)