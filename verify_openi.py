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
import mindspore.dataset.vision as transforms
import mindspore.dataset.vision as vision
from mindspore import nn, ops

IMAGESIZE = 672

parser = argparse.ArgumentParser()
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--data_path', default="/cache/data", type=str, help='data path')

parser.add_argument('--data_url', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--train_url', metavar='DIR', default='', help='save output')
parser.add_argument('--result_url', metavar='DIR', default='', help='save result url')
parser.add_argument('--multi_data_url',help='path to multi dataset', default= '/cache/data/')
parser.add_argument('--use_qizhi', type=bool, default=False,help='use qizhi')
parser.add_argument('--use_zhisuan', type=bool, default=True, help='use zhisuan')
parser.add_argument('--ckpt_url', type=str, default=None,help='load ckpt file path')
parser.add_argument('--ckpt_path', type=str, default='/cache/pretrain/',help='load ckpt file path')
parser.add_argument('--pretrain_url', type=str, default=None, help='load ckpt file path')

args = parser.parse_args()

data_path = args.data_path

if args.use_qizhi:
    from openi import openi_multidataset_to_env as DatasetToEnv  
    from openi import pretrain_to_env as PretrainToEnv
    from openi import env_to_openi as EnvToOpeni

    data_dir = '/cache/data'  
    train_dir = '/cache/output'
    pretrain_dir = '/cache/pretrain'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)
    DatasetToEnv(args.multi_data_url,data_dir)
    PretrainToEnv(args.pretrain_url,pretrain_dir)


if args.use_zhisuan:
    from openi import c2net_multidataset_to_env as DatasetToEnv  
    from openi import pretrain_to_env as PretrainToEnv
    from openi import env_to_openi as EnvToOpeni

    data_dir = '/cache/data'  
    train_dir = '/cache/output'
    pretrain_dir = '/cache/pretrain'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)
    DatasetToEnv(args.multi_data_url,data_dir)
    PretrainToEnv(args.pretrain_url,pretrain_dir)

def datapipe(dataset, batch_size):
    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]
    image_transforms = [
        vision.RandomCropDecodeResize(size=IMAGESIZE,
                                        scale=(0.08, 1.0),
                                        ratio=(0.75, 1.333)),
        vision.RandomHorizontalFlip(prob=0.5),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.batch(batch_size)
    return dataset

data_path = args.data_dir

dataset_train = ImageFolderDataset(dataset_dir=os.path.join(data_path, "train"),
                                class_indexing={"falciparum":0, "uninfected":1,"vivax":2},
                                extensions=[".tiff", ".jpg"],
                                shuffle=True)


dataset_train = datapipe(dataset_train,1)

class Attention(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = ms.Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(p=1.0-attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(p=1.0-keep_prob)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct."""
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)
        attn = self.q_matmul_k(q, k)
        attn = ops.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = self.attn_matmul_v(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out

from typing import Optional, Dict

class FeedForward(nn.Cell):
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0):
        super(FeedForward, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(p=1.0-keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x

class ResidualCell(nn.Cell):
    def __init__(self, cell):
        super(ResidualCell, self).__init__()
        self.cell = cell

    def construct(self, x):
        """ResidualCell construct."""
        return self.cell(x) + x

class TransformerEncoder(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(TransformerEncoder, self).__init__()
        layers = []

        for _ in range(num_layers):
            normalization1 = norm((dim,))
            normalization2 = norm((dim,))
            attention = Attention(dim=dim,
                                  num_heads=num_heads,
                                  keep_prob=keep_prob,
                                  attention_keep_prob=attention_keep_prob)

            feedforward = FeedForward(in_features=dim,
                                      hidden_features=mlp_dim,
                                      activation=activation,
                                      keep_prob=keep_prob)

            layers.append(
                nn.SequentialCell([
                    ResidualCell(nn.SequentialCell([normalization1, attention])),
                    ResidualCell(nn.SequentialCell([normalization2, feedforward]))
                ])
            )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Transformer construct."""
        return self.layers(x)

class PatchEmbedding(nn.Cell):
    MIN_NUM_PATCHES = 4

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 input_channels: int = 3):
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)

    def construct(self, x):
        """Path Embedding construct."""
        x = self.conv(x)
        b, c, h, w = x.shape
        x = ops.reshape(x, (b, c, h * w))
        x = ops.transpose(x, (0, 2, 1))

        return x


from mindspore.common.initializer import Normal
from mindspore.common.initializer import initializer
from mindspore import Parameter


def init(init_type, shape, dtype, name, requires_grad):
    """Init."""
    initial = initializer(init_type, shape, dtype).init_data()
    return Parameter(initial, name=name, requires_grad=requires_grad)

class ViT(nn.Cell):
    def __init__(self,
                 image_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: Optional[nn.Cell] = nn.LayerNorm,
                 pool: str = 'cls') -> None:
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              input_channels=input_channels)
        num_patches = self.patch_embedding.num_patches

        self.cls_token = init(init_type=Normal(sigma=1.0),
                              shape=(1, 1, embed_dim),
                              dtype=ms.float32,
                              name='cls',
                              requires_grad=True)

        self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                  shape=(1, num_patches + 1, embed_dim),
                                  dtype=ms.float32,
                                  name='pos_embedding',
                                  requires_grad=True)

        self.pool = pool
        self.pos_dropout = nn.Dropout(p=1.0-keep_prob)
        self.norm = norm((embed_dim,))
        self.transformer = TransformerEncoder(dim=embed_dim,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              mlp_dim=mlp_dim,
                                              keep_prob=keep_prob,
                                              attention_keep_prob=attention_keep_prob,
                                              drop_path_keep_prob=drop_path_keep_prob,
                                              activation=activation,
                                              norm=norm)
        self.dropout = nn.Dropout(p=1.0-keep_prob)
        self.dense = nn.Dense(embed_dim, num_classes)

    def construct(self, x):
        """ViT construct."""
        x = self.patch_embedding(x)
        cls_tokens = ops.tile(self.cls_token.astype(x.dtype), (x.shape[0], 1, 1))
        x = ops.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding

        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]
        if self.training:
            x = self.dropout(x)
        x = self.dense(x)

        return x

from mindspore.nn import LossBase
from mindspore.train import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import train
#from download import download
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
# define loss function
class CrossEntropySmooth(LossBase):
    """CrossEntropy."""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss

network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=num_classes)
# set checkpoint
ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=100)
ckpt_callback = ModelCheckpoint(prefix='vit_b_16', directory=train_dir, config=ckpt_config)

# initialize model
# "Ascend + mixed precision" can improve performance
ascend_target = (ms.get_context("device_target") == "Ascend")
if ascend_target:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O2")
else:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O0")


# train model
#model.train(epoch_size,
#            dataset_train,callbacks=[TimeMonitor(), LossMonitor()])



############################################################
#verify

dataset_val = ImageFolderDataset(os.path.join(data_path, "test"), shuffle=True)

mean = [0.485*255, 0.456*255, 0.406*255]
std = [0.229*255, 0.224*255, 0.225*255]

MAGESIZE = 672

trans_val = [
    transforms.Decode(),
    transforms.Resize(MAGESIZE + 32),
    transforms.CenterCrop(MAGESIZE),
    transforms.Normalize(mean=mean, std=std),
    transforms.HWC2CHW()
]

dataset_val = dataset_val.map(operations=trans_val, input_columns=["image"])
dataset_val = dataset_val.batch(batch_size=1, drop_remainder=True)

# construct model
#network = ViT()

#MAGESIZE = 672
#MAGESIZE = 1200
network = ViT(image_size=IMAGESIZE)



# load ckpt
vit_path = '/home/together/ai/Vit/vit_672_4832_0.801.ckpt'
param_dict = ms.load_checkpoint(vit_path)
ms.load_param_into_net(network, param_dict)

network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=num_classes)

# define metric
eval_metrics = {'Top_1_Accuracy': train.Top1CategoricalAccuracy(),
                'Top_5_Accuracy': train.Top5CategoricalAccuracy()}

if ascend_target:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=eval_metrics, amp_level="O2")
else:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=eval_metrics, amp_level="O0")

# evaluate model
result = model.eval(dataset_val)
print(result)

if args.use_qizhi:
    EnvToOpeni(train_dir,args.train_url)