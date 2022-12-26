# Copyright 2021 Pengcheng Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import ast

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from config import config as cfg
from dataset import create_dataset
from dice_metric import DiceMetric
from loss import SoftmaxCrossEntropyWithLogits
from lr_schedule import dynamic_lr
from suwen.algorithm.nets.unet3d import UNet3D
from suwen.engine import Engine

mindspore.set_seed(1)

def train_net(data_dir,
              seg_dir,
              run_distribute,
              config=None):
    if run_distribute:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode = parallel_mode,
                                          device_num = rank_size,
                                          gradients_mean = True)
    else:
        rank_id = 0
        rank_size = 1

    net = UNet3D(n_channels = config.in_channels, n_classes = config.num_classes)

    train_dataset = create_dataset(data_path = data_dir, seg_path = seg_dir, config = config, \
                                   rank_size = rank_size, rank_id = rank_id, is_training = True)
    eval_dataset = create_dataset(data_path = data_dir, seg_path = seg_dir, config = config, is_training = False)
    dataset_steps = train_dataset.get_dataset_size()
    new_epochs = config.epoch_size
    print("train dataset length is:", dataset_steps)

    criterion = SoftmaxCrossEntropyWithLogits()

    lr = Tensor(dynamic_lr(config, dataset_steps), mstype.float32)
    optimizer = nn.Adam(params = net.trainable_params(), learning_rate = lr)

    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update = False)

    mine_train = Engine(net,
                        loss_fn = criterion,
                        optimizer = optimizer,
                        amp_level = "O3",
                        metrics = {'dice_loss': DiceMetric(config.num_classes)},
                        loss_scale_manager = loss_scale_manager)

    save_checkpoint_config = {'max_num': args.keep_ckpt_max,
                              'prefix': 'unet3d',
                              "root": args.ckpt_path}

    mine_train.train(new_epochs, train_dataset = train_dataset,
                     sink_size = dataset_steps, save_checkpoints = save_checkpoint_config)

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet3D on images and target masks')
    parser.add_argument('--device_id', type = int, default = 0, help = 'device id')
    parser.add_argument('--data_path', dest = 'data_path', type = str, default = './data/MM-WHS', help = 'image data directory')
    parser.add_argument('--seg_path', dest = 'seg_path', type = str, default = './data/MM-WHS/seg/', help = 'seg data directory')
    parser.add_argument('--run_distribute', dest = 'run_distribute', type = ast.literal_eval, default = False, \
                        help = 'Run distribute, default: false')
    parser.add_argument('--keep_ckpt_max', type = int, default = 1,
                        help = 'max number of saving checkpoint, default: 1')
    parser.add_argument('--ckpt_path', type = str, default = './ckpt', help = 'save checkpoint path')
    parser.add_argument('--load_ckpt_path', type = str, default = '', help = 'load checkpoint path')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train_net(data_dir = args.data_path,
              seg_dir = args.seg_path,
              run_distribute = args.run_distribute,
              config = cfg)
