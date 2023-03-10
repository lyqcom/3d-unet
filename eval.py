#
#Copyright 2021 Pengcheng Laboratory
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

import numpy as np
from mindspore import Model, context

from config import config as cfg
from dataset import create_dataset
from utils import create_sliding_window, CalculateDice
from suwen.algorithm.nets import UNet3D
from suwen.data import swtype, Tensor
from suwen.utils import load_checkpoint, load_param_into_net
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description = 'Test the UNet3D on images and target masks')
    parser.add_argument('--device_id', type = int, default = 0, help = 'device id')
    parser.add_argument('--data_path', dest = 'data_path', type = str, default = '', help = 'image data directory')
    parser.add_argument('--seg_url', dest = 'seg_url', type = str, default = '', help = 'seg data directory')
    parser.add_argument('--ckpt_path', dest = 'ckpt_path', type = str, default = '', help = 'checkpoint path')
    return parser.parse_args()

def test_net(data_dir, seg_dir, device_id, ckpt_path, config=None):
    context.set_context(mode = context.GRAPH_MODE, device_target = "Ascend", save_graphs = False, device_id = device_id)
    eval_dataset = create_dataset(data_path = data_dir, seg_path = seg_dir, config = config, is_training = False)
    eval_data_size = eval_dataset.get_dataset_size()
    print("train dataset length is:", eval_data_size)

    network = UNet3D(n_channels = config.in_channels, n_classes = config.num_classes)
    network.set_train(False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    model = Model(network)
    index = 0
    total_dice = 0
    for batch in eval_dataset.create_dict_iterator(num_epochs = 1, output_numpy = True):
        image = batch["image"]
        seg = batch["seg"]
        print("current image shape is {}".format(image.shape), flush = True)
        sliding_window_list, slice_list = create_sliding_window(image, config.roi_size, config.overlap)
        image_size = (config.batch_size, config.num_classes) + image.shape[2:]
        output_image = np.zeros(image_size, np.float32)
        count_map = np.zeros(image_size, np.float32)
        importance_map = np.ones(config.roi_size, np.float32)
        for window, slice_ in zip(sliding_window_list, slice_list):
            window_image = Tensor(window, swtype.float32)
            pred_probs = model.predict(window_image)
            output_image[slice_] += pred_probs.asnumpy()
            count_map[slice_] += importance_map
        output_image = output_image / count_map
        print(np.sum(seg))
        dice, _ = CalculateDice(output_image, seg)
        print("The {} batch dice is {}".format(index, dice), flush = True)
        total_dice += dice
        index = index + 1
    avg_dice = total_dice / eval_data_size
    print("**********************End Eval***************************************")
    print("eval average dice is {}".format(avg_dice))

if __name__ == '__main__':
    args = get_args()
    print("Testing setting:", args)
    test_net(data_dir = args.data_path,
             seg_dir = args.seg_url,
             device_id = args.device_id,
             ckpt_path = args.ckpt_path,
             config = cfg)
