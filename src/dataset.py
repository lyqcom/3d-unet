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

import os
import glob
import numpy as np

import suwen.data.dataset as ds
from config import config as cfg
from suwen.transforms import Compose
from transform import Dataset, ExpandChannel, LoadData, Orientation, ScaleIntensityRange, RandomCropSamples, OneHot

class ConvertLabel:
    """
    """

    def operation(self, data):
        """
        """
        origin_labels = [205., 420., 421., 500., 550., 600., 820., 850]

        for new_label, ori_label in enumerate(origin_labels):
            data[data == ori_label] = new_label

        return data

    def __call__(self, image, label):
        label = self.operation(label)
        return image, label


def create_dataset(data_path, seg_path, config, rank_size=1, rank_id=0, is_training=True):
    seg_files = sorted(glob.glob(os.path.join(seg_path, "*.nii.gz")))
    train_files = sorted(glob.glob(os.path.join(data_path, "*.nii.gz")))
    train_ds = Dataset(data = train_files, seg = seg_files)
    train_loader = ds.GeneratorDataset(train_ds, column_names = ["image", "seg"], num_parallel_workers = 4, \
                                       shuffle = is_training, num_shards = rank_size, shard_id = rank_id)

    if is_training:
        transform_image = Compose([LoadData(),
                                   ExpandChannel(),
                                   Orientation(),
                                   ScaleIntensityRange(src_min = config.min_val, src_max = config.max_val,
                                                       tgt_min = 0.0, \
                                                       tgt_max = 1.0, is_clip = True),
                                   RandomCropSamples(roi_size = config.roi_size, num_samples = 2),
                                   ConvertLabel(),
                                   OneHot(num_classes = config.num_classes)])
    else:
        transform_image = Compose([LoadData(),
                                   ExpandChannel(),
                                   Orientation(),
                                   ScaleIntensityRange(src_min = config.min_val, src_max = config.max_val,
                                                       tgt_min = 0.0, \
                                                       tgt_max = 1.0, is_clip = True),
                                   ConvertLabel()])

    train_loader = train_loader.map(operations = transform_image, input_columns = ["image", "seg"],
                                    num_parallel_workers = 12,
                                    python_multiprocessing = True)
    if not is_training:
        train_loader = train_loader.batch(1)
    return train_loader
