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

from easydict import EasyDict

config = EasyDict({
    'model': 'Unet3d',
    'lr': 0.0001,
    'epoch_size': 600,
    'batch_size': 1,
    'warmup_step': 120,
    'warmup_ratio': 0.3,
    'num_classes': 9,
    'in_channels': 1,
    'keep_checkpoint_max': 5,
    'loss_scale': 256.0,
    'roi_size': [96, 96, 96],
    'overlap': 0.25,
    'min_val': 0,
    'max_val': 900,
    'upper_limit': 5,
    'lower_limit': 3,
})
