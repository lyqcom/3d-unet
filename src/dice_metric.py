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

import numpy as np
from mindspore import nn
from mindspore.ops import operations as P

from suwen.data import Tensor
from suwen.data import swtype
from suwen.metrics import Metric

class DiceMetric(Metric):
    r"""
    The Dice coefficient metric is a set similarity metric. It is used to calculate the similarity between two samples. The
    value of the Dice coefficient is 1 when the segmentation result is the best and 0 when the segmentation result
    is the worst. The Dice coefficient indicates the ratio of the area between two objects to the total area.
    The function is shown as follows:

    .. math::
        dice = \frac{2 * (pred \bigcap true)}{pred \bigcup true}

    Args:
        num_classes (int): number of classes. Default: 2.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
        >>> metric = DiceMetric(num_classes=2)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> dice = metric.eval()
        >>> print(dice)
        0.20467791371802546
    """

    def __init__(self, num_classes=2):
        super(DiceMetric, self).__init__()
        self.num_classes = num_classes
        self.argmax = P.Argmax(axis = 1)
        self.one_hot = P.OneHot()
        self.softmax = nn.Softmax(axis = 1)
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.on_value = Tensor(1.0, swtype.float32)
        self.off_value = Tensor(0.0, swtype.float32)
        self._dice_coeff_sum = 0
        self._samples_num = 0
        self.smooth = 1e-5
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._dice_coeff_sum = 0
        self._samples_num = 0

    def ignore_background(self, y_pred, y):
        """
        This function is used to remove background (the first channel) for `y_pred` and `y`.
        Args:
            y_pred: predictions. As for classification tasks, `y_pred` should has the shape [BN] where N is
            larger than 1. As for segmentation tasks, the shape should be [BNHW] or [BNHWD].
            y: ground truth, the first dim is batch.
        """
        y = y[:, 1:] if y.shape[1] > 1 else y
        y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred
        return y_pred, y

    def update(self, logits, label):
        N, C, H, W = logits.shape
        logits = self.reshape(logits, (N, C, -1))
        out = self.softmax(logits)
        logits = self.reshape(out, (N, C, H, W))

        label = self.transpose(label, (0, 2, 3, 1))
        label = self.cast(label, swtype.int32)
        one_hot_labels = self.one_hot(label, self.num_classes, self.on_value, self.off_value)
        one_hot_labels = self.reshape(one_hot_labels, (N, H, W, C))
        one_hot_labels = self.transpose(one_hot_labels, (0, 3, 1, 2))

        y_pred = self._convert_data(logits)
        y = self._convert_data(one_hot_labels)
        y_pred, y = self.ignore_background(y_pred, y)
        self._samples_num += 1
        if y_pred.shape != y.shape:
            raise RuntimeError('y_pred and y should have same the dimension, but the shape of y_pred is{}, '
                               'the shape of y is {}.'.format(y_pred.shape, y.shape))

        intersection = np.dot(y_pred.flatten(), y.flatten())
        unionset = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(intersection) / float(unionset + self.smooth)
        self._dice_coeff_sum += single_dice_coeff

    def eval(self):
        r"""
        Computes the Dice.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the total samples num is 0.
        """
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')

        return self._dice_coeff_sum / float(self._samples_num)
