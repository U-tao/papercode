# =========================================================================
# Copyright (C) 2022. FuxiCTR Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import numpy as np
import torch
import os, sys
import logging
from custom_func.pytorch.models import BaseModel
from custom_func.pytorch.torch_utils import get_device, get_optimizer, get_loss
from tqdm import tqdm
from collections import defaultdict


class MultiTaskModel(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MultiTaskModel",
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(MultiTaskModel, self).__init__(feature_map=feature_map,
                                           model_id=model_id,
                                           task="binary_classification",
                                           gpu=gpu,
                                           loss_weight=loss_weight,
                                           monitor=monitor,
                                           save_best_only=save_best_only,
                                           monitor_mode=monitor_mode,
                                           early_stop_patience=early_stop_patience,
                                           eval_steps=eval_steps,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           reduce_lr_on_plateau=reduce_lr_on_plateau,
                                           **kwargs)
        self.device = get_device(gpu)
        self.num_tasks = num_tasks
        self.loss_weight = loss_weight
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList([self.get_output_activation(task) for _ in range(num_tasks)])

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        if isinstance(loss, list):
            self.loss_fn = [get_loss(l) for l in loss]
        else:
            self.loss_fn = [get_loss(loss) for _ in range(self.num_tasks)]

    def compute_loss(self, return_dict, y_true):
        labels = self.feature_map.labels
        loss = [self.loss_fn[i](return_dict["y_pred"], y_true[i], reduction='mean')
                for i in range(len(labels))]
        if self.loss_weight == 'EQ':
            # Default: All losses are weighted equally
            loss = torch.sum(torch.stack(loss))
        loss += self.regularization_loss()
        return loss
    
    def evaluate(self, data_generator, metrics=None, is_test=False):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []

            # 设置进度条
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                if self.task.lower() == 'pretrain':
                    y_true.extend(return_dict["y_true"].data.cpu().numpy().reshape(-1))
                else:
                    y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None

            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id, is_test)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id, is_test, self.kwargs)
            logging.info('===')
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            labels = self.feature_map.labels
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                for i in range(len(labels)):
                    y_pred_all[labels[i]].extend(
                        return_dict["y_pred"].data.cpu().numpy().reshape(-1))
        return y_pred_all

