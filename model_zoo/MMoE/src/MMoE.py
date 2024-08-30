# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import numpy as np
import pandas
import torch
from torch import nn
import logging, os, sys
from tqdm import tqdm
from collections import defaultdict

from custom_func.pytorch.models import MultiTaskModel
from custom_func.pytorch.layers import FeatureEmbedding, MLP_Block
from custom_func.pytorch.torch_utils import get_activation
from custom_func.pytorch.dataloaders.h5_dataloader import H5DataLoader
from custom_func.preprocess.build_dataset import split_train_test, save_h5

class MMoE_Layer(nn.Module):
    def __init__(self,
                 feature_map,
                 num_experts,
                 num_tasks,

                 embedding_dim,
                 expert_hidden_units,
                 net_dropout,
                 hidden_activations,

                 gate_embedding_dim,
                 gate_hidden_units,
                 gate_net_dropout,
                 gate_hidden_activations,

                 num_fields,
                 output_dim,
                 batch_norm,
                 **kwargs):
        super(MMoE_Layer, self).__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.use_field_gate = kwargs['use_field_gate']
        self.gate_use_field_gate = kwargs['gate_use_field_gate']

        if not isinstance(embedding_dim, list):
            embedding_dim = [embedding_dim] * num_experts
        elif len(embedding_dim) != num_experts:
            embedding_dim = [embedding_dim[0]] * num_experts

        if not isinstance(self.use_field_gate, list):
            self.use_field_gate = [self.use_field_gate] * num_experts
        elif len(self.use_field_gate) != num_experts:
            self.use_field_gate = [self.use_field_gate[0]] * num_experts
        if not isinstance(self.gate_use_field_gate, list):
            self.gate_use_field_gate = [self.gate_use_field_gate] * num_experts
        elif len(self.use_field_gate) != num_experts:
            self.gate_use_field_gate = [self.gate_use_field_gate[0]] * num_experts

        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * num_experts
        elif len(hidden_activations) != num_experts:
            hidden_activations = [hidden_activations[0]] * num_experts
        if not isinstance(gate_hidden_activations, list):
            gate_hidden_activations = [gate_hidden_activations] * num_tasks
        elif len(gate_hidden_activations) != num_experts:
            gate_hidden_activations = [gate_hidden_activations[0]] * num_tasks

        if not isinstance(net_dropout, list):
            net_dropout = [[net_dropout] * num_experts]
        elif not isinstance(net_dropout[0], list):
            net_dropout = [net_dropout] * num_experts
        elif len(net_dropout) != net_dropout:
            net_dropout = [net_dropout[0]] * num_experts
        if not isinstance(gate_net_dropout, list):
            gate_net_dropout = [[gate_net_dropout] * num_tasks]
        elif not isinstance(gate_net_dropout[0], list):
            gate_net_dropout = [gate_net_dropout] * num_tasks
        elif len(gate_net_dropout) != gate_net_dropout:
            gate_net_dropout = [gate_net_dropout[0]] * num_tasks

        if not isinstance(expert_hidden_units, list):
            expert_hidden_units = [expert_hidden_units] * num_experts
        elif not isinstance(expert_hidden_units[0], list):
            expert_hidden_units = [expert_hidden_units] * num_experts
        elif len(expert_hidden_units) != expert_hidden_units:
            expert_hidden_units = [expert_hidden_units[0]] * num_experts
        if not isinstance(gate_hidden_units, list):
            gate_hidden_units = [gate_hidden_units] * num_tasks
        elif not isinstance(gate_hidden_units[0], list):
            gate_hidden_units = [gate_hidden_units] * num_tasks
        elif len(gate_hidden_units) != gate_hidden_units:
            gate_hidden_units = [gate_hidden_units[0]] * num_tasks

        # 多个expert(MLP)是否共享 embedding层
        if kwargs['embedding_share']:
            # 多个专家网络的所共享的 ebedding层
            self.embedding_layer = nn.ModuleList(
                [FeatureEmbedding(feature_map, embedding_dim[0])]
            )
            # gata 网络所用的 embedding层
            self.gata_embedding_layer = self.embedding_layer

            # 每个embedding层的输出维度 = MLP输入维度 = 特征数量 * embedding扩展维度（如果使用门控单元，要再*2）
            input_dim = list(
                [embedding_dim[0] * num_fields * 2 if self.use_field_gate[i] else embedding_dim[0] * num_fields
                 for i in range(self.num_experts)])
            # 门控网络的输入维度
            gate_input_dim = list(
                [gate_embedding_dim[0] * num_fields * 2 if self.gate_use_field_gate[i] else gate_embedding_dim[0] * num_fields
                 for i in range(self.num_tasks)])
        else:
            # 生成多个专家网络的各自的 embedding层
            self.embedding_layer = nn.ModuleList(
                [FeatureEmbedding(feature_map, embedding_dim[i]
                ) for i in range(self.num_experts)])

            # gata 网络所用的 ebedding层（dim=gate_embedding_dim）
            self.gata_embedding_layer = nn.ModuleList(
                [FeatureEmbedding(feature_map, gate_embedding_dim[i]
                ) for i in range(self.num_tasks)])

            # 每个embedding层的输出维度 (= MLP输入维度 = 特征数量 * embedding扩展维度（如果使用门控单元，要再*2）)
            input_dim = list(
                [embedding_dim[i] * num_fields * 2 if self.use_field_gate[i] else embedding_dim[i] * num_fields
                 for i in range(self.num_experts)])
            # 门控网络的输入维度
            gate_input_dim = list(
                [gate_embedding_dim[i] * num_fields * 2 if self.gate_use_field_gate[i] else gate_embedding_dim[i] * num_fields
                for i in range(self.num_tasks)])

        # 门控单元
        if True in self.use_field_gate:
            self.field_gate = FinalGate(feature_map.num_fields, gate_residual="concat")
        if True in self.gate_use_field_gate:
            self.gate_field_gate = FinalGate(feature_map.num_fields, gate_residual="concat")

        self.experts = nn.ModuleList([MLP_Block(input_dim           =input_dim[i],
                                                hidden_units        =expert_hidden_units[i],
                                                output_dim          =output_dim,
                                                hidden_activations  =hidden_activations[i],
                                                output_activation   =None,
                                                dropout_rates       =net_dropout[i],
                                                batch_norm          =batch_norm)
                                      for i in range(self.num_experts)])

        self.gate = nn.ModuleList([MLP_Block(input_dim          =gate_input_dim[i],
                                             hidden_units       =gate_hidden_units[i],
                                             output_dim         =num_experts,
                                             hidden_activations =gate_hidden_activations[i],
                                             output_activation  =None,
                                             dropout_rates      =gate_net_dropout[i],
                                             batch_norm         =batch_norm)
                                   for i in range(self.num_tasks)])

        self.gate_activation = get_activation('softmax')

    def forward(self, x):   # 输入为一个字典

        layer_output = []

        for ind in range(self.num_experts):
            # 判断该层专家网络是否共享embedding层（否）
            if len(self.embedding_layer) > 1:
                feature_emb = self.embedding_layer[ind](x)  # batch×num_feat×embed_dim
            else:
                feature_emb = self.embedding_layer[0](x)
            # 判断该层专家网络是否使用门控单元
            if self.use_field_gate[ind]:
                feature_emb = self.field_gate(feature_emb)

            # 展成二维
            feature_emb = feature_emb.flatten(start_dim=1)  # batch × (num_feat×embed_dim)
            # 进入该层专家网络进行训练
            feature_layer = self.experts[ind](feature_emb)  # batch × out_dim
            # 收集该层专家网络的结果
            layer_output.append(feature_layer)

        experts_output = torch.stack(layer_output, dim=1)  # (batch × num_experts × expert_outdim)

        mmoe_output = []

        # 专家网络的结果会输出为多个目标（num_tasks，本论文只有一个),并将多个目标聚合
        for ind in range(self.num_tasks):
            feature_emb = self.gata_embedding_layer[ind](x)

            if self.gate_use_field_gate[ind]:
                feature_emb = self.field_gate(feature_emb)
            feature_emb = feature_emb.flatten(start_dim=1)

            gate_output = self.gate[ind](feature_emb)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (batch, num_experts)
            mmoe_output.append(
                torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1)
            )
        return mmoe_output


class MMoE(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 feature_encoder,
                 model_id               ="MMoE",
                 gpu                    =-1,
                 learning_rate          =1e-3,
                 num_experts            =3,
                 task                   =["binary_classification"],
                 num_tasks              =1,

                 embedding_dim          =[10, 10, 10],
                 expert_hidden_units    =[400],
                 net_dropout            =[[0], [0], [0]],
                 hidden_activations     =["ReLU", "ReLU", "ReLU"],

                 gata_embedding_dim     =[10],
                 gate_hidden_units      =[[[128, 64], [128, 64], [128, 64]]],
                 gate_net_dropout       =[[0]],
                 gate_hidden_activations=["ReLu"],

                 batch_norm             =False,
                 net_regularizer        =None,
                 embedding_regularizer  =None,

                 tower_hidden_units     =[[[128, 64], [128, 64], [128, 64]]],
                 tower_net_dropout      =[[0]],
                 tower_hidden_activations=["ReLu"],

                 **kwargs):

        super(MMoE, self).__init__(feature_map,
                                   task                 =task,
                                   num_tasks            =num_tasks,
                                   model_id             =model_id,
                                   gpu                  =gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer      =net_regularizer,
                                   **kwargs)

        # 定义所有专家最后的输出维度（当前选择：所有专家最后一层最小的作为输出维度）
        # output_dim = min([expert_hidden_units[i][-1] for i in range(len(expert_hidden_units))])
        if not isinstance(expert_hidden_units[0], list):
            output_dim = expert_hidden_units[-1] * -1
        elif len(expert_hidden_units) == num_experts:
            for i in range(num_experts):
                if expert_hidden_units[i]!= expert_hidden_units[0]:
                    output_dim = min([expert_hidden_units[i][-1] for i in range(len(expert_hidden_units))])
                    break
                output_dim = expert_hidden_units[0][-1] * -1
        else:
            raise ValueError("expert_hidden_units 输入错误")

        self.mmoe_layer = MMoE_Layer(feature_map            =feature_map,
                                     num_experts            =num_experts,
                                     num_tasks              =self.num_tasks,

                                     embedding_dim          =embedding_dim,
                                     expert_hidden_units    =expert_hidden_units,
                                     net_dropout            =net_dropout,
                                     hidden_activations     =hidden_activations,

                                     embedding_dim_gata     =gata_embedding_dim,
                                     gate_hidden_units      =gate_hidden_units,
                                     gate_net_dropout       =gate_net_dropout,
                                     gate_hidden_activations=gate_hidden_activations,

                                     num_fields             =feature_map.num_fields,
                                     output_dim             =output_dim,
                                     batch_norm             =batch_norm,
                                     **kwargs)
        self.tower = nn.ModuleList([MLP_Block(input_dim         =abs(output_dim),
                                              output_dim        =1,
                                              hidden_units      =tower_hidden_units[i],
                                              hidden_activations=tower_hidden_activations[i],
                                              output_activation =None,
                                              dropout_rates     =tower_net_dropout[i],
                                              batch_norm        =batch_norm)
                                    for i in range(num_tasks)])
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        self.feature_map = feature_map
        self.feature_encoder = feature_encoder

    def forward(self, inputs):
        # 将输入转换成一个字典，{特征名称：数据（对应的一列数据），。。。}
        # 这里会将inputs 中的 y 去掉 ——> X
        X = self.get_inputs(inputs)

        expert_output = self.mmoe_layer(X)

        tower_output = [self.tower[i](expert_output[i]) for i in range(self.num_tasks)]

        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        for i in range(self.num_tasks):
            return_dict["y_pred"] = y_pred[i]
        return return_dict


    def fit(self,
            X_train,
            y_train,
            max_gradient_norm=10.,
            sample_weight=None) -> object:

        self.train_gen, self.valid_gen = self.transform(X_train, y_train, 'train')
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(self.train_gen) - 1
        self.epochs = self.kwargs['epochs']
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(self.epochs):
            self._epoch_index = epoch
            self.train_epoch(self.train_gen)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def predict_proba(self, X_test):
        test_gen = self.transform(X=X_test,filename='test')
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            labels = self.feature_map.labels
            if self._verbose > 0:
                data_generator = tqdm(test_gen, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                for i in range(len(labels)):
                    y_pred_all[labels[i]].extend(
                        return_dict["y_pred"].data.cpu().numpy().reshape(-1))
        return y_pred_all

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = FocalLoss(return_dict["y_pred"], y_true, reduction='mean')
        return loss

    def transform(
            self,
            X,
            y=None,
            filename='test'):

        if 'train' in filename:
            ddf = pandas.concat([X, y], axis=1).reset_index(drop=True)
            valid_size = 0.2
            train_ddf, valid_ddf, test_ddf = split_train_test(source_ddf=ddf, valid_size=valid_size, test_size=0.1)
            train_ddf = self.feature_encoder.preprocess(train_ddf)
            valid_ddf = self.feature_encoder.preprocess(valid_ddf)
            darray_train = self.feature_encoder.transform(train_ddf)
            darray_valid = self.feature_encoder.transform(valid_ddf)
            save_h5(darray_train, os.path.join(self.feature_encoder.data_dir, 'train.h5'))
            save_h5(darray_valid, os.path.join(self.feature_encoder.data_dir, 'valid.h5'))
            train_data_dir = os.path.join(self.feature_encoder.data_dir, 'train.h5')
            valid_data_dir = os.path.join(self.feature_encoder.data_dir, 'valid.h5')
            train_gen, valid_gen = H5DataLoader(self.feature_map, train_data=train_data_dir, valid_data= valid_data_dir, stage='train', **self.kwargs).make_iterator()
            return train_gen, valid_gen
        elif 'test' in filename:
            ddf = X
            darray_test = self.feature_encoder.transform(ddf)
            save_h5(darray_test, os.path.join(self.feature_encoder.data_dir, 'test.h5'))
            test_data_dir = os.path.join(self.feature_encoder.data_dir, 'test.h5')
            test_gen = H5DataLoader(self.feature_map, test_data=test_data_dir, stage='test', **self.kwargs).make_iterator()
            return test_gen
        else:
            ValueError(f'transform DataFram -> DataLoder as {filename} error')



def FocalLoss(y_pred, y_true, gamma=2, alpha=0.90, reduction='mean'):
    loss = torch.functional.F.binary_cross_entropy(y_pred, y_true, reduction='none')

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    modulating_factor = (1.0 - p_t) ** gamma
    loss = loss * modulating_factor

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # 预测为真的概率
    modulating_factor = (1.0 - p_t) **gamma * (alpha * y_true + (1 - alpha) * (1 - y_true))
    loss = loss * modulating_factor

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

class FinalGate(nn.Module):
    def __init__(self, num_fields, gate_residual="concat"):
        super(FinalGate, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)
        assert gate_residual in ["concat", "sum"]
        self.gate_residual = gate_residual

    def reset_custom_params(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.ones_(self.linear.bias)

    def forward(self, feature_emb):
        feature_emb_tran = feature_emb.transpose(1, 2)
        gates = self.linear(feature_emb_tran).transpose(1, 2)
        if self.gate_residual == "concat":
            out = torch.cat([feature_emb, feature_emb * gates], dim=1)  # b x 2f x d
        else:
            out = feature_emb + feature_emb * gates
        return out


