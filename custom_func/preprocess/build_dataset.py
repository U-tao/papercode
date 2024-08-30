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


import h5py
import os
import logging
import numpy as np
import gc
import multiprocessing as mp

import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def save_h5(darray_dict, data_path):
    logging.info("Saving data to h5: " + data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with h5py.File(data_path, 'w') as hf:
        for key, arr in darray_dict.items():
            hf.create_dataset(key, data=arr)

def resample(dff, **params):
    y_name = params['label_col']['name']
    X = dff.drop(columns=y_name)
    y = dff[y_name]

    if params['resample'].upper() in ['OVER', 'O']:
        sm = SMOTE(random_state=params['seed'], n_jobs=-1)
        X_smote, y_smote = sm.fit_resample(X, y)
        df_resample = pd.concat([y_smote, X_smote], axis=1)
    elif params['resample'].upper() in ['LOWER', 'L']:
        rus = RandomUnderSampler(random_state=params['seed'])
        X_smote, y_smote = rus.fit_resample(X, y)
        df_resample = pd.concat([y_smote, X_smote], axis=1)
    return df_resample

def split_train_test(
        source_ddf=None,
        train_ddf=None,
        valid_ddf=None,
        test_ddf=None,
        valid_size=0,
        test_size=0,
        split_type="random"):

    num_samples = len(source_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)

    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = source_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]

    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = source_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]

    if valid_size > 0 or test_size > 0:
        train_ddf = source_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def transform_h5(feature_encoder, ddf, filename, preprocess=False, block_size=0):
    def _transform_block(feature_encoder, df_block, filename, preprocess):
        if preprocess:
            df_block = feature_encoder.preprocess(df_block)
        darray_dict = feature_encoder.transform(df_block)
        save_h5(darray_dict, os.path.join(feature_encoder.data_dir, filename))

    if block_size > 0:
        pool = mp.Pool(mp.cpu_count() // 2)
        block_id = 0
        for idx in range(0, len(ddf), block_size):
            df_block = ddf[idx: (idx + block_size)]
            pool.apply_async(_transform_block, args=(feature_encoder,
                                                     df_block,
                                                     '{}/part_{}.h5'.format(filename, block_id),
                                                     preprocess))
            block_id += 1
        pool.close()
        pool.join()
    else:
        _transform_block(feature_encoder, ddf, filename + ".h5", preprocess)


def build_dataset(
        feature_encoder,
        source_data=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        valid_size=0,
        test_size=0,
        split_type="random",
        data_block_size=0,
        **kwargs):
    """ Build feature_map and transform h5 data """

    if kwargs['data_regen']:
        if not os.path.exists(os.path.join(feature_encoder.data_dir, 'source.csv')):
            raise ValueError("缺少数据源文件(source.csv)")
        for file_name in os.listdir(feature_encoder.data_dir):
            path = os.path.join(feature_encoder.data_dir, file_name)
            if file_name != 'source.csv':
                os.remove(path)

    feature_map_json = os.path.join(feature_encoder.data_dir, "feature_map.json")
    if os.path.exists(feature_map_json):
        logging.warn("Skip rebuilding {}. Please delete it manually if rebuilding is required." \
                     .format(feature_map_json))
    else:
        # Load csv data
        source_ddf = feature_encoder.read_csv(source_data, **kwargs)
        train_ddf = None
        valid_ddf = None
        test_ddf = None

        # Split data for train/validation/test
        if valid_size > 0 or test_size > 0:
            if valid_data != None and os.path.exists(valid_data):
                valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
            if test_data != None and os.path.exists(test_data):
                test_ddf = feature_encoder.read_csv(test_data, **kwargs)
            if train_data != None and os.path.exists(train_data):
                test_ddf = feature_encoder.read_csv(test_data, **kwargs)
            train_ddf, valid_ddf, test_ddf = split_train_test(source_ddf=source_ddf, valid_size=valid_size, test_size=test_size, split_type=split_type)

        voting_train = []
        voting_test  = []
        voting_sort  = list(source_ddf.columns)

        # fit and transform train_ddf
        train_ddf = feature_encoder.preprocess(train_ddf)   # 数据预处理：1，缺省值填充；2，数据类型转换
        feature_encoder.fit(train_ddf, **kwargs)            # 连续变量归一化处理，离散变量向量化处理
        if kwargs['resample'].upper() in ['OVER', 'LOWER']:
            train_ddf = resample(train_ddf, **kwargs)
        voting_train.append(train_ddf)
        del train_ddf
        gc.collect()

        # Transfrom valid_ddf
        if valid_ddf is None and (valid_data is not None):
            valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
        if valid_ddf is not None:
            valid_ddf = feature_encoder.preprocess(valid_ddf)
            voting_train.append(valid_ddf)
            del valid_ddf
            gc.collect()

        # Transfrom test_ddf
        if test_ddf is None and (test_data is not None):
            test_ddf = feature_encoder.read_csv(test_data, **kwargs)
        if test_ddf is not None:
            test_ddf = feature_encoder.preprocess(test_ddf)
            voting_test.append(test_ddf)
            del test_ddf
            gc.collect()
        logging.info("Transform csv data to h5 done.")

        voting_train = pd.concat(voting_train, axis=0)
        voting_test  = pd.concat(voting_test, axis=0)
        voting_train = voting_train[voting_sort]
        voting_test  = voting_test[voting_sort]

        voting_X_train = voting_train.drop(columns=kwargs['label_col']['name'])
        voting_y_train = voting_train[kwargs['label_col']['name']]
        voting_X_test = voting_test.drop(columns=kwargs['label_col']['name'])
        voting_y_test = voting_test[kwargs['label_col']['name']]

        return (
            voting_X_train,
            voting_X_test,
            voting_y_train,
            voting_y_test
        )

