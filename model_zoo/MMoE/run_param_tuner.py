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
# import sys
# sys.path.append("/mnt/public/lhh/code/")
import sys

sys.path.append("/home/lhh/code")
from datetime import datetime
import gc
import argparse
from custom_func import autotuner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/FINAL_tuner_config.yaml', help='The config file for para tuning.')
    parser.add_argument('--tag',    type=str, default=None, help='Use the tag to determine which expid to run (e.g. 001 for the first expid).')
    parser.add_argument('--gpu',    nargs='+', default=[0], help='The list of gpu indexes, -1 for cpu.')


    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    expid_tag = args['tag']

    # generate parameter space combinations（生成参数空间组合，并放置到FINAL_tuner_config目录中
    config_dir = autotuner.enumerate_params(args['config'])
    # 开始搜索
    autotuner.grid_search(config_dir, gpu_list, expid_tag)
