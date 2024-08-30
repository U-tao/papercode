from custom_func.utils import load_config, set_logger, print_to_json, print_to_list
from custom_func.features import FeatureMap
from custom_func.pytorch.torch_utils import seed_everything
from custom_func.preprocess import FeatureProcessor, build_dataset


import sys
base_path = '../../'
sys.path.append(base_path)

import logging
from datetime import datetime
import src as model_zoo
import argparse
import os
from pathlib import Path
from src.other_model import model_gen

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='MMoE_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    # 加载参数 -> params
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))

    # 设立随机种子c
    seed_everything(seed=params['seed'])

    # 生成 Feature_map params
    data_dir = os.path.join(params['data_root'], params['dataset_id'])  # 数据文件夹 ../../data/SAT
    feature_map_json = os.path.join(data_dir, "feature_map.json")       # 数据地图文件：../../data/feature_map.json
    if params["data_format"] == "csv":
        feature_encoder = FeatureProcessor(**params)
        X_train, X_test, y_train, y_test = build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # 定义模型
    estimators = model_gen(
        feature_map=feature_map,
        feature_encoder=feature_encoder,
        feat=X_train,
        label=y_train,
        **params)

    model_class = getattr(model_zoo, 'Voting')  # 获取 模型 库函数
    model = model_class(
        estimators=estimators,
        voting='soft',
        feature_map=feature_map,
        **params
    )
    model.fit(X_train, y_train)

    logging.info('******** Test evaluation ********')
    test_result = model.evaluate(X_test, y_test, is_test=True)

    # shap分析
    model.Ans_shap(model, X_test, params['shap'])

    # 记录模型预测结果
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'

    if not os.path.exists(result_filename):
        with open(result_filename, 'a+') as fw:
            fw.write('Time,Code,Exp_id,dataset,resample,Exp_list,Res_Train_,Res_val,Res_test,AUC,other\n')
    other_list = []
    if len(params['result_record']) > 0:
        for i in range(len(params['result_record'])):
            if isinstance(params[params['result_record'][i]], list):
                c = ' - '.join(map(str,params[params['result_record'][i]])).replace(',', '-')
            else :
                c = ' - '.join(map(str, [params[params['result_record'][i]]])).replace(',', '-')
            other_list.append(c)
        ','.join(other_list)
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[resample] {},[model_list] {},[train] {},[val] {},[test] {}, {}, {}\n' \
                 .format(
            datetime.now().strftime('%Y.%m.%d-%H:%M:%S'),
                  ' '.join(sys.argv),
                  experiment_id,
                  params['dataset_id'],
                  params['resample'],
                  ' - '.join(params['model_list']),
                  "N.A.",
                  None,
                  print_to_list(test_result),
                  test_result['AUC'],
                  other_list
                )
        )