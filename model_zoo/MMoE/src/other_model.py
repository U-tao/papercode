import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

import lightgbm
import xgboost
import optuna
import sklearn.ensemble as ensemble

from model_zoo.MMoE import src as model_zoo


# Opt_lightgbm 参数优化函数
def objective_binary(trial, features_df, labels_df, model_id):
    features = np.array(features_df)
    labels = np.array(labels_df)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    score = []
    if model_id[0:3].upper() in ['L', 'LIG']:
        lgbparams_optuna = {
            # 控制树结构
            'num_leaves'        : trial.suggest_int('num_leaves', 20, 4000, step=20),
            "max_depth"         : trial.suggest_int("max_depth", 1, 25, step=1),
            'min_child_samples' : trial.suggest_int('min_child_samples', 1, 1000, step=1),

            # 控制精度
            'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 0.5, step=0.01),
            'n_estimators'      : trial.suggest_categorical('n_estimators', [4000, 8000, 12000, 16000]),
            'max_bin'           : trial.suggest_int('max_bin', 200, 800, step=1),

            # 控制过拟合
            'lambda_l1'         : trial.suggest_float('lambda_l1', 0, 100, step=0.1),
            'lambda_l2'         : trial.suggest_float('lambda_l2', 0, 100, step=1.),
            'min_gain_to_split' : trial.suggest_float('min_gain_to_split', 0, 15, step=0.1),
            'bagging_fraction'  : trial.suggest_float('bagging_fraction', 0.01, 0.99, step=0.01),
            'feature_fraction'  : trial.suggest_float('feature_fraction', 0.01, 0.99, step=0.01),
            'bagging_freq'      : trial.suggest_int('bagging_freq', 1, 9, step=1),

            # 其他
            # 'boosting_type'     : trial.suggest_categorical('boosting_type', ['gbdt', 'rf']),
            # 'metric'            : trial.suggest_categorical('metric', ['auc', 'binary']),
            # 'objective'         : trial.suggest_categorical('objective', ['binary']),
            # "device_type"       : trial.suggest_categorical("device_type", ['gpu']),
            'scale_pos_weight'  : trial.suggest_float('scale_pos_weight', 0.1, 5, step=0.1),
            # 'num_iterations': trial.suggest_categorical('num_iterations', [10000]),
            'verbose'           : trial.suggest_categorical('verbose', [-1]),

        }
        model = lightgbm.LGBMClassifier(**lgbparams_optuna, class_weight='balanced')

        for idx, (trn_idx, val_idx) in enumerate(skf.split(features, labels)):
            print('\n{} of kfold {}'.format(idx, skf.n_splits))
            X_trn, X_val = features[trn_idx], features[val_idx]
            y_trn, y_val = labels[trn_idx], labels[val_idx]

            model.fit(
                X_trn,
                y_trn,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lightgbm.early_stopping(stopping_rounds=50)
                ]
            )
            y_pred_prob = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, y_pred_prob[:, 1])
            score.append(auc)
    elif model_id[0:3].upper()  in ['X', 'XGB']:
        xgbparams_optuna = {
            'tree_method'           : 'hist',  # Use GPU acceleration
            'lambda'                : trial.suggest_float('lambda', 1e-3, 10.0),
            'alpha'                 : trial.suggest_float('alpha', 1e-3, 10.0),
            'colsample_bytree'      : trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample'             : trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 1.0]),
            'learning_rate'         : trial.suggest_categorical('learning_rate',
                                                       [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
            'n_estimators'          : trial.suggest_categorical("n_estimators", [150, 200, 300, 3000]),
            'max_depth'             : trial.suggest_categorical('max_depth', [4, 5, 7, 9, 11, 13, 15, 17]),
            'random_state'          : 2024,
            'min_child_weight'      : trial.suggest_int('min_child_weight', 1, 300),

            'device'                : "cuda",
            'early_stopping_rounds' : 50
        }
        model = xgboost.XGBClassifier(**xgbparams_optuna)

        for idx, (trn_idx, val_idx) in enumerate(skf.split(features, labels)):
            print('\n{} of kfold {}'.format(idx, skf.n_splits))
            X_trn, X_val = features[trn_idx], features[val_idx]
            y_trn, y_val = labels[trn_idx], labels[val_idx]

            model.fit(
                X_trn,
                y_trn,
                eval_set=[(X_val, y_val)],
            )
            y_pred_prob = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, y_pred_prob[:, 1])
            score.append(auc)
    else:
        raise ValueError("model_id 不存在")

    return np.mean(score)

def Opt(X_train, y_train, n_trial=10, model_id=None):
    print('----------------Opt优化' + model_id + '分类模型-开始-----------------')

    study = optuna.create_study(
        direction='maximize',
        study_name='optuna_' + model_id,
        load_if_exists=True,
    )
    func = lambda trial: objective_binary(trial, X_train, y_train, model_id)
    study.optimize(func, n_trials=n_trial)

    print(f"\tBest value (roc_score): {study.best_value:.5f}")
    print(f"\tBest params : ")

    for key, value in study.best_params.items():
        print(f'\t\t{key} : {value}')
    print('{:-^50}'.format(f'模型优化结束'))

    return study.best_params

def model_grid(X_train, y_train, n_trial=10, model_id=None):
    model_name = model_id[0:3].upper()
    if model_name in ['RF','R']:
        param_grid = {
            'n_estimators'        : [int(x) for x in np.linspace(start = 10, stop = 150, num = 8)],
            'max_features'        : [0.3, 0.4, 0.5],
            'max_depth'           : [5, 6, 7, 8],
            'min_samples_split'   : [4, 8, 12, 16],
            'min_samples_leaf'    : [1, 2],
            'bootstrap'           : [True, False],
        }
        #                        JAT
        #                        blance    over    lower
        # 'n_estimators'        :           72      90
        # 'max_features'        :           0.3     0.4
        # 'min_samples_split'   :           8       12
        # 'max_depth'           :           8       7
        # 'min_samples_leaf'    :           1       2
        # 'bootstrap'           :           False   False
        rf_Model = ensemble.RandomForestClassifier()
        Grid = GridSearchCV(
            estimator=rf_Model,
            param_grid=param_grid,
            cv=n_trial,
            verbose=2,
            n_jobs=4,
            scoring='roc_auc')
        Grid.fit(X_train, y_train)

    elif model_name.upper() in ['S', 'SVC', 'SVM']:
        param_grid = {
            'C'                   : [0.001, 0.01, 0.1, 1., 10, 100, 1000],  # [0.001, 0.01, 0.1, 1., 10, 100, 1000]
            'kernel'              : ["linear", "poly", "rbf", "sigmoid"],   # ["linear", "poly", "rbf", "sigmoid"]
            'gamma'               : ["scale", "auto"],                      # ["scale", "auto"]
            'probability'         : [True],
            'max_iter'            : [10000],

            #         JAT       JAT-T1    SAT-T1
            #         blance    blance    blance
            # C     : 10,       1         10
            # gamma : scale,    scale,    scale
            # kernel: linear    linear    linear
        }
        svc_Model = SVC(
            C=10,
            kernel='linear',
            gamma='scale',
            probability=True,
        )
        # return svc_Model
        svc_Model = SVC()
        Grid = GridSearchCV(
            estimator=svc_Model,
            param_grid=param_grid,
            cv=n_trial,
            verbose=2,
            n_jobs=4,
            scoring='roc_auc')
        Grid.fit(X_train, y_train)

    else:
        raise ValueError("model_id 不存在")

    print(f'\nGrid_{model_name} : {Grid.best_score_}')
    for key, value in Grid.best_params_.items():
        print (f'\t{key}: {value}')
    return Grid.best_estimator_

def model_gen(feature_map,
              feature_encoder,
              feat       =None,
              label      =None,
              model_list =[],
              **params):

    estimators=[]
    count_model = 0

    for name in model_list:
        if name[0:3].upper() in ['XGB', 'X']:
            param = {

            }
            count_model += 1
            if params['optim']:
                best_opt_params = Opt(feat, label, n_trial=50, model_id=name.upper())
                param.update(best_opt_params)
            model = xgboost.XGBClassifier(**param)
            estimators.append(tuple(('XGBOOST_'+ str(count_model), model)))
        elif name[0:3].upper() in ['LIG', 'L']:
            param = {

            }
            count_model += 1
            if params['optim']:
                best_opt_params = Opt(feat, label, n_trial=50, model_id=name.upper())
                param.update(best_opt_params)
            model = lightgbm.LGBMClassifier(**param, class_weight='balanced')
            estimators.append(tuple(('LightGBM_' + str(count_model), model)))
        elif name.upper() in ['MLP']:
            count_model += 1
            model_class = getattr(model_zoo, params['model'])   # 获取 模型 库函数
            model = model_class(feature_map, feature_encoder, **params)          # 定义 模型 类
            model.count_parameters()                            # 打印 模型 参数
            estimators.append(tuple(('MLP_' + str(count_model), model)))
        elif name[0:3].upper() in ['SVC','S', 'SVM']:
            count_model += 1
            if params['optim']:
                model = model_grid(feat, label, n_trial=5, model_id=name.upper())
            else:
                model = SVC(probability=True)
            estimators.append(tuple(('SVC_' + str(count_model), model)))
        elif name[0:3].upper() in ['RF','R']:
            count_model += 1
            if params['optim']:
                model = model_grid(feat, label, n_trial=5, model_id=name.upper())
            else:
                model = ensemble.RandomForestClassifier()
            estimators.append(tuple(('RF_' + str(count_model), model)))
        else:
            raise ValueError("model_list " + name + "输入错误")

    return estimators


def graph_roc_curve_multiple(dict_pred):
    plt.figure(figsize=(16, 8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)

    for key, result in dict_pred.items():
        fpr, tpr, thresold = roc_curve(result['y_true'], result['y_pred_prob'])
        label = str(key) + ' Classifier Score: {:.4f}'.format(roc_auc_score(result['y_true'], result['y_pred_prob']))
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1.2], [0, 1.2], 'k--')
    plt.axis([-0.01, 1.01, -0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                 arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                 )
    plt.legend()
    plt.show()




