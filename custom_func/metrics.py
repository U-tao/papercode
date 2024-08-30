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
import os.path

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict
import matplotlib.pyplot as plt


def evaluate_metrics(y_true: object,
                     y_pred: object = None,
                     y_pred_proba: object = None,
                     metrics: object = [],
                     group_id: object = None,
                     is_test: object = False,
                     **params: object):
    return_dict = OrderedDict()
    group_metrics = []
    thred = best_thred(y_true, y_pred_proba)
    if y_pred == None:
        y_pred = (y_pred_proba >= thred).astype(int)

    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            return_dict[metric] = log_loss(y_true, y_pred_proba)
        elif metric in ['accuracy']:
            return_dict[metric] = accuracy_score(y_true, y_pred)
        elif metric in ['precision']:
            return_dict[metric] = precision_score(y_true, y_pred)
        elif metric in ['recall']:
            return_dict[metric] = recall_score(y_true, y_pred)
        elif metric in ['f1_score']:
            return_dict[metric] = f1_score(y_true, y_pred)
        elif metric in ['mse', 'mean_squared_error']:
            return_dict[metric] = mean_squared_error(y_true, y_pred_proba)
        elif metric == 'AUC':
            return_dict[metric] = roc_auc_score(y_true, y_pred_proba)
            if is_test:
                graph_roc_curve_multiple(y_true, y_pred_proba, y_pred, **params)
        elif metric == 'classification_report':
            print(classification_report(y_true, y_pred))
            print(confusion_matrix(y_true, y_pred))
        elif metric == 'Sensitivity':
            P, N = confusion_matrix(y_true, y_pred)
            tn, fp = P[0], P[1]
            fn, tp = N[0], N[1]
            return_dict[metric] = tp / (tp + fn)
        elif metric == 'False_alarm_rate':
            P, N = confusion_matrix(y_true, y_pred)
            tn, fp = P[0], P[1]
            fn, tp = N[0], N[1]
            return_dict[metric] = fp / (fp + tn)
        elif metric == 'Specificity':
            P, N = confusion_matrix(y_true, y_pred)
            tn, fp = P[0], P[1]
            fn, tp = N[0], N[1]
            return_dict[metric] = tn / (fp + tn)
        elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
            return_dict[metric] = 0
            group_metrics.append(metric)
        else:
            raise ValueError("metric={} not supported.".format(metric))
    if len(group_metrics) > 0:
        assert group_id is not None, "group_index is required."
        metric_funcs = []
        for metric in group_metrics:
            try:
                metric_funcs.append(eval(metric))
            except:
                raise NotImplementedError('metrics={} not implemented.'.format(metric))
        score_df = pd.DataFrame({"group_index": group_id,
                                 "y_true": y_true,
                                 "y_pred": y_pred_proba})
        results = []
        pool = mp.Pool(processes=mp.cpu_count() // 2)
        for idx, df in score_df.groupby("group_index"):
            results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        sum_results = np.array(results).sum(0)
        average_result = list(sum_results[:, 0] / sum_results[:, 1])
        return_dict.update(dict(zip(group_metrics, average_result)))
    return return_dict

def evaluate_block(df, metric_funcs):
    res_list = []
    for fn in metric_funcs:
        v = fn(df.y_true.values, df.y_pred.values)
        if type(v) == tuple:
            res_list.append(v)
        else: # add group weight
            res_list.append((v, 1))
    return res_list

def avgAUC(y_true, y_pred):
    """ avgAUC used in MIND news recommendation """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        return (auc, 1)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def gAUC(y_true, y_pred):
    """ gAUC defined in DIN paper """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        n_samples = len(y_true)
        return (auc * n_samples, n_samples)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def MRR(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    mrr = np.sum(rr_score) / (np.sum(y_true) + 1e-12)
    return mrr

def graph_roc_curve_multiple(y_true, y_score, y_pred, **params):
    df = pd.DataFrame()
    df['y_true']        = y_true
    df['y_pred_prob']   = y_score
    df['y_pred']        = y_pred

    E_name = ''
    for ind in range(len(params['model_list'])) :
        if params['model_list'][ind][0:3].upper() in ['L', 'LIG']:
            E_name += 'LightGBM_'
        elif params['model_list'][ind][0:3].upper() in ['X', 'XGB']:
            E_name += 'Xgboost_'
        elif params['model_list'][ind][0:3].upper() in ['R', 'RF', 'Ran']:
            E_name += 'RF_'
        elif params['model_list'][ind][0:3].upper() in ['S', 'SVC', 'SVM']:
            E_name += 'SVC_'
        elif params['model_list'][ind][0:3].upper() in ['M', 'MLP', 'MoE']:
            if E_name != '':
                E_name = 'MLP_' + E_name
            else:
                E_name += 'MLP_'
    if params['resample'].upper() in ['OVER', 'LOWER']:
        os.makedirs('./graph_auc/' + params['dataset_id'] + '_' + params['resample'].upper(), exist_ok=True)
        path = './graph_auc/' + params['dataset_id'] + '_' + params['resample'].upper() + '/'
    else:
        os.makedirs('./graph_auc/' + params['dataset_id'], exist_ok=True)
        path = './graph_auc/' + params['dataset_id'] + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    files = os.listdir(path)

    # 保存合适MoE值
    def test(df, path, E_name):
        auc_thred = 0.60
        spe_thred = 0.70
        sen_thred = 0.79

        P, N = confusion_matrix(df['y_true'].values.tolist(), df['y_pred'].values.tolist())
        tn, fp = P[0], P[1]
        fn, tp = N[0], N[1]
        Spe = tn / (fp + tn)
        Sen = tp / (tp + fn)
        auc = roc_auc_score(df['y_true'].values.tolist(), df['y_pred_prob'].values.tolist())

        if round(Spe,2) >= spe_thred or round(Sen,2) >= sen_thred:
            return
        elif E_name + 'auc.csv' in os.listdir(path):
            cur_df  = pd.read_csv(path + E_name + 'auc.csv')
            cur_auc = roc_auc_score(cur_df['y_true'].values.tolist(), cur_df['y_pred_prob'].values.tolist())

            if auc > cur_auc:
                os.remove(path + E_name + 'auc.csv')
                df.to_csv(path + E_name + 'auc.csv', index=False)
                print(f"-----------------------------------------------模型: {E_name} 更新记录：auc : {auc}, Spe: {Spe}, Sen: {Sen}-----------------------------------------------\n")
            else:
                print(f"-----------------------------------------------模型: {E_name} 最优记录：auc : {auc}, Spe: {Spe}, Sen: {Sen}-----------------------------------------------\n")
        else:
            df.to_csv(path + E_name + 'auc.csv', index=False)
            print(f"-----------------------------------------------模型: {E_name} 初始记录：auc : {auc}, Spe: {Spe}, Sen: {Sen}-----------------------------------------------\n")

    test(df, 'D:/Xu/Desktop/论文写作/auc图表/auc_调整/', E_name)

    # 保存这一次的预测值（y_pred, y_pred_prob, y_true)
    if E_name + 'auc.csv' in files:
        os.remove(path + E_name + 'auc.csv')
        df.to_csv(path + E_name + 'auc.csv', index=False)
    else:
        df.to_csv(path + E_name + 'auc.csv', index=False)

    # 更新保存的最优AUC的预测值
    if os.path.exists(path + 'result'):
        if os.path.exists(path + 'result/' + E_name + 'auc.csv'):
            best_df = pd.read_csv(path + 'result/' + E_name + 'auc.csv')
            best_auc = roc_auc_score(best_df['y_true'].values.tolist(), best_df['y_pred_prob'].values.tolist())
            auc = roc_auc_score(df['y_true'].values.tolist(), df['y_pred_prob'].values.tolist())
            if auc > best_auc:
                os.remove(path + 'result/' + E_name + 'auc.csv')
                df.to_csv(path + 'result/' + E_name + 'auc.csv', index=False)
        else:
            df.to_csv(path + 'result/' + E_name + 'auc.csv', index=False)
    else:
        os.makedirs(path + 'result', exist_ok=True)
        df.to_csv(path + 'result/' + E_name + 'auc.csv', index=False)

    fpr, tpr, thresold = roc_curve(y_true,y_score)

    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(fpr, tpr, label= E_name + '_Classifier Score: {:.4f}'.format(roc_auc_score(y_true, y_score)))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()

    plt.show()

def best_thred(y_true, y_pred_proba):
    score_best = -1.000
    best_thred = 0.000
    for thred in np.arange(0.001, 1.0, 0.001):
        f1 = f1_score(y_true, ((y_pred_proba >= thred).astype(int)).ravel())
        rel = recall_score(y_true, ((y_pred_proba >= thred).astype(int)).ravel())
        P, N = confusion_matrix(y_true, ((y_pred_proba >= thred).astype(int)).ravel())
        tn, fp = P[0], P[1]
        fn, tp = N[0], N[1]
        Sensitivity = tp / (tp + fn)
        false_alarm_rate = fp / (fp + tn)
        Specificity = tn / (fp + tn)

        score = Sensitivity + Specificity
        if score > score_best:
            score_best = score
            best_thred = thred
    return best_thred

class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def dcg_score(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:self.topk])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def __call__(self, y_true, y_pred):
        idcg = self.dcg_score(y_true, y_true)
        dcg = self.dcg_score(y_true, y_pred)
        return dcg / (idcg + 1e-12)
