import copy
import logging
import os

import numpy as np
import pandas
import shap

from sklearn.base import(
    ClassifierMixin,
    TransformerMixin,
    clone,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils import Bunch, _print_elapsed_time
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from numbers import Integral
from sklearn.exceptions import NotFittedError
from sklearn.utils._estimator_html_repr import _VisualBlock

from custom_func.metrics import evaluate_metrics
import matplotlib.pyplot as plt


def _fit_single_estimator(estimator,
                          X_train,
                          y_train,
                          sample_weight=None,
                          message_clsname=None,
                          message=None,
                          est_names=None,
                          **kwargs):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights.".format(
                        estimator.__class__.__name__
                    )
                ) from exc
            raise
    else:
        with _print_elapsed_time(message_clsname, message):
            print("--------------------model" + est_names[-2:] + ":" + est_names[:-2] + "开始训练--------------------\n")
            estimator.fit(X_train, y_train)
            print("--------------------model" + est_names[-2:] + ":" + est_names[:-2] + "训练结束--------------------\n")
    return estimator

class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    _parameter_constraints: dict = {
        "estimators": [list],
        "weights": ["array-like", None],
        "n_jobs": [None, Integral],
        "verbose": ["verbose"],
    }

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return f"({idx} of {total}) Processing {name}"

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] != "drop"]

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([est.predict(X) for est in self.estimators_]).T


    def fit(self, X, y, sample_weight=None):
        """Get common fit operations."""
        names, clfs = self._validate_estimators()

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal; got"
                f" {len(self.weights)} weights, {len(self.estimators)} estimators"
            )

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(clf),
                X,
                y,
                sample_weight=sample_weight,
                message_clsname="Voting",
                message=self._log_message(names[idx], idx + 1, len(clfs)),
            )
            for idx, clf in enumerate(clfs)
            if clf != "drop"
        )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

            if hasattr(current_est, "feature_names_in_"):
                self.feature_names_in_ = current_est.feature_names_in_

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return class labels or probabilities for each estimator.

        Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            Input samples.

        y : ndarray of shape (n_samples,), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return super().fit_transform(X, y, **fit_params)

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.estimators_[0].n_features_in_

    def _sk_visual_block_(self):
        names, estimators = zip(*self.estimators)
        return _VisualBlock("parallel", estimators, names=names)

    def _more_tags(self):
        return {"preserves_dtype": []}

class Voting(ClassifierMixin, _BaseVoting):

    def __init__(
        self,
        estimators,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
        feature_map=None,
        **kwargs):

        super().__init__(estimators=estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose
        self.feature_map = feature_map
        self.est_names = list(name for name, clf in self.estimators)
        self.est = list(clf for name, clf in self.estimators)
        self.params = kwargs

    def fit(self, X_train, y_train, sample_weight=None):


        # 检查 y 值是否存在 numric
        check_classification_targets(y_train)

        if isinstance(y_train, np.ndarray) and len(X_train.shape) > 1 and y_train.shape[1] > 1:
            raise NotImplementedError(
                "Multilabel and multi-output classification is not supported."
            )

        self.le_ = LabelEncoder().fit(y_train)
        self.classes_ = self.le_.classes_
        y = self.le_.transform(y_train)

        """Get common fit operations."""
        names = tuple(name for name, clf in self.estimators)
        clfs = tuple(clf for name, clf in self.estimators)

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal; got"
                f" {len(self.weights)} weights, {len(self.estimators)} estimators"
            )

        parallel = Parallel(n_jobs=self.n_jobs)
        self.estimators_ = parallel(delayed(_fit_single_estimator)
                            (self.est_clone(names[idx], clf),
                            X_train,
                            y_train,
                            sample_weight=sample_weight,
                            message_clsname="Voting",
                            message=self._log_message(names[idx], idx + 1, len(clfs)),
                            est_names=self.est_names[idx],
                            **self.params)

            for idx, clf in enumerate(clfs)
            if clf != "drop"
        )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

            # 判断对象（current_est）是否具有该属性（feature_names_in_）
            if hasattr(current_est, "feature_names_in_"):
                self.feature_names_in_ = current_est.feature_names_in_

        return self

    def predict(self, X, data_generator):
        """Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X, data_generator), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )

        maj = self.le_.inverse_transform(maj)

        return maj

    def predict_proba(self, X):
        check_is_fitted(self)

        if type(X) is np.ndarray and self.shap_columns is not None:
            X = pandas.DataFrame(X, columns=self.shap_columns)

        y_pred_proba = self._collect_probas(X)
        weights = self._weights_not_none
        avg = np.average(
            y_pred_proba, axis=0, weights= weights
        )
        return avg

    def _collect_probas(self, X):
        """Collect results from clf.predict calls."""
        prob = []
        label = self.params['label_col']['name']
        for ind, clf in enumerate(self.estimators_):
            if self.est_names[ind][0:3].upper() in ['MLP', 'mlp']:
                pred_prob_1 = np.array(clf.predict_proba(X)[label])
                pred_prob_0 = np.array([1-x for x in pred_prob_1])
                pred_prob = np.concatenate([pred_prob_0.reshape(-1, 1), pred_prob_1.reshape(-1, 1)], axis=1)
            else:
                pred_prob = clf.predict_proba(X)
            prob.append(pred_prob)
        return np.asarray(prob)

    def _check_voting(self):
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={repr(self.voting)}"
            )
        return True

    def transform(self, X):

        check_is_fitted(self)

        if self.voting == "soft":
            probas = self._collect_probas(X)
            if not self.flatten_transform:
                return probas
            return np.hstack(probas)

        else:
            return self._predict(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        if self.voting == "soft" and not self.flatten_transform:
            raise ValueError(
                "get_feature_names_out is not supported when `voting='soft'` and "
                "`flatten_transform=False`"
            )

        _check_feature_names_in(self, input_features, generate_names=False)
        class_name = self.__class__.__name__.lower()

        active_names = [name for name, est in self.estimators if est != "drop"]

        if self.voting == "hard":
            return np.asarray(
                [f"{class_name}_{name}" for name in active_names], dtype=object
            )

        # voting == "soft"
        n_classes = len(self.classes_)
        names_out = [
            f"{class_name}_{name}{i}" for name in active_names for i in range(n_classes)
        ]
        return np.asarray(names_out, dtype=object)

    def est_clone(self, name, estimator):
        if name[0:3].upper() in ['MLP', 'mlp']:
            return copy.deepcopy(estimator)
        else:
            return clone(estimator)

    def evaluate(self, X_test, y_test, is_test):

        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_true = y_test

        val_logs = evaluate_metrics(y_true      = y_true,
                                    y_pred_proba= y_pred_proba,
                                    is_test     = is_test,
                                    **self.params)
        logging.info('===')
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

    def Ans_shap(self, model, X_test, rate):

        def shap_show_feature(feature_cols, X_test):
            show_features = []
            if 'SAT' in self.params['dataset_id']:
                unshow_features = ['Id', 'Md', 'Rs', 'Di', 'Kl']
            else:
                unshow_features = []
            for feature_name in shap_values.feature_names:
                if feature_name not in feature_cols[0]['name'] and feature_name not in feature_cols[1]['name']:
                    print(f"Warning: {feature_name} is not in feature_cols.")
                    continue
                if feature_name in unshow_features:
                    continue
                elif feature_cols[0]['type'] in ['numeric'] and feature_name in feature_cols[0]['name']:    # 连续变量
                    show_features.append(feature_name)
                elif feature_cols[1]['type'] in ['categorical'] and feature_name in feature_cols[1]['name']:# 离散变量，但类别小于 3
                    if len(X_test[feature_name].unique()) <= 2:
                        show_features.append(feature_name)
                    else:
                        unshow_features.append(feature_name)
                else:
                    print(f"Warning: {feature_name} is not in feature_cols.")
            print("shap unshow_features :", unshow_features)
            return show_features

        num_sample = int(len(X_test))
        dff_size = num_sample
        if int(num_sample * rate) > 0 and int(num_sample * rate) <= num_sample:
            instance_IDs = np.arange(num_sample)
            np.random.shuffle(instance_IDs)
            dff_size = dff_size - int(num_sample * rate)
            X_test_shap = X_test.loc[instance_IDs[dff_size:], :].reset_index()
            X_test_shap = X_test_shap.drop(columns='index')

            print('-'*20 + 'SHAP ANALYSE: sample_num:', len(X_test_shap), '开始' + '-'*20 + '\n')
            shap.initjs()
            self.shap_columns = X_test_shap.columns
            exp = shap.KernelExplainer(model.predict_proba, X_test_shap)
            shap_values = exp(X_test_shap)


            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            show_features = shap_show_feature(self.params['feature_cols'], X_test)
            shap_show_values = shap_values[:, show_features]

            if not os.path.exists('./shap/'):
                os.mkdir('./shap/')
            if 'SAT' in self.params['dataset_id']:
                title = 'Dateset-domestic'
            elif 'JAT' in self.params['dataset_id']:
                title = 'Dateset-foreign'
            else:
                raise ValueError('shap picture title is wrong')

            # plt.title(title, fontsize=12)
            shap.plots.waterfall(shap_show_values[0], show=False)
            plt.gcf().set_size_inches(8, 5)
            plt.subplots_adjust(left=0.4)
            plt.savefig('./shap/' + title + '-' + '-'.join(self.params['model_list']) + '-fig1')
            plt.show()

            # shap.image_plot(shap_values, X_test[:10])

            # plt.title(title, fontsize=12)
            shap.plots.bar(shap_show_values, show=False, max_display=10)
            # shap.summary_plot(shap_show_values, show=False, plot_type="bar")
            ax1 = plt.gca()
            # ax1.set_yticklabels([])          # 取消y轴上的特征显示
            # ax1.invert_xaxis()               # 图像左右翻转
            plt.grid(False)                  # 网格显示
            plt.gcf().set_size_inches(8, 5)
            plt.subplots_adjust(left=0.4)
            plt.savefig('./shap/' + title + '-' + '-'.join(self.params['model_list']) + '-fig2')
            plt.show()

            # plt.title(title, fontsize=12)
            shap.plots.beeswarm(shap_show_values, show=False, max_display=10)
            # shap.summary_plot(shap_show_values, show=False, plot_type="dot")
            plt.gcf().set_size_inches(8, 5)
            plt.subplots_adjust(left=0.4)
            plt.savefig('./shap/' + title + '-' + '-'.join(self.params['model_list']) + '-fig3')
            plt.show()
