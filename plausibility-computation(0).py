# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# -

nodes = pd.read_csv('data/nodes(0).csv', index_col=0)

nodes

node_embeddings = pd.read_csv('data/nodes_embedding_LINE(0).txt',
                              header=None,
                              sep='\t',
                              index_col=0,
                              names=['embedding'])
node_embeddings

# +
# consistency check

for i in nodes.index:
    assert i in node_embeddings.index
# -

nodes['type'].value_counts()

edges = pd.read_csv('data/edges(0).csv')
edges_freq = edges['type'].value_counts()
edges_freq_reduced = edges_freq[edges_freq > 100]
edges_freq_reduced

edges.head()


class OCSVM(BaseEstimator):
    def __init__(self, nu=.5, gamma='scale', verbose=False):
        self.nu = nu
        self.gamma = gamma
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.oc_classifier_ = OneClassSVM(nu=self.nu,
                                          gamma=self.gamma,
                                          verbose=self.verbose,
                                          cache_size=7000)
        X_pos = np.array([x for x, l in zip(X, y) if l==1])
        self.oc_classifier_.fit(X_pos)
        return self

    def predict(self, X):
        check_is_fitted(self, 'oc_classifier_')
        X = check_array(X)
        return self.oc_classifier_.predict(X)

    def score(self, X, y):
        check_is_fitted(self, 'oc_classifier_')
        X, y = check_X_y(X, y)
        return metrics.accuracy_score(y, self.oc_classifier_.predict(X))

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        
        self.oc_classifier_ = OneClassSVM(nu=self.nu, gamma=self.gamma)
        return self
    
    def plausibility(self, X):
        check_is_fitted(self, 'oc_classifier_')
        return self.oc_classifier_.decision_function(X)


# +
def get_embedding(node, embeddings):
    return np.array([float(e)
                     for e in embeddings.loc[node, 'embedding'].split()])

def get_examples(edges, embeddings):
    X = []

    for row in edges.iterrows():
        subj = row[1]['subject']
        subj_embedding = get_embedding(subj, embeddings)
        obj = row[1]['object']
        obj_embedding = get_embedding(obj, embeddings)
        X.append(np.hstack([subj_embedding, obj_embedding]))
    return np.array(X)


# -

def run_experiment(predicate, sample=None, serialize_dir=None):
    edges_positive = edges[edges['type'] == predicate]
    if sample is not None:
        edges_positive = edges_positive.sample(sample)

    edges_negative = edges[edges['type'] != predicate].sample(
                                                2 * len(edges_positive))
    num_predicates_pos = edges_positive['predicate'].nunique()
    num_predicates_neg = edges_negative['predicate'].nunique()
    X_pos = get_examples(edges_positive, node_embeddings)
    X_neg = get_examples(edges_negative, node_embeddings)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [-1] * len(X_neg))
    
    ocsvm = OCSVM()

    external_holdout = StratifiedShuffleSplit(n_splits=1,
                                             test_size=0.15,
                                             random_state=42)
    internal_holdout = StratifiedShuffleSplit(n_splits=1,
                                              test_size=0.15,
                                              random_state=42)

    for train_idx, test_idx in external_holdout.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        gamma_values = ['scale', 'auto'] + list(np.logspace(-10, 2, 11))
        gs = GridSearchCV(ocsvm,
                          param_grid={'nu': np.array([x for x in np.logspace(-8, 0, 9) if x != 1]),
                                     'gamma': gamma_values},
                          cv=internal_holdout,
                          n_jobs=-1,
                          error_score='raise')

        gs.fit(X_train, y_train)

        model = gs.best_estimator_
        
    if serialize_dir is not None:
        if not os.path.exists(serialize_dir):
            os.makedirs(serialize_dir)
        name = predicate.replace(' ', '-')
        print(f'serializing to {os.path.join(serialize_dir, f"{name}_model.pickle")}')
        path = os.path.join(serialize_dir, f'{name}_model.pickle')
        with open(path, 'wb') as f:
            pickle.dump(model, f)

        path = os.path.join(serialize_dir, f'{name}_X-test.pickle')
        with open(path, 'wb') as f:
            pickle.dump(X_test, f)

        path = os.path.join(serialize_dir, f'{name}_y-test.pickle')
        with open(path, 'wb') as f:
                pickle.dump(y_test, f)

    accuracy = model.score(X_test, y_test)
    precision = metrics.precision_score(y_test, model.predict(X_test))
    recall = metrics.recall_score(y_test, model.predict(X_test))
    f1 = metrics.f1_score(y_test, model.predict(X_test))
    specificity = metrics.recall_score(y_test, model.predict(X_test), pos_label=-1)
    print(f'predicate <{predicate}>')
    print(f'({num_predicates_pos} positive, '
          f'{num_predicates_neg} negative) -- '
          f'{len(X_pos)} edges '
          f'{"(sampled)" if sample is not None else ""}')
    print('test set results:')
    print(f'accuracy: {accuracy:.3f}')
    print(f'precision: {precision:.3f}')
    print(f'recall: {recall:.3f}')
    print(f'F1: {f1:.3f}')
    print(f'specificity: {specificity:.3f}')
          
    return model


model = {}
for predicate in edges_freq_reduced.index:
    sample = 500 if edges_freq_reduced[predicate] > 500 else None
    m = run_experiment(predicate, sample=sample, serialize_dir=None)
    print(20*'-')
    model[predicate] = m

for predicate in edges_freq_reduced.index:
    name = predicate.replace(' ', '-')

    with open(f'models/{name}_X-test.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open(f'models/{name}_y-test.pickle', 'rb') as f:
        y_test = pickle.load(f)

    bins = 30

    plt.subplot(131)
    plt.hist(model[predicate].plausibility(X_test[y_test==1]),
             bins=bins, density=True)
    plt.title('negative')
    plt.subplot(132)
    plt.hist(model[predicate].plausibility(X_test[y_test==-1]),
             bins=bins, density=True)
    plt.title('positive')

    plt.subplot(133)
    plt.hist(model[predicate].plausibility(X_test),
             bins=bins, density=True)
    plt.title('both')

    plt.suptitle(predicate)

    plt.show()


