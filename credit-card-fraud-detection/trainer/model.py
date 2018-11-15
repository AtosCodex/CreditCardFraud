#!/usr/bin/env python

# Copyright 2018 Atos. All Rights Reserved.
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
# ==============================================================================

"""
Credit Card Fraud Detection using DNNClassifier estimator
"""

# import required libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# set global configuration
tf.logging.set_verbosity(tf.logging.INFO)

# set global variables
CSV_COLUMNS = ["Time","V1","V2","V3","V4","V5","V6","V7","V9","V10","V11","V12","V14","V16","V17","V18","V19","V21","Amount","Class"]
LABEL_COLUMN = "Class"
DEFAULTS = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0]]

# Input feature columns
def input_feature_columns():
    """
    Returns input feature columns
    """
    i_feature_columns = [
        tf.feature_column.numeric_column("Time", dtype=tf.float32),
        tf.feature_column.numeric_column("V1", dtype=tf.float32),
        tf.feature_column.numeric_column("V2", dtype=tf.float32),
        tf.feature_column.numeric_column("V3", dtype=tf.float32),
        tf.feature_column.numeric_column("V4", dtype=tf.float32),
        tf.feature_column.numeric_column("V5", dtype=tf.float32),
        tf.feature_column.numeric_column("V6", dtype=tf.float32),
        tf.feature_column.numeric_column("V7", dtype=tf.float32),
        tf.feature_column.numeric_column("V9", dtype=tf.float32),
        tf.feature_column.numeric_column("V10", dtype=tf.float32),
        tf.feature_column.numeric_column("V11", dtype=tf.float32),
        tf.feature_column.numeric_column("V12", dtype=tf.float32),
        tf.feature_column.numeric_column("V14", dtype=tf.float32),
        tf.feature_column.numeric_column("V16", dtype=tf.float32),
        tf.feature_column.numeric_column("V17", dtype=tf.float32),
        tf.feature_column.numeric_column("V18", dtype=tf.float32),
        tf.feature_column.numeric_column("V19", dtype=tf.float32),
        tf.feature_column.numeric_column("V21", dtype=tf.float32),
        tf.feature_column.numeric_column("Amount", dtype=tf.float32)
    ]
    return i_feature_columns

# Additional feature columns
def additional_feature_columns():
    """
    Returns additional enggineering features
    """
    a_feature_columns = [
        #tf.feature_column.numeric_column("V1_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V2_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V3_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V4_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V5_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V6_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V7_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V9_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V10_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V11_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V12_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V14_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V16_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V17_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V18_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V19_", dtype=tf.float32),
        #tf.feature_column.numeric_column("V21_", dtype=tf.float32),
        #tf.feature_column.numeric_column("Amount_max_fraud", dtype=tf.float32)
    ]
    return a_feature_columns

# Get feature columns
def get_feature_columns():
    """
    Get feature columns
    """
    i_feature_columns = input_feature_columns()
    a_feature_columns = additional_feature_columns()
    return i_feature_columns + a_feature_columns

# Create feature engineering function that will be used in the input and serving input functions
def add_more_features(features):
    """
    Add additional feature engineering columns
    """
    #tf_one = tf.constant(1.0)
    #tf_zero = tf.constant(0.0)
    #print('Type of tensor is {}'.format(type(features['V1'])))
    #print('Rank of tensor is {}'.format(tf.rank(features['V1'])))
    #print('Shape of tensor is {}'.format(tf.shape(features['V1'])))
    #features['V1_'] = tf.cond(features['V1'] < tf.constant(-3.0), lambda: tf_one, lambda: tf_zero)
    #features['V2_'] = tf.cond(features['V2'] > tf.constant(2.5), lambda: tf_one, lambda: tf_zero)
    #features['V3_'] = tf.cond(features['V3'] < tf.constant(-4.0), lambda: tf_one, lambda: tf_zero)
    #features['V4_'] = tf.cond(features['V4'] > tf.constant(2.5), lambda: tf_one, lambda: tf_zero)
    #features['V5_'] = tf.cond(features['V5'] < tf.constant(-4.5), lambda: tf_one, lambda: tf_zero)
    #features['V6_'] = tf.cond(features['V6'] < tf.constant(2.5), lambda: tf_one, lambda: tf_zero)
    #features['V7_'] = tf.cond(features['V7'] < tf.constant(-3.0), lambda: tf_one, lambda: tf_zero)
    #features['V9_'] = tf.cond(features['V9'] < tf.constant(-2.0), lambda: tf_one, lambda: tf_zero)
    #features['V10_'] = tf.cond(features['V10'] < tf.constant(-2.5), lambda: tf_one, lambda: tf_zero)
    #features['V11_'] = tf.cond(features['V11'] > tf.constant(2.0), lambda: tf_one, lambda: tf_zero)
    #features['V12_'] = tf.cond(features['V12'] < tf.constant(-2.0), lambda: tf_one, lambda: tf_zero)
    #features['V14_'] = tf.cond(features['V14'] < tf.constant(-2.5), lambda: tf_one, lambda: tf_zero)
    #features['V16_'] = tf.cond(features['V16'] < tf.constant(-2.0), lambda: tf_one, lambda: tf_zero)
    #features['V17_'] = tf.cond(features['V17'] < tf.constant(-2.0), lambda: tf_one, lambda: tf_zero)
    #features['V18_'] = tf.cond(features['V18'] < tf.constant(-2.0), lambda: tf_one, lambda: tf_zero)
    #features['V19_'] = tf.cond(features['V19'] > tf.constant(1.5), lambda: tf_one, lambda: tf_zero)
    #features['V21_'] = tf.cond(features['V21'] > tf.constant(0.6), lambda: tf_one, lambda: tf_zero) 
    #features['Amount_max_fraud'] = tf.cond(features['Amount'] <= tf.constant(2125.87), lambda: tf_zero, lambda: tf_one)
    return features

# Create input function to load data into datasets
def make_input_fn(filename, mode, batch_size = 512):
    """
    Returns input function either for training or evalution
    """
    def _input_fn():
        def _decode_csv(feature_row):
            columns = tf.decode_csv(feature_row, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_more_features(features), label
        
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(_decode_csv)
        #print(dataset)
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels
    return _input_fn

# Create serving input function to be able to serve predictions
def serving_input_fn():
    """
    Serving function
    """
    i_feature_columns = input_feature_columns()
    feature_placeholders = {
      column.name: tf.placeholder(tf.float32, [None]) for column in i_feature_columns
    }
    features = add_more_features(feature_placeholders.copy())
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

# Build the estimator
def build_estimator(output_dir, hidden_units):
    """
    Build DNNClassifier model for binary prediction
    """
    feature_columns = get_feature_columns()
    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=2,
        dropout=0.1,
        model_dir=output_dir)
    return estimator

# Create estimator train and evaluate function
def train_and_evaluate(args):
    """
    Train, Evaulate and Serve Model
    """
    estimator = build_estimator(args['output_dir'], args['hidden_units'].split(' '))
    
    train_spec = tf.estimator.TrainSpec(        
        input_fn = make_input_fn(
            filename = args['train_data_paths'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    
    eval_spec = tf.estimator.EvalSpec(        
        input_fn = make_input_fn(
            filename = args['eval_data_paths'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = 100,
        throttle_secs=args['eval_delay_secs'],
        exporters = exporter)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
