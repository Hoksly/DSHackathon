from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import metrics
import os

PATH = "models"
EPOCHS = 50

try:
    os.mkdir(PATH)
except:
    pass


class DeepModel(tf.keras.models.Sequential):

    def __init__(self, model_file=None,
                 loss_func='mean_absolute_error',
                 n_input_layer=305,
                 number_of_hidden_layers=5,
                 neurons_hidden_layer=305,
                 neurons_output_layer=4,
                 dropout=False,
                 dropout_rate=0.2,
                 metrics=[
                     metrics.RootMeanSquaredError(),
                     metrics.MeanAbsoluteError(),
                     metrics.MeanAbsolutePercentageError()],
                 * args, **kwargs):

        super(DeepModel, self).__init__(*args, **kwargs)

        self.add(tf.keras.layers.Dense(n_input_layer, activation='relu'))

        for i in range(1, number_of_hidden_layers+1):
            self.add(tf.keras.layers.Dense(
                neurons_hidden_layer / i, activation='relu'))
            if dropout:
                self.add(tf.keras.layers.Dropout(dropout_rate))

        self.add(tf.keras.layers.Dense(
            neurons_output_layer, activation='linear'))

        if (loss_func == 'mean_absolute_error'):
            self.compile(optimizer='Adam',
                         loss='mean_absolute_error', metrics=metrics)

        elif (loss_func == 'mean_squared_error'):
            self.compile(optimizer='Adam',
                         loss='mean_squared_error', metrics=metrics)

        elif (loss_func == 'mean_squared_logarithmic_error'):
            self.compile(
                optimizer='Adam', loss='mean_squared_logarithmic_error', metrics=metrics)


def get_data(path: str):
    df = pd.read_csv(path)

    df = df.dropna()
    df.describe()

    platform_dummies = pd.get_dummies(df['platform'], drop_first=True)
    df = df.drop(['platform'], axis=1)
    df = pd.concat([df, platform_dummies], axis=1)

    media_source_dummies = pd.get_dummies(df['media_source'], drop_first=True)
    df = df.drop(['media_source'], axis=1)
    df = pd.concat([df, media_source_dummies], axis=1)

    country_code_dummies = pd.get_dummies(df['country_code'], drop_first=True)
    df = df.drop(['country_code'], axis=1)
    df = pd.concat([df, country_code_dummies], axis=1)

    df = df.drop(['install_date'], axis=1)

    Y = df[['target_sub_ltv_day30', 'target_iap_ltv_day30',
            'target_ad_ltv_day30']]

    df.drop(['target_sub_ltv_day30', 'target_iap_ltv_day30',
            'target_ad_ltv_day30', 'target_full_ltv_day30'], axis=1, inplace=True)

    return df, Y


def split(X, y):
    return train_test_split(X, y, test_size=0.001)


def preprocess(X_train, X_test):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


X, y = get_data('data.csv')

pd_X_train, pd_X_test, pd_y_train, pd_y_test = split(X, y)
del X, y


X_train = pd_X_train.values


del pd_X_train
# Some coments, nevermind


def train_several_models():
    for loss_function in ['mean_absolute_error', 'mean_squared_error', 'mean_squared_logarithmic_error']:
        for dropout in [True, False]:
            for n_hidden_layers in [1, 3, 5, 7]:
                for target in ['target_sub_ltv_day30', 'target_iap_ltv_day30',
                               'target_ad_ltv_day30']:

                    cur_model_folder = "model_" + loss_function + "_" + \
                        str(n_hidden_layers) + \
                        ("_dropout_" if (dropout) else "_nodropout_") + target

                    os.mkdir(PATH + '/' + cur_model_folder)

                    print(cur_model_folder)

                    model = DeepModel(loss_func=loss_function,
                                      number_of_hidden_layers=n_hidden_layers,
                                      dropout=dropout)

                    single_y_train = pd_y_train[target]
                    singl_y_train_values = single_y_train.values

                    del single_y_train

                    history = model.fit(
                        X_train, singl_y_train_values, verbose=1, epochs=EPOCHS, validation_data=(pd_X_test, pd_y_test[target]))

                    log_folder = PATH + '/' + cur_model_folder + "/logs"
                    os.mkdir(log_folder)

                    history = pd.DataFrame(history.history)

                    hist_json_file = log_folder + '/history.json'
                    with open(hist_json_file, mode='w') as f:
                        history.to_json(f)

                    hist_csv_file = log_folder + '/history.csv'
                    with open(hist_csv_file, mode='w') as f:
                        history.to_csv(f)

                    model.save(PATH + '/' + cur_model_folder + '/model')


train_several_models()
