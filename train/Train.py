from numpy.lib.npyio import save
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import metrics
import os


PATH = "models"
TOTAL_EPOCHS = 6

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
                 dropout=True,
                 dropout_rate=0.2,
                 metrics=[
                     metrics.RootMeanSquaredError(),
                     metrics.MeanAbsoluteError(),
                     metrics.MeanAbsolutePercentageError()],
                 * args, **kwargs):

        super(DeepModel, self).__init__(*args, **kwargs)

        for i in range(1, number_of_hidden_layers+1):

            self.add(Dense(
                neurons_hidden_layer / i, activation='relu'))

            if dropout:
                self.add(Dropout(dropout_rate))

        self.add(Dense(
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


def save_model(model: DeepModel, history, path):
    os.mkdir(path)
    model.save(path + "/model")

    with open(path + '/history.csv', mode='w') as f:
        pd.DataFrame(history.history).to_csv(f)


class MasterModel:
    def __init__(self, model_sub: DeepModel, model_iap: DeepModel, model_ad: DeepModel) -> None:
        self.model_sub = model_sub
        self.model_iap = model_iap
        self.model_ad = model_ad

    def __init__(self):
        self.model_sub = DeepModel()
        self.model_iap = DeepModel()
        self.model_ad = DeepModel()

    def predict(self, X: np.ndarray):
        sub_prediction = self.model_sub.predict(X)
        iap_prediction = self.model_iap.predict(X)
        ad_prediction = self.model_ad.predict(X)
        return sub_prediction + iap_prediction + ad_prediction

    def fit(self, X_train, y_train, save_folder, verbose, epochs,
            validation_data, callbacks):

        print("-"*30)
        print("Training sub model")
        print("-"*30)
        history_sub = self.model_sub.fit(X_train, y_train[0], verbose=verbose, epochs=epochs,
                                         callbacks=callbacks)

        print("-"*30)
        print("Training iap model")
        print("-"*30)
        history_iap = self.model_iap.fit(X_train, y_train[1], verbose=verbose, epochs=epochs,
                                         callbacks=callbacks)

        print("-"*30)
        print("Training ad model")
        print("-"*30)
        history_ad = self.model_ad.fit(X_train, y_train[2], verbose=verbose, epochs=epochs,
                                       callbacks=callbacks)
        os.mkdir(save_folder)

        save_model(self.model_ad, history_ad, save_folder + '/model_ad')
        save_model(self.model_iap, history_iap, save_folder + '/model_iap')
        save_model(self.model_sub, history_sub, save_folder + '/model_sub')


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
            'target_ad_ltv_day30', 'target_full_ltv_day30']]

    df.drop(['target_sub_ltv_day30', 'target_iap_ltv_day30',
            'target_ad_ltv_day30', 'target_full_ltv_day30'], axis=1, inplace=True)

    return df, Y


def split(X, y):
    return train_test_split(X, y, test_size=0.001)


def train_several_models():
    for loss_function in ['mean_absolute_error', 'mean_squared_error', 'mean_squared_logarithmic_error']:
        for dropout in [True, False]:
            for n_hidden_layers in [1, 3, 5, 7]:
                for target in ['target_sub_ltv_day30', 'target_iap_ltv_day30',
                               'target_ad_ltv_day30']:

                    cur_model_folder = "model_" + loss_function + "_" + \
                        str(n_hidden_layers) + \
                        ("_dropout_" if (dropout) else "_nodropout_") + target

                    try:
                        os.mkdir(PATH + '/' + cur_model_folder)

                        print("Training:", cur_model_folder)

                    except:
                        print(cur_model_folder, "alrady exist, skipping...")
                        continue
                    stopping = tf.keras.callbacks.EarlyStopping(
                        monitor="var_loss", patience=3)

                    model = DeepModel(loss_func=loss_function,
                                      number_of_hidden_layers=n_hidden_layers,
                                      dropout=dropout)

                    single_y_train = pd_y_train[target]
                    singl_y_train_values = single_y_train.values

                    del single_y_train

                    history = model.fit(
                        X_train, singl_y_train_values, verbose=1, epochs=EPOCHS,
                        validation_data=(pd_X_test, pd_y_test[target]), callbacks=[stopping])

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


def train_master():
    Master = MasterModel()
    os.mkdir("Masters")

    for epochs in range(TOTAL_EPOCHS):
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor="var_loss", patience=3)

        Master.fit(X_train, (y_sub_train, y_iap_train, y_ad_train), "Masters/Master" + str(epochs*10 + 10), verbose=1, epochs=10,
                   validation_data=(pd_X_test, pd_y_test), callbacks=[stopping])


def train_main():
    model = DeepModel()

    os.mkdir("MonoModel")
    targets = pd_y_train["target_full_ltv_day30"].values

    for i in range(TOTAL_EPOCHS):

        stopping = tf.keras.callbacks.EarlyStopping(
            monitor="var_loss", patience=3)

        history = model.fit(
            X_train, targets, verbose=1, epochs=10,
            validation_data=(pd_X_test, pd_y_test["target_full_ltv_day30"]), callbacks=[stopping])

        save_model(model, history, "MonoModel/model" + str(i*10 + 10))


X, y = get_data('data.csv')

pd_X_train, pd_X_test, pd_y_train, pd_y_test = split(X, y)
del X, y


X_train = pd_X_train.values

y_sub_train = pd_y_train['target_sub_ltv_day30'].values
y_iap_train = pd_y_train['target_iap_ltv_day30'].values
y_ad_train = pd_y_train['target_ad_ltv_day30'].values
y_full_train = pd_y_train['target_full_ltv_day30'].values


del pd_X_train, pd_y_train
# Some coments, nevermind


train_master()
train_main()
