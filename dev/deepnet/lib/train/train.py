import sys
import os
import argparse
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D,\
     Activation, LSTM, Bidirectional, Input, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'make_datasets/tests'))

from setup import random_argument_generator, make_datasets


parser = argparse.ArgumentParser()
parser.add_argument(
            "--batch_size",
            action="store",
            default=256,
            help="Batch Size. Default=256",
            type=int
        )
parser.add_argument(
            "--branches",
            action="store",
            help="Names of branches",
            required=True,
            type=list
        )
parser.add_argument(
            "--dropout",
            action="store",
            default=0.2,
            help="Dropout. Default=0.2",
            required=False,
            type=int
        )
parser.add_argument(
            "--lr",
            action="store",
            default=0.0001,
            help="Learning Rate. Default=0.0001",
            required=False,
            type=int
        )
parser.add_argument(
            "--filter_num",
            action="store",
            default=40,
            help="Filter Number. Default=40",
            required=False
        )
parser.add_argument(
            "--conv_num",
            action="store",
            default=3,
            help="Number of convolutional layers. Default=3",
            type=int
            )
parser.add_argument(
            "--dense_num",
            action="store",
            default=3,
            help="Number of dense layers. Default=3",
            type=int
            )
parser.add_argument(
            "--epochs",
            action="store",
            default=600,
            help="Number of epochs to train",
            type=int
            )
parser.add_argument(
            "--hyper_tuning",
            action="store",
            default=False,
            help="Whether to enable hyper parameters tuning",
            type=bool
            )

args = parser.parse_args()


def create_data():
    for argument in random_argument_generator(shuffles=1):
        data = make_datasets(argument)
        train_x, test_x, train_y, test_y = data # ... how to correctly assign datasets?
        return train_x, test_x, train_y, test_y


class Config:
    data_file_path = None
    tmp_output_directory = None


class HyperTuning:
    #using hyperas
    def __init__(self):
        self.BRANCHES = args.branches
        self.train_x, self.test_x, self.train_y, self.test_y = create_data()



    def tune(self):
        params, self.best_model = optim.minimize(model=self.build_model, data=create_data,
                                                 algo=tpe.suggest,
                                                 max_evals=5,
                                                 trials=Trials())
        return params


    def build_model(self):
        input_data = []
        branches_models = []
        for branch in self.BRANCHES:
            x = Input(shape=(self.x_train[branch]['train'].dictionary.shape))
            input_data.append(x)
            conv_num = {{choice([1, 2, 3, 4])}}
            for convolution in range(0, conv_num):
                x = Conv1D(filters={{choice([8, 16, 32, 64])}}, kernel_size={{choice([1, 3, 5])}}, strides=1,
                           padding="same")(x)
                x = LeakyReLU()(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=2, padding="same")(x)
                x = Dropout(rate={{uniform(0, 1)}}, noise_shape=None, seed=None)(x)
            branches_models.append(Flatten()(x))

        if len(self.BRANCHES) == 1:
            model = branches_models[0]
        else:
            model = keras.layers.concatenate(branches_models)

        for dense in range(0, {{choice([1, 2, 3, 4])}}):
            model = Dense({{choice([16, 32, 64])}})(model)
            model = LeakyReLU()(model)
            model = BatchNormalization()(model)
            model = Dropout(rate={{uniform(0, 1)}}, noise_shape=None, seed=None)(model)
            model = Dense(units=len(self.BRANCHES), activation="softmax")(model)
            model = Model(input_data, model)

        sgd = SGD(
            lr={{choice([0.0001, 0.005])}},
            decay=1e-6,
            momentum=0.9,
            nesterov=True)

        model.compile(
            optimizer=sgd,
            loss="categorical_crossentropy",
            metrics=["accuracy"])

        mcp = ModelCheckpoint(filepath=Config.tmp_output_directory + "/CNNonRaw.hdf5",
                              verbose=0,
                              save_best_only=True)

        earlystopper = EarlyStopping(monitor='val_loss',
                                     patience=40,
                                     min_delta=0,
                                     verbose=1,
                                     mode='auto')

        csv_logger = CSVLogger(Config.tmp_output_directory + "/CNNonRaw.log.csv",
                               append=True,
                               separator='\t')

        history = model.fit(
            self.train_x[0],
            self.train_y,
            batch_size={{choice([16, 32, 64, 128])}},
            epochs={{choice([50, 100, 150])}},
            verbose=1,
            validation_data=(self.test_x, self.test_y),
            callbacks=[mcp, earlystopper, csv_logger])
        validation_acc = np.amax(history.history['val_acc'])
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}



class TrainModel(HyperTuning):
    def __init__(self):
        super().__init__()
        self.x_train, self.y_train = Config.data_file_path #!
        self.params = {
                "batch_size":args.batch_size,
                "dropout": args.dropout,
                "learn_rate":args.lr,
                "conv_num":args.conv_num,
                "dense_num":args.dense_num,
                "filter_num":args.filter_num,
                "epochs":args.epochs,
                "nodes":args.nodes
        }
        if args.hyper_tuning:
            self.params = self.tune()
            self.model = self.compile_model()
            self.plot_graphs()
        else:
            self.model = self.compile_model()
            self.plot_graphs()


    def compile_model(self):
        input_data = []
        branches_models = []
        for branch in self.BRANCHES:
            x = Input(shape = (self.x_train[branch]['train'].dictionary.shape))
            input_data.append(x)
            for convolution in range(0, self.params["conv_layers"] - 1):
                x = Conv1D(filters = self.params['filter_num'] * 2, kernel_size = 3, strides = 1, padding = "same")(x)
                x = LeakyReLU()(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size = 2, padding = "same")(x)
                x = Dropout(rate = self.params["dropout"], noise_shape = None, seed = None)(x)
            branches_models.append(Flatten()(x))


        if len(self.BRANCHES) == 1:
            model = branches_models[0]
        else:
            model = keras.layers.concatenate(branches_models)

        for dense in range(0, self.params['dense_num']):
            model = Dense(self.params['nodes'])(model)
            model = LeakyReLU()(model)
            model = BatchNormalization()(model)
            model = Dropout(rate = self.params['dropout'], noise_shape = None, seed = None)(model)
            model = Dense(units = len(self.BRANCHES), activation = "softmax")(model)
            model = Model(input_data, model)

        sgd = SGD(
            lr = self.params['learn_rate'],
            decay = 1e-6,
            momentum = 0.9,
            nesterov = True)

        model.compile(
                optimizer = sgd,
                loss = "categorical_crossentropy",
                metrics = ["accuracy"])

        mcp = ModelCheckpoint(filepath = Config.tmp_output_directory + "/CNNonRaw.hdf5",
                        verbose = 0,
                        save_best_only = True)

        earlystopper = EarlyStopping(monitor = 'val_loss',
                        patience = 40,
                        min_delta = 0,
                        verbose = 1,
                        mode = 'auto')

        csv_logger = CSVLogger(Config.tmp_output_directory + "/CNNonRaw.log.csv",
                    append=True,
                    separator='\t')

        self.history = model.fit(
        self.train_x[0], # what is train_x ?
        self.train_y, # what is train_y?
        batch_size = self.params['batch_size'],
        epochs = self.params['epochs'],
        verbose = 1,
        validation_data = (self.test_x, self.test_y), # valid_x, valid_y ?
        callbacks = [mcp, earlystopper, csv_logger])


    def plot_graphs(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.ylim(0.0, 1.0)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig(Config.tmp_output_directory + "/CNNonRaw.acc.png", dpi=300)
        plt.clf()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.ylim(0.0, max(max(self.history.history['loss']), max(self.history.history['val_loss'])))
        plt.title('Model Loss')
        plt.ylabel('Categorical Crossentropy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(Config.tmp_output_directory + "/CNNonRaw.loss.png", dpi=300)
        plt.clf()

