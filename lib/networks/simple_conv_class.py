import keras
import numpy as np
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input, BatchNormalization
from keras.models import Model

from .network import Network


class SimpleConvClass(Network):
    # Classification network using few convolutional layers

    def __init__(self, dims={}, branches=[], hyperparams={}, labels=[]):
        super().__init__(dims=dims, branches=branches, hyperparams=hyperparams, labels=labels)
        self.name = 'simpleCNN'
        return

    def build_model(self):
        super().build_model()
        # TODO kernel size and no. of filters does/not depend on the index of the convolutional layer?
        # TODO in mustard the kernel sizes were 16, 30 and 20 per branch and decreasing with each layer

        inputs = []
        branches_models = []
        for branch in self.branches:
            # Create convolutional network for each branch separately

            # batch size left undefined, thus variable
            x = Input(shape=(self.dims[branch][0], self.dims[branch][1]))
            inputs.append(x)

            for convolution in range(0, self.hyperparams['conv_num']):
                x = Conv1D(filters=self.hyperparams['filter_num'], kernel_size=self.hyperparams['kernel_size'], strides=1,
                           padding="same")(x)
                x = LeakyReLU()(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=2, padding="same")(x)
                x = Dropout(rate=self.hyperparams['dropout'], noise_shape=None, seed=None)(x)
            branches_models.append(Flatten()(x))

        if len(self.branches) == 1:
            x = branches_models[0]
        else:
            x = keras.layers.concatenate(branches_models)

        # Continue to dense layers using concatenated results from convolution of the branches
        for dense in range(0, self.hyperparams['dense_num']):
            units = int(self.hyperparams['dense_units'] / pow(2, dense))
            # Just ensure the number does not drop bellow the number of classes
            if units < len(self.labels):
                units = int(len(self.labels))

            x = Dense(units)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=self.hyperparams['dropout'], noise_shape=None, seed=None)(x)

        output = Dense(units=len(self.labels), activation="softmax")(x)
        model = Model(inputs, output)

        return model

    def tune_model(self, x_train, y_train):
        # conv_num = {{choice([1, 2, 3, 4])}} # Kept fixed for now to lower tuning time. default = 3

        inputs = []
        branches_models = []
        for branch in self.branches:
            # Create convolutional network for each branch separately

            # batch size left undefined, thus variable
            x = Input(shape=(self.dims[branch][0], self.dims[branch][1]))
            inputs.append(x)

            for convolution in range(0, self.hyperparams['conv_num']):
                x = Conv1D(filters={{choice([16, 32, 64])}}, kernel_size={{choice([1, 3, 5])}}, strides=1,
                           padding="same")(x)
                x = LeakyReLU()(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=2, padding="same")(x)
                x = Dropout(rate={{uniform(0.2, 0.4)}}, noise_shape=None, seed=None)(x)
            branches_models.append(Flatten()(x))

        if len(self.branches) == 1:
            x = branches_models[0]
        else:
            x = keras.layers.concatenate(branches_models)

        # Continue to dense layers using concatenated results from convolution of the branches
        for dense in range(0, {{choice([1, 2, 3, 4])}}):
            units = int({{choice([32, 64, 128])}} / pow(2, dense))
            # Just ensure the number does not drop bellow the number of classes
            if units < len(self.labels):
                units = int(len(self.labels))

            x = Dense(units)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(rate={{uniform(0.2, 0.4)}}, noise_shape=None, seed=None)(x)

        output = Dense(units=len(self.labels), activation="softmax")(x)
        model = Model(inputs, output)

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

        result = model.fit(x_train, y_train,
                           batch_size={{choice([64, 128, 256])}},
                           epochs=2,
                           verbose=2,
                           validation_split=0.1)

        #get the highest validation accuracy of the training epochs
        validation_acc = np.amax(result.history['val_acc'])
        print('Best validation acc of epoch:', validation_acc)

        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
