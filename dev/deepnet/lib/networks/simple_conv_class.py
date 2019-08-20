import keras
from hyperas.distributions import choice, uniform
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, \
    Activation, LSTM, Bidirectional, Input, BatchNormalization
from keras.models import Sequential, Model

from .network import Network


class SimpleConvClass(Network):
    # Classification network using few convolutional layers

    def __init__(self, dims={}, branches=[], hyperparams={}, labels=[]):
        super().__init__(dims=dims, branches=branches, hyperparams=hyperparams, labels=labels)
        # TODO omit completely or add something above just super call?
        return

    def build_model(self, tune=False):
        super().build_model(tune=tune)

        if tune:
            # conv_num = {{choice([1, 2, 3, 4])}} # Kept fixed for now to lower tuning time. default = 3

            # TODO kernel size and no. of filters does/not depend on the index of the convolutional layer?
            conv_filters = {{choice([16, 32, 64])}}
            # TODO in mustard the kernel sizes were 16, 30 and 20 per branch and decreasing with each layer
            conv_kernels = {{choice([1, 3, 5])}}
            dense_num = {{choice([1, 2, 3, 4])}}
            dense_units = {{choice([32, 64, 128])}}
            do_rate = {{uniform(0.2, 0.4)}}
        else:
            conv_filters = self.hyperparams['filter_num']
            conv_kernels = self.hyperparams['kernel_size']
            dense_num = self.hyperparams['dense_num']
            dense_units = self.hyperparams['dense_units']
            do_rate = self.hyperparams['dropout']

        inputs = []
        branches_models = []
        for branch in self.branches:
            # Create convolutional network for each branch separately

            # batch size left undefined, thus variable
            x = Input(shape=(self.dims[branch][0], self.dims[branch][1]))
            inputs.append(x)

            for convolution in range(0, self.hyperparams['conv_num']):
                x = Conv1D(filters=conv_filters, kernel_size=conv_kernels, strides=1,
                           padding="same")(x)
                x = LeakyReLU()(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=2, padding="same")(x)
                x = Dropout(rate=do_rate, noise_shape=None, seed=None)(x)
            branches_models.append(Flatten()(x))

        if len(self.branches) == 1:
            x = branches_models[0]
        else:
            x = keras.layers.concatenate(branches_models)

        # Continue to dense layers using concatenated results from convolution of the branches
        for dense in range(0, dense_num):
            units = int(dense_units / pow(2, dense))
            # Just ensure the number does not drop bellow the number of classes
            if units < len(self.labels):
                units = int(len(self.labels))

            x = Dense(units)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=do_rate, noise_shape=None, seed=None)(x)

        output = Dense(units=len(self.labels), activation="softmax")(x)
        model = Model(inputs, output)

        # validation_metric = np.amax(history[val_metric])
        # return {'loss': -validation_metric, 'status': STATUS_OK, 'model': model}
        return model
