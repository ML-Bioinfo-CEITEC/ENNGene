import keras
from hyperas.distributions import choice, uniform
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, \
    Activation, LSTM, Bidirectional, Input, BatchNormalization
from keras.models import Sequential, Model

from .network import Network


class SimpleConvClass(Network):
    # Classification network using few convolutional layers

    def __init__(self, data={}, branches=[], hyperparams={}):
        super().__init__(data=data, branches=branches, hyperparams=hyperparams)
        # TODO omit completely or add something above just super call?
        return

    def build_model(self, tune=False):
        super().build_model(tune=tune)

        if tune:
            # TODO Is it really good idea to leave the number of convolutional layers random?
            # Maybe use fixed number from an argument? Also it should be probably the same for all the branches.
            # conv_num = {{choice([1, 2, 3, 4])}}

            # TODO kernel size and no. of filters does/not depend on the index of the convolutional layer?
            conv_filters = {{choice([8, 16, 32, 64])}}
            conv_kernels = {{choice([1, 3, 5])}}
            dense_num = {{choice([1, 2, 3, 4])}}
            dense_units = {{choice([16, 32, 64])}}
            do_rate = {{uniform(0, 1)}}
        else:
            conv_filters = self.hyperparams['filter_num']
            conv_kernels = self.hyperparams['kernel_num']
            dense_num = self.hyperparams['dense_num']
            dense_units = self.hyperparams['dense_units']  # TODO keep it fixed or allow arg? Dependent on layer index?
            do_rate = self.hyperparams['dropout']

        input_data = []
        branches_models = []
        for branch in self.branches:
            # Create convolutional network for each branch separately

            # TODO What kind of data it needs here?
            x = Input(shape=self.data[branch]['train'].dictionary.shape)
            input_data.append(x)

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
            x = Dense(dense_units)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=do_rate, noise_shape=None, seed=None)(x)

        x = Dense(units=len(self.branches), activation="softmax")(x)
        model = Model(input_data, x)

        # validation_metric = np.amax(history[val_metric])
        # return {'loss': -validation_metric, 'status': STATUS_OK, 'model': model}
        return model
