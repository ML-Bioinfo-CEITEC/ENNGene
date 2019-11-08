import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

from .network import Network


class SimpleConvClass(Network):
    # Classification network using few convolutional layers

    def __init__(self, branch_shapes=None, branches=None, hyperparams=None, labels=None, epochs=int):
        super().__init__(branch_shapes=branch_shapes, branches=branches, hyperparams=hyperparams, labels=labels, epochs=epochs)
        self.name = 'simpleCNN'
        return

    def build_model(self):
        super().build_model()
        # TODO kernel size and no. of filters does/not depend on the index of the convolutional layer?
        # TODO in mustard the kernel sizes were 16, 30 and 20 per branch and decreasing with each layer
        # TODO should the filter numbers increase with each convolution?
        #  to progressively learn more discriminative features

        inputs = []
        branches_models = []
        for branch in self.branches:
            # Create convolutional network for each branch separately

            # batch size left undefined, thus variable
            x = Input(shape=(self.branch_shapes[branch][1:]))
            inputs.append(x)

            for convolution in range(0, self.hyperparams['conv_num']):
                x = Conv1D(filters=self.hyperparams['filter_num'], kernel_size=self.hyperparams['kernel_size'], strides=1,
                           padding='same')(x)
                x = LeakyReLU()(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=2, padding='same')(x)
                x = Dropout(rate=self.hyperparams['dropout'], noise_shape=None, seed=None)(x)
            branches_models.append(Flatten()(x))

        if len(self.branches) == 1:
            x = branches_models[0]
        else:
            x = tf.keras.layers.concatenate(branches_models)

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

        output = Dense(units=len(self.labels), activation='softmax')(x)

        if len(self.branches) == 1:
            model = Model(inputs[0], output)
        else:
            model = Model(inputs, output)

        return model

    def build_tunable_model(self, x_train, y_train, x_val, y_val, params):
    
        inputs = []
        branches_models = []
        for branch in self.branches:
            # Create convolutional network for each branch separately

            # batch size left undefined, thus variable
            x = Input(shape=(self.branch_shapes[branch][1:]))
            inputs.append(x)

            for convolution in range(0, int(params['conv_num'])):
                x = Conv1D(filters=int(params['filter_num']), kernel_size=int(params['kernel_size']), strides=1,
                           padding='same')(x)
                x = LeakyReLU()(x)
                x = BatchNormalization()(x)
                x = MaxPooling1D(pool_size=2, padding='same')(x)
                x = Dropout(rate=float(params['dropout']), noise_shape=None, seed=None)(x)
            branches_models.append(Flatten()(x))

        if len(self.branches) == 1:
            x = branches_models[0]
        else:
            x = tf.keras.layers.concatenate(branches_models)

        # Continue to dense layers using concatenated results from convolution of the branches
        for dense in range(0, int(params['dense_num'])):
            units = int(int(params['dense_units']) / pow(2, dense))
            # Just ensure the number does not drop bellow the number of classes
            if units < len(self.labels):
                units = int(len(self.labels))

            x = Dense(units)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=float(params['dropout']), noise_shape=None, seed=None)(x)

        output = Dense(units=len(self.labels), activation='softmax')(x)

        if len(self.branches) == 1:
            model = Model(inputs[0], output)
        else:
            model = Model(inputs, output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        out = model.fit(x_train, y_train, epochs=self.epochs, batch_size=50, verbose=0)

        return out, model