from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, LeakyReLU


class MyConv1D:

    @staticmethod
    def build(x, filters=40, kernel=4, batchnorm=False, dropout=None):
        x = Conv1D(filters=filters, kernel_size=kernel, strides=1, padding='same')(x)
        x = LeakyReLU()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        if dropout: x = Dropout(rate=dropout, noise_shape=None, seed=None)(x)
        return x


class MyDense:

    @staticmethod
    def build(x, units=32, batchnorm=False, dropout=None):
        x = Dense(units)(x)
        x = LeakyReLU()(x)
        if batchnorm: x = BatchNormalization()(x)
        if dropout: x = Dropout(rate=dropout, noise_shape=None, seed=None)(x)
        return x


class MyLSTM:

    @staticmethod
    def build(x, batchnorm=False, dropout=None):
        return x


LAYERS = {'Convolutional layer': MyConv1D,
          'Dense layer': MyDense,
          'LSTM': MyLSTM}
