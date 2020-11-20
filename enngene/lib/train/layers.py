from tensorflow.keras.layers import Dense, Dropout, Conv1D, GRU, LocallyConnected1D, MaxPooling1D, BatchNormalization, LeakyReLU, LSTM, RNN


class MyConv1D:

    @staticmethod
    def build(x, filters=40, kernel=4, batchnorm=False, dropout=None):
        x = Conv1D(filters=filters, kernel_size=kernel, padding='same')(x)
        x = LeakyReLU()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        if dropout: x = Dropout(rate=dropout, noise_shape=None, seed=None)(x)
        return x


class MyLocallyConnected1D:

    @staticmethod
    def build(x, filters=40, kernel=4, batchnorm=False, dropout=None):
        x = LocallyConnected1D(filters=filters, kernel_size=kernel, padding='same')(x)
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
    def build(x, units=32, batchnorm=False, dropout=None):
        x = LSTM(units)(x)
        if batchnorm: x = BatchNormalization()(x)
        if dropout: x = Dropout(rate=dropout, noise_shape=None, seed=None)(x)
        return x


class MyRNN:

    @staticmethod
    def build(x, units=32, batchnorm=False, dropout=None):
        x = RNN(units)(x)
        if batchnorm: x = BatchNormalization()(x)
        if dropout: x = Dropout(rate=dropout, noise_shape=None, seed=None)(x)
        return x


class MyGRU:

    @staticmethod
    def build(x, units=32, batchnorm=False, dropout=None):
        x = GRU(units)(x)
        if batchnorm: x = BatchNormalization()(x)
        if dropout: x = Dropout(rate=dropout, noise_shape=None, seed=None)(x)
        return x


LAYERS = {'Convolution layer': MyConv1D,
          'Locally Connected 1D layer': MyLocallyConnected1D,
          'Dense layer': MyDense}
BRANCH_LAYERS = {'Convolution layer': MyConv1D,
                 'Locally Connected 1D layer': MyLocallyConnected1D}
COMMON_LAYERS = {'Dense layer': MyDense}
# 'RNN': MyRNN,
# 'GRU': MyGRU,
# 'LSTM': MyLSTM,
