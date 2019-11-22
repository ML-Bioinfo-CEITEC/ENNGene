from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, LeakyReLU


class MyConv1D:

    @staticmethod
    def build(x):
        x = Conv1D(filters=40, kernel_size=4, strides=1, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Dropout(rate=0.3, noise_shape=None, seed=None)(x)
        return x


class MyDense:

    @staticmethod
    def build(x):
        x = Dense(32)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.3, noise_shape=None, seed=None)(x)
        return x


class MyLSTM:

    @staticmethod
    def build(x):
        return x


LAYERS = {'Convolutional layer': MyConv1D,
          'Dense layer': MyDense,
          'LSTM': MyLSTM}
