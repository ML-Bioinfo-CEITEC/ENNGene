import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

from .layers import LAYERS


class ModelBuilder:

    def __init__(self, branches=None, labels=None, branch_shapes=None, branches_layers=None, common_layers=None):
        if not branches and labels and branch_shapes and branches_layers and common_layers:
            raise
        self.branches = branches
        self.labels = labels
        self.branch_shapes = branch_shapes
        self.branches_layers = branches_layers
        self.common_layers = common_layers

    def build_model(self):
        inputs = []
        branches_models = []
        for branch in self.branches:
            # batch size left undefined, thus variable
            x = Input(shape=(self.branch_shapes[branch][1:]))
            inputs.append(x)

            for layer in self.branches_layers[branch]:
                x = LAYERS[layer['name']]().build(x, **layer['args'])
            branches_models.append(Flatten()(x))

        if len(self.branches) == 1:
            x = branches_models[0]
        else:
            x = tf.keras.layers.concatenate(branches_models)

        for layer in self.common_layers:
            x = LAYERS[layer['name']]().build(x, **layer['args'])

        output = Dense(units=len(self.labels), activation='softmax')(x)

        if len(self.branches) == 1:
            model = Model(inputs[0], output)
        else:
            model = Model(inputs, output)

        return model
