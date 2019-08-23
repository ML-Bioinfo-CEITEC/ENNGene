import os

import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD, RMSprop, Adam
import math
import matplotlib.pyplot as plt
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe

from ..networks.simple_conv_class import SimpleConvClass
from ..utils.dataset import Dataset
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand


# TODO fix imports in all the files to be consistent (relative vs. absolute)
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'make_datasets/tests'))

# TODO add logger


class Train(Subcommand):

    def __init__(self):
        help_message = '''deepnet <subcommand> [<args>]
            Train a model on preprocessed files.
            '''
        parser = self.create_parser(help_message)
        super().__init__(parser)

        self.train_dir = os.path.join(self.output_folder, 'training')
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)

        if type(self.args.branches) == list:
            self.branches = self.args.branches
        else:
            self.branches = [self.args.branches]

        if self.args.datasets == '-':
            # TODO read from STDIN ?
            pass
        else:
            # labels are the same in all the datasets, it is sufficient to read and encode them only once
            self.labels = seq.onehot_encode_alphabet(Dataset.load_from_file(self.args.datasets[0]).labels())
            self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = \
                self.parse_data(self.args.datasets, self.branches, self.labels)
        self.dims = self.get_dims(self.train_x, self.branches)

        self.network = self.args.network
        if self.args.hyper_tuning:
            self.hyper_tuning = self.args.hyper_tuning
            self.tune_rounds = self.args.tune_rounds
        else:
            self.hyper_tuning = False

        self.metric = self.args.metric
        self.loss = self.args.loss
        self.lr = self.args.lr
        self.lr_scheduler = self.args.lr_scheduler
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs

        self.hyperparams = {
            "dropout": self.args.dropout,
            "conv_num": self.args.conv_num,
            "kernel_size": self.args.kernel_size,
            "dense_num": self.args.dense_num,
            "dense_units": self.args.dense_units,
            "filter_num": self.args.filter_num
        }

        self.optimizer = self.args.optimizer

    def create_parser(self, message):
        parser = self.initialize_parser(message)

        parser.add_argument(
            "--datasets",
            required=True,
            nargs='+',
            help="Files containing preprocessed Dataset objects, omit for STDIN.",
            default='-'
        )
        parser.add_argument(
            "--batch_size",
            action="store",
            default=256,
            help="Batch Size. Default=256",
            type=int
        )
        parser.add_argument(
            "--branches",
            action='store',
            choices=['seq', 'cons', 'fold'],
            nargs='+',
            default="seq",
            help="Branches. [default: 'seq']"
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
            "--lr_scheduler",
            action="store",
            default=False,
            help="Whether to use learning rate scheduler (decreasing lr from 0.1). Default=False",
            type=bool
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
            "--kernel_size",
            action="store",
            default=4,
            help="Kernel size for convolutional layers. Default=4",
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
            "--dense_units",
            action="store",
            default=64,
            help="Number of units in first dense layer. Each next dense layer gets half the units. Default=64",
            type=int
        )
        parser.add_argument(
            "--epochs",
            action="store",
            default=3,  # 600,
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
        parser.add_argument(
            "--tune_rounds",
            action="store",
            default=5,
            help="Maximal number of hyperparameter tuning rounds. --hyper_tuning must be True."
        )
        parser.add_argument(
            "--optimizer",
            action='store',
            choices=['sgd', 'rmsprop', 'adam'],
            default='sgd',
            help="Optimizer to be used. Default = 'sgd'."
        )
        parser.add_argument(
            "--network",
            action='store',
            choices=['simpleconv'],
            default='simpleconv',
            help="Predefined network architecture to be used. Default = 'simpleconv' (simple convolutional network)."
        )
        parser.add_argument(
            "--metric",
            action='store',
            choices=['accuracy'],
            default=['accuracy'],
            help="Metric to be used during training. Default = 'accuracy'."
        )
        parser.add_argument(
            "--loss",
            action='store',
            choices=['categorical_crossentropy'],
            default=['categorical_crossentropy'],
            help="Loss function to be used during training. Default = 'categorical_crossentropy'."
        )
        return parser

    @staticmethod
    def parse_data(dataset_files, branches, alphabet):
        # TODO D What was the meaning of the random_argument_generator ?
        # for argument in random_argument_generator(shuffles=1):

        datasets = set()
        for file in dataset_files:
            datasets.add(Dataset.load_from_file(file))

        split_datasets = ['train_x', 'train_y', 'valid_x', 'valid_y', 'test_x', 'test_y']
        split_datasets_dict = dict(zip(split_datasets, [[] for _ in range(len(split_datasets))]))

        dictionary = {'train': {'x': split_datasets_dict['train_x'], 'y': split_datasets_dict['train_y']},
                      'validation': {'x': split_datasets_dict['valid_x'], 'y': split_datasets_dict['valid_y']},
                      'test': {'x': split_datasets_dict['test_x'], 'y': split_datasets_dict['test_y']}}

        for branch in branches:
            for dataset in datasets:
                # to ensure the right order of data within arrays
                if dataset.branch != branch: continue
                values = dataset.values()
                labels = dataset.labels(alphabet=alphabet)

                dictionary[dataset.category]['x'].append(values)
                dictionary[dataset.category]['y'].append(labels)

        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            [np.array(split_dataset) for split_dataset in split_datasets_dict.values()]

        print(train_x.shape)
        return [train_x, valid_x, test_x, train_y, valid_y, test_y]

    @staticmethod
    def get_dims(data, branches):
        # TODO for now, only 1D data are taken into account
        # Assuming fixed order of the branches
        dims = {}
        for i, branch in enumerate(branches):
            seq_len = len(data[i][0])  # e.g. 10 for a sequence of 10 bases
            seq_size = len(data[i][0][0])  # e.g. 5 for one-hot encoded bases or 1 for conservation score
            dims.update({branch: [seq_len, seq_size]})

        return dims

    def run(self):
        # define model based on chosen architecture
        if self.network == 'simpleconv':
            network = SimpleConvClass(
                dims=self.dims, branches=self.branches, hyperparams=self.hyperparams, labels=self.labels)
        else:
            raise Exception  # should not be possible to happen, later add other architectures here

        # hyperparameter tuning (+ export/import)
        if self.hyper_tuning:
            # TODO what is the model returned by hyperas? Can we use it?
            # Or I'm going to create new model using the given hyperparameters?
            hyperparams, model = self.tune_hyperparameters(network, self.tune_rounds, self.dims)
            # TODO enable to export best hyperparameters for future use (then pass them as one argument within file?)
        else:
            model = network.build_model(tune=False)
            hyperparams = self.hyperparams

        # optimizer definition
        optimizer = self.create_optimizer(self.optimizer, self.lr)

        # model compilation
        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metric)

        # training & testing the model (fit)
        callbacks = self.create_callbacks(self.train_dir, network.name, self.lr_scheduler)
        history = self.train(model, self.epochs, self.batch_size, callbacks,
                             self.train_x, self.valid_x, self.train_y, self.valid_y)
        test_results = self.test(model, self.batch_size, self.test_x, self.test_y)

        # plot metrics
        self.plot_graph(history, self.metric, self.metric.capitalize(), self.train_dir)
        self.plot_graph(history, self.loss, self.loss.capitalize(), self.train_dir)

        # export results

        # TODO save results & plots to the files
        # TODO save test results
        # TODO save resulting model

        # TODO return something that can be passed on to the next module

    @staticmethod
    def tune_hyperparameters(network, tune_rounds, data):
        # using hyperas package
        model = network.build_model(tune=True)
        params, best_model = optim.minimize(model=model, data=data,
                                            algo=tpe.suggest,
                                            max_evals=tune_rounds,
                                            trials=Trials())
        return params, best_model

    @staticmethod
    def step_decay(epoch):
        drop = 0.5
        epochs_drop = 10.0
        initial_lr = 0.01

        lr = initial_lr * math.pow(drop, math.floor(epoch + 1)/epochs_drop)

        return lr

    @staticmethod
    def create_callbacks(out_dir, net_name, scheduler):
        mcp = ModelCheckpoint(filepath=out_dir + "/{}.hdf5".format(net_name),
                              verbose=0,
                              save_best_only=True)

        earlystopper = EarlyStopping(monitor='val_loss',
                                     patience=40,
                                     min_delta=0,
                                     verbose=1,
                                     mode='auto')

        csv_logger = CSVLogger(out_dir + "/{}.log.csv".format(net_name),
                               append=True,
                               separator='\t')
        callbacks = [mcp, earlystopper, csv_logger]

        if scheduler:
            lr_scheduler = LearningRateScheduler(step_decay)
            callbacks.append(lr_scheduler)

        return callbacks

    @staticmethod
    def create_optimizer(chosen, learning_rate):
        if chosen == 'sgd':
            optimizer = SGD(
                lr=learning_rate,
                decay=1e-6,
                momentum=0.9,
                nesterov=True)
        elif chosen == 'rmsprop':
            optimizer = RMSprop(
                lr=learning_rate,
                rho=0.9,
                epsilon=None,
                decay=0.0
            )
        elif chosen == 'adam':
            optimizer = Adam(
                lr=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=False
            )

        return optimizer

    @staticmethod
    def train(model, epochs, batch_size, callbacks, train_x, valid_x, train_y, valid_y):
        history = model.fit(
            train_x,
            train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(valid_x, valid_y),
            callbacks=callbacks)

        return history

    @staticmethod
    def test(model, batch_size, test_x, test_y):
        test_results = model.evaluate(
            test_x,
            test_y,
            batch_size=batch_size,
            verbose=1,
            sample_weight=None)

        return test_results

    @staticmethod
    def plot_graph(history, metric, title, out_dir):
        # TODO separate class for plotting? probably combined with the Evaluate module
        val_metric = "val_{}".format(metric)

        plt.plot(history[metric])
        plt.plot(history[val_metric])

        # TODO is it necessary?
        # if metric == 'acc':
        #     plt.ylim(0.0, 1.0)
        # elif metric == 'loss':
        #     plt.ylim(0.0, max(max(history[metric]), max(history[val_metric])))

        plt.title("Model {}".format(title))
        plt.ylabel(title)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig(out_dir + "/CNNonRaw.acc.png", dpi=300)
        plt.clf()

    # TODO Is this class needed?
    # class Config:
    #     data_file_path = None
    #     tmp_output_directory = None
