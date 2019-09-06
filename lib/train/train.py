from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD, RMSprop, Adam
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# from hyperas import optim
# from hyperopt import Trials, tpe

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
            self.labels = seq.onehot_encode_alphabet(list(set(Dataset.load_from_file(self.args.datasets[0]).labels())))
            self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = \
                self.parse_data(self.args.datasets, self.branches, self.labels)
        self.branch_shapes = self.get_shapes(self.train_x, self.branches)

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
            required=False,
            type=int
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
        parser.add_argument(
            "--tune_rounds",
            action="store",
            default=5,
            type=int,
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
            default='accuracy',
            help="Metric to be used during training. Default = 'accuracy'."
        )
        parser.add_argument(
            "--loss",
            action='store',
            choices=['categorical_crossentropy'],
            default='categorical_crossentropy',
            help="Loss function to be used during training. Default = 'categorical_crossentropy'."
        )
        return parser

    @staticmethod
    def parse_data(dataset_files, branches, alphabet):
        datasets = set()
        for file in dataset_files:
            datasets.add(Dataset.load_from_file(file))

        dictionary = {}
        for dataset in datasets:
            dictionary.update({dataset.category: {}})
            values = []
            for branch in branches:
                values.append(dataset.values(branch))

            # Do not return data in an extra array if there's only one branch
            if len(values) == 1:
                values = values[0]

            dictionary[dataset.category].update({'values': values})
            dictionary[dataset.category].update({'labels': dataset.labels(alphabet=alphabet)})

        return [dictionary['train']['values'], dictionary['validation']['values'], dictionary['test']['values'],
                dictionary['train']['labels'], dictionary['validation']['labels'], dictionary['test']['labels']]

    # def get_data_model(self, network):
    #     def data():
    #         return self.train_x, self.train_y, self.test_x, self.test_y
    #
    #     # FIXME why does hyperas have issue with the indentation?
    #     def create_model(x_train, y_train, x_test, y_test):
    #         return network.tune_model(x_train, y_train)
    #
    #     return data, create_model
    #
    # def tune_params(self, network):
    #     data, create_model = self.get_data_model(network)
    #     best_run, best_model = optim.minimize(model=create_model,
    #                                           data=data,
    #                                           algo=tpe.suggest,
    #                                           max_evals=self.tune_rounds,
    #                                           trials=Trials())
    #     return best_run, best_model

    @staticmethod
    def get_shapes(data, branches):
        shapes = {}

        if len(branches) == 1:
            shapes.update({branches[0]: data.shape})
        else:
            # Assuming fixed order of the branches
            for i, branch in enumerate(branches):
                shapes.update({branch: data[i].shape})

        return shapes

    def run(self):
        # Define model based on chosen architecture
        if self.network == 'simpleconv':
            network = SimpleConvClass(
                branch_shapes=self.branch_shapes, branches=self.branches, hyperparams=self.hyperparams, labels=self.labels)
        else:
            raise Exception  # should not be possible to happen, later add other architectures here

        # Hyperparameter tuning (+ export/import)
        if self.hyper_tuning:
            # TODO enable to export best hyperparameters for future use (then pass them as one argument within file?)
            best_run, best_model = self.tune_params(network)
        else:
            model = network.build_model()
            hyperparams = self.hyperparams

        # Optimizer definition
        optimizer = self.create_optimizer(self.optimizer, self.lr)

        # Model compilation
        model.compile(
            optimizer=optimizer,
            loss=[self.loss],
            metrics=[self.metric])

        # Training & testing the model (fit)
        callbacks = self.create_callbacks(self.train_dir, network.name, self.lr_scheduler)

        print('Training the network')
        history = self.train(model, self.epochs, self.batch_size, callbacks,
                             self.train_x, self.valid_x, self.train_y, self.valid_y)

        # Plot metrics
        self.plot_graph(history.history, self.metric, self.metric.capitalize(), self.train_dir, network.name)
        self.plot_graph(history.history, 'loss', "Loss: {}".format(self.loss.capitalize()), self.train_dir, network.name)

        print('Testing the network')
        test_results = self.test(model, self.batch_size, self.test_x, self.test_y)
        print('[loss, acc]')
        print(test_results)
        # TODO save the test results ? (did not find that in the old code)

        # TODO return something that can be passed on to the next module

    @staticmethod
    def tune_hyperparameters(network, tune_rounds, data):
        # using hyperas package
        # model = network.build_model(tune=True)
        # params, best_model = optim.minimize(model=model, data=data,
        #                                     algo=tpe.suggest,
        #                                     max_evals=tune_rounds,
        #                                     trials=Trials())
        # return params, best_model
        pass

    @staticmethod
    def step_decay_schedule(initial_lr=1e-3, drop = 0.5, epochs_drop = 10.0):
        def schedule(epoch):
            return initial_lr * math.pow(drop, math.floor(epoch + 1) / epochs_drop)
        return LearningRateScheduler(schedule)

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
            # TODO probably does not make sense to use with ADAM and RMSProp?
            callbacks.append(Train.step_decay_schedule())

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
                epsilon=None,  # 10−8 for ϵ
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
    def plot_graph(history, metric, title, out_dir, network_name):
        # TODO separate class for plotting? probably combined with the Evaluate module
        # For some reason here it calls accuracy just 'acc'
        if metric == 'accuracy':
            metric = 'acc'
        val_metric = "val_{}".format(metric)
        file_name = "/{}.{}.png".format(network_name, metric)

        plt.plot(history[metric])
        plt.plot(history[val_metric])

        if metric == 'acc':
            plt.ylim(0.0, 1.0)
        elif metric == 'loss':
            plt.ylim(0.0, max(max(history[metric]), max(history[val_metric])))

        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig(out_dir + file_name, dpi=300)
        plt.clf()
