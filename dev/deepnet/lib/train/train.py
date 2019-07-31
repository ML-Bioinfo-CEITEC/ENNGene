import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe

from setup import random_argument_generator, make_datasets
from ..networks.simple_conv_class import SimpleConvClass
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

        # TODO create dir "training" in the output folder, or something like that, and the inside hierarchy

        self.branches = self.args.branches
        if self.args.datasets == '-':
            # TODO read from STDIN ?
            pass
        else:
            self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = self.parse_data(
                self.args.datasets)

        if self.args.hyper_tuning:
            self.hyper_tuning = self.args.hyper_tuning
            self.tune_rounds = self.args.tune_rounds

        self.hyperparams = {
            "batch_size": self.args.batch_size,
            "dropout": self.args.dropout,
            "learn_rate": self.args.lr,
            "conv_num": self.args.conv_num,
            "dense_num": self.args.dense_num,
            "filter_num": self.args.filter_num,
            "epochs": self.args.epochs,
            "nodes": self.args.nodes
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
            default='seq',
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
            "--dense_num",
            action="store",
            default=3,
            help="Number of dense layers. Default=3",
            type=int
        )
        parser.add_argument(
            "--epochs",
            action="store",
            default=600,
            help="Number of epochs to train",
            type=int
        )
        # TODO allow also defining number of hyperparameter tuning trials for Hyperas?
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
            choices=['sgd', 'rmsprop', 'adam'],  # TODO allow all Keras optimizers or choose just some?
            default='sgd',
            help="Optimizer to be used. Default = 'sgd'."
        )
        return parser

    @staticmethod
    def parse_data(dataset_files):
        # TODO read in datasets from the files and divide them to data (x) and labels (y), call the method per branch?
        train_x, valid_x, test_x, train_y, valid_y, test_y = dataset_files

        # TODO D What was the meaning of the random_argument_generator ?
        # for argument in random_argument_generator(shuffles=1):

        return [train_x, valid_x, test_x, train_y, valid_y, test_y]

    def run(self):
        # define model (separate class)
        # TODO use arg to choose the network architecture
        data = [self.train_x, self.valid_x, self.train_y, self.valid_y]
        network = SimpleConvClass(data=data, branches=self.branches, hyperparams=self.hyperparams)

        # hyperparameter tuning (+ export/import)
        if self.hyper_tuning:
            model = network.build_model(tune=True)
            hyperparams = self.tune_hyperparameters(model, self.tune_rounds, data)
            # TODO enable to export best hyperparameters for future use (then pass them as one argument within file?)
        else:
            model = network.build_model(tune=False)
            hyperparams = self.hyperparams

        # optimizer definition
        optimizer = self.create_optimizer(self.optimizer)

        # model compilation
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",  # TODO allow different one?
            metrics=["accuracy"])  # TODO use more useful metric? Define our own metrics (separate class)
        # Custom metrics can be passed at the compilation step. The function would need to take (y_true, y_pred)
        # as arguments and return a single tensor value.

        # training & testing the model (fit)
        callbacks = self.create_callbacks
        history = self.train(model, hyperparams, callbacks,
                             self.train_x, self.valid_x, self.train_y, self.valid_y)
        test_results = self.test(model, hyperparams['batch_size'], self.test_x, self.test_y)

        # plot metrics
        self.plot_graph(history, 'acc', 'Accuracy', self.out_dir)
        self.plot_graph(history, 'loss', 'Categorical crossentropy loss', self.out_dir)

        # export results

        # TODO save results & plots to the files
        # TODO save test results
        # TODO save resulting model

        # TODO return something that can be passed on to the next module

    @staticmethod
    def tune_hyperparameters(model, tune_rounds, data):
        # using hyperas package
        params, best_model = optim.minimize(model=model, data=data,
                                            algo=tpe.suggest,
                                            max_evals=tune_rounds,
                                            trials=Trials())
        return params  # best_model?

    @staticmethod
    def create_callbacks(out_dir):
        # TODO replace "/CNNonRaw.hdf5" with something dynamic
        mcp = ModelCheckpoint(filepath=out_dir + "/CNNonRaw.hdf5",
                              verbose=0,
                              save_best_only=True)

        earlystopper = EarlyStopping(monitor='val_loss',
                                     patience=40,
                                     min_delta=0,
                                     verbose=1,
                                     mode='auto')

        csv_logger = CSVLogger(out_dir + "/CNNonRaw.log.csv",
                               append=True,
                               separator='\t')

        return [mcp, earlystopper, csv_logger]

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
        # define other ...

        return optimizer

    @staticmethod
    def train(model, hyperparameters, callbacks, train_x, valid_x, train_y, valid_y):
        history = model.fit(
            train_x,
            train_y,
            batch_size=hyperparameters['batch_size'],
            epochs=hyperparameters['epochs'],
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
        if metric == 'acc':
            plt.ylim(0.0, 1.0)
        elif metric == 'loss':
            plt.ylim(0.0, max(max(history[metric]), max(history[val_metric])))

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
