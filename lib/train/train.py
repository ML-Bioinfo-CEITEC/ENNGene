import datetime
import os
import sys
import logging
import math

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import talos as ta
from talos.utils.best_model import best_model
import pandas as pd

from ..networks.simple_conv_class import SimpleConvClass
from ..utils.dataset import Dataset
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand


# TODO fix imports in all the files to be consistent (relative vs. absolute)
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'make_datasets/tests'))

logger = logging.getLogger('main')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def transform_int_input(arg):
    try:
        transformed_input = [int(x.replace('[', '').replace(']', '')) for x in arg.split(',')]
        if len(transformed_input) == 1:
            return transformed_input[0]
        else:
            return transformed_input
    except:
        raise Exception('A numerical value or values is expected.')


def transform_float_input(arg):
    try:
        transformed_input = [float(x.replace('[', '').replace(']', '')) for x in arg.split(',')]
        
        if len(transformed_input) == 1:
            return transformed_input[0]
        else:
            return transformed_input
    except:
        raise Exception('A numerical value or values is expected.')      


class Train(Subcommand):

    def __init__(self):
        help_message = '''deepnet <subcommand> [<args>]
            Train a model on preprocessed files.
            '''
        parser = self.create_parser(help_message)
        super().__init__(parser)

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

        if self.args.hyper_tuning:
            self.hyper_tuning = self.args.hyper_tuning
            self.tune_rounds = self.args.tune_rounds
            self.experiment_name = self.args.experiment_name
            self.hyper_param_metric = self.args.hyper_param_metric
        else:
            self.hyper_tuning = False

        self.metric = self.args.metric
        self.loss = self.args.loss
        self.lr = self.args.lr
        self.optimizer = self.args.optimizer
        if self.optimizer == 'sgd':
            self.lr_scheduler = self.args.lr_scheduler
        else:
            self.lr_scheduler = False
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.tb = self.args.tb

        self.hyperparams = {
            "dropout": self.args.dropout,
            "conv_num": self.args.conv_num,
            "kernel_size": self.args.kernel_size,
            "dense_num": self.args.dense_num,
            "dense_units": self.args.dense_units,
            "filter_num": self.args.filter_num
        }

        if self.args.network == 'simpleconv':
            self.network = SimpleConvClass(
                branch_shapes=self.branch_shapes, branches=self.branches, hyperparams=self.hyperparams, labels=self.labels, epochs=self.epochs)
        else:
            raise Exception  # should not be possible to happen, later add other architectures here

        self.train_dir = os.path.join(self.output_folder, 'training',
                                      f'{self.network.name}_{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(self.train_dir)

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
            type=transform_float_input
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
            help="Whether to use learning rate scheduler (decreasing lr from 0.1). Applied only in combination \
                 with SGD optimizer. Default=False",
            type=str2bool
        )
        parser.add_argument(
            "--filter_num",
            action="store",
            default=40,
            help="Filter Number. Default=40",
            required=False,
            type=transform_int_input
        )
        parser.add_argument(
            "--conv_num",
            action="store",
            default=3,
            help="Number of convolutional layers. Default=3",
            type=transform_int_input
        )
        parser.add_argument(
            "--kernel_size",
            action="store",
            default=4,
            help="Kernel size for convolutional layers. Default=4",
            type=transform_int_input
        )
        parser.add_argument(
            "--dense_num",
            action="store",
            default=3,
            help="Number of dense layers. Default=3",
            type=transform_int_input
        )
        parser.add_argument(
            "--dense_units",
            action="store",
            default=64,
            help="Number of units in first dense layer. Each next dense layer gets half the units. Default=64",
            type=transform_int_input
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
            type=str2bool
        )
        parser.add_argument(
            "--tune_rounds",
            action="store",
            default=5,
            type=int,
            help="Maximal number of hyperparameter tuning rounds. --hyper_tuning must be True."
        )
        parser.add_argument(
            "--hyper_param_metric",
            action="store",
            choices=['acc'],
            default='acc',
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
        parser.add_argument(
            "--tb",
            default=False,
            help="Output TensorBoard file. Default = False.",
            type=bool
        )
        parser.add_argument(
            "--experiment_name",
            action='store',
            default='e1',
            help="Name of the hyperparameter tuning experiment."
        )
        return parser

    @staticmethod
    def parse_data(dataset_files, branches, alphabet):
        
        #TODO include metadata in the dataset names using hash
        logger_datasets = []
        for i in range(len(dataset_files)):
            logger_datasets.append(dataset_files[i].split('/')[-1])
        #TODO add hash to the dataset names so that we can distinguish between the datasets?
        logger.info('Using the following datasets: ' + str(logger_datasets))
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
        logger.info('Hyperparameters: ' + str(list(self.hyperparams.items())))
        logger.info('Chosen network architecture: ' + str(self.network.name))
        logger.info('Using the following branches: ' + str(self.branches))
        logger.info('Using hyperparameter tuning: ' + str(self.hyper_tuning))
        if self.hyper_tuning:
            logger.info('Tune rounds: ' + str(self.tune_rounds))
        logger.info('Learning rate: ' + str(self.lr))
        logger.info('Optimizer: ' + str(self.optimizer))
        if self.optimizer == 'adam':
            logger.info('Learning rate scheduler: ' + str(self.lr_scheduler))
        logger.info('Batch size: ' + str(self.batch_size))
        logger.info('Epoch rounds: ' + str(self.epochs))
        logger.info('Loss function: ' + str(self.loss))
        logger.info('Metric function: ' + str(self.metric))

        # Hyperparameter tuning (+ export/import)
        if self.hyper_tuning:
            self.tune_hyperparameters(train_x=self.train_x,
                                      train_y=self.train_y,
                                      valid_x=self.valid_x,
                                      valid_y=self.valid_y,
                                      network=self.network.build_tunable_model,
                                      experiment_name=self.experiment_name,
                                      tune_rounds=self.tune_rounds,
                                      metric=self.hyper_param_metric,
                                      params=self.hyperparams,
                                      train_dir=self.train_dir)
            print(self.get_best_model_params(self.experiment_name))
        else:
            model = self.network.build_model()
            hyperparams = self.hyperparams

            # Optimizer definition
            optimizer = self.create_optimizer(self.optimizer, self.lr)

            # Model compilation
            model.compile(
                optimizer=optimizer,
                loss=[self.loss],
                metrics=[self.metric])

            # Training & testing the model (fit)
            callbacks = self.create_callbacks(self.train_dir, self.lr_scheduler, self.tb)

            print('Training the network')
            history = self.train(model, self.epochs, self.batch_size, callbacks,
                                self.train_x, self.valid_x, self.train_y, self.valid_y)
            # Plot metrics
            self.plot_graph(history.history, self.metric, self.metric.capitalize(), self.train_dir)
            self.plot_graph(history.history, 'loss', f'Loss: {self.loss.capitalize()}', self.train_dir)

            print('Testing the network')
            test_results = self.test(model, self.batch_size, self.test_x, self.test_y)
            
            # TODO save the test results ? (did not find that in the old code)

            model_json = model.to_json()
            with open(f'{self.train_dir}/{self.network.name}_model.json', 'w') as json_file:
                json_file.write(model_json)

    @staticmethod
    def return_last_experiment_results(exp_name):
        results = os.listdir(os.path.join(os.getcwd(), exp_name))
        results.sort()
        return results[-1]

    def get_best_model_params(self, exp_name):
        results = pd.read_csv(f'{exp_name}/{self.return_last_experiment_results(exp_name)}')
        results.sort_values(self.hyper_param_metric, ascending=False, inplace=True)
        best_model_params = results.iloc[0]
        print(results)

        return best_model_params

    @staticmethod
    def tune_hyperparameters(train_x, train_y, valid_x, valid_y, network, params, experiment_name, tune_rounds, metric, train_dir):
        t = ta.Scan(x=train_x, y=train_y, x_val=valid_x, y_val=valid_y, model=network, params=params, experiment_name=experiment_name, round_limit=tune_rounds, reduction_metric=metric, print_params=True, save_weights=True, reduction_method='correlation', reduction_threshold=0.1)
        best_model = t.best_model('acc')
        best_model_json = best_model.to_json()
        print(best_model.summary())
        with open(f'{train_dir}/training/{experiment_name}.json', 'w') as json_file:
            json_file.write(best_model_json)
        best_model.save_weights(f'{train_dir}/training/{experiment_name}.h5')

        return best_model
    
    @staticmethod
    def step_decay_schedule(initial_lr=1e-3, drop=0.5, epochs_drop=10.0):
        def schedule(epoch):
            return initial_lr * math.pow(drop, math.floor(epoch + 1) / epochs_drop)
        return LearningRateScheduler(schedule)

    @staticmethod
    def create_callbacks(out_dir, scheduler, tb):
        mcp = ModelCheckpoint(filepath=out_dir + '/model.hdf5',
                              verbose=0,
                              save_best_only=True)

        earlystopper = EarlyStopping(monitor='val_loss',
                                     patience=10,
                                     min_delta=0.1,
                                     verbose=1,
                                     mode='auto')

        csv_logger = CSVLogger(out_dir + '/log.csv',
                               append=True,
                               separator='\t')
        callbacks = [mcp, earlystopper, csv_logger]

        if scheduler:
            callbacks.append(Train.step_decay_schedule())

        if tb:
            callbacks.append(TensorBoard(log_dir=out_dir, histogram_freq=1, profile_batch=3))

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

        logger.info('Evaluation loss: ' + str(round(test_results[0],4)))
        logger.info('Evaluation acc: ' + str(round(test_results[1],4)))
    
        return test_results

    @staticmethod
    def plot_graph(history, metric, title, out_dir):
        # TODO separate class for plotting? probably combined with the Evaluate module
        # For some reason here it calls accuracy just 'acc'
        val_metric = f'val_{metric}'
        file_name = f'/{metric}.png'

        plt.plot(history[metric])
        plt.plot(history[val_metric])

        if metric == 'accuracy':
            plt.ylim(0.0, 1.0)
        elif metric == 'loss':
            plt.ylim(0.0, max(max(history[metric]), max(history[val_metric])))

        logger.info('Best achieved ' + metric + ' - ' + str(round(max(history[metric]),4)))
        logger.info('Best achieved ' + val_metric + ' - ' + str(round(max(history[val_metric]),4)))
        
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(out_dir + file_name, dpi=300)
        plt.clf()
