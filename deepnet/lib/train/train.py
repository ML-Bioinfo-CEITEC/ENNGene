import altair as alt
import datetime
import copy
import os
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yaml

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

from .callbacks import ProgressMonitor, LRFinder, OneCycleLR
from .layers import BRANCH_LAYERS, COMMON_LAYERS
from .model_builder import ModelBuilder
from ..utils.dataset import Dataset
from ..utils.exceptions import UserInputError
from ..utils import file_utils as f
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('root')


class Train(Subcommand):
    def __init__(self):
        self.params = {'task': 'Train'}
        self.validation_hash = {'not_empty_branches': [],
                                'is_dataset_dir': []}

        st.markdown('# Training')
        st.markdown('')

        self.general_options()

        self.params['input_folder'] = st.text_input(
            'Datasets folder', value=self.defaults['input_folder'])
        self.validation_hash['is_dataset_dir'].append(self.params['input_folder'])

        if self.params['input_folder']:
            note = ''
            previous_param_files = [f for f in os.listdir(self.params['input_folder']) if (f == 'parameters.yaml') and
                                    (os.path.isfile(os.path.join(self.params['input_folder'], f)))]
            if len(previous_param_files) == 1:
                self.previous_param_file = os.path.join(self.params['input_folder'], previous_param_files[0])
                with open(self.previous_param_file, 'r') as file:
                    user_params = yaml.safe_load(file)
                    preprocess_branches = user_params['Preprocess']['branches']
            else:
                # We could allow to continue without it, but the user would have to select correct branches
                raise UserInputError('Did not find parameters.yaml file in the given dataset folder.')
        else:
            preprocess_branches = []
            note = ' (You must first provide the dataset folder (containing also the parameters.yaml file))'

        available_branches = [self.get_dict_key(b, self.BRANCHES) for b in preprocess_branches]
        default_branches = [self.get_dict_key(b, self.BRANCHES) for b in self.defaults['branches']]
        self.params['branches'] = list(map(lambda name: self.BRANCHES[name],
                                           st.multiselect('Branches'+note,
                                                          available_branches,
                                                          default=default_branches)))
        self.validation_hash['not_empty_branches'].append(self.params['branches'])

        self.params['tb'] = st.checkbox('Output TensorBoard log files', value=self.defaults['tb'])

        st.markdown('## Training Options')
        # TODO make sure batch size is smaller than dataset size
        self.params['batch_size'] = st.number_input('Batch size', min_value=1, value=self.defaults['batch_size'])
        self.params['epochs'] = st.number_input('No. of training epochs', min_value=1, value=self.defaults['epochs'])
        self.params['early_stop'] = st.checkbox('Apply early stopping (patience 10, delta 0.1)', value=self.defaults['early_stop'])
        self.params['optimizer'] = self.OPTIMIZERS[st.selectbox(
            'Optimizer', list(self.OPTIMIZERS.keys()), index=self.get_dict_index(self.defaults['optimizer'], self.OPTIMIZERS))]
        if self.params['optimizer'] == 'sgd':
            # TODO move lr finder to hyperparam tuning, runs one epoch on small sample, does not need valid and test data
            lr_options = {'Use fixed learning rate (applies learning rate value throughout whole training)': 'fixed',
                          'Use learning rate scheduler (gradually decreasing from the learning rate value)': 'lr_scheduler',
                          # 'Use learning rate finder (beta)': 'lr_finder',
                          'Apply one cycle policy (uses the learning rate value as max)': 'one_cycle'}
            self.params['lr_optim'] = lr_options[st.radio('Learning rate options',
                                                          list(lr_options.keys()),
                                                          index=self.get_dict_index(self.defaults['lr_optim'], lr_options))]
        self.params['lr'] = st.number_input(
            'Learning rate', min_value=0.0001, max_value=0.1, value=self.defaults['lr'], step=0.0001, format='%.4f')
        self.params['metric'] = self.METRICS[st.selectbox('Metric',
                                                          list(self.METRICS.keys()),
                                                          self.get_dict_index(self.defaults['metric'], self.METRICS))]
        self.params['loss'] = self.LOSSES[st.selectbox('Loss function',
                                                       list(self.LOSSES.keys()),
                                                       self.get_dict_index(self.defaults['loss'], self.LOSSES))]

        # TODO change the logic when stateful design is enabled
        # self.branch_layers = {}
        # for branch in self.branches:
        #     self.branch_layers.update({branch: [list(LAYERS.keys())[0]]})
        #
        # for branch in self.branch_layers.keys():
        #     st.markdown(f'**{branch} branch**')
        #     for i, layer in enumerate(self.branch_layers[branch]):
        #         new_layer = st.selectbox('Layer', list(LAYERS.keys()), key=f'layer{branch}{i}')
        #         self.branch_layers[branch][i] = new_layer
        #     if st.button('Add layer', key=f'{branch}addlayer'):
        #         self.branch_layers[branch].append(
        #             st.selectbox('Layer', list(LAYERS.keys()), key=f'layer{branch}{len(self.branch_layers[branch])}'))
        #         print(self.branch_layers[branch])

        default_args = {'batchnorm': False, 'dropout': 0.0, 'filters': 40, 'kernel': 4, 'units': 32}

        st.markdown('## Network Architecture')
        self.params['branches_layers'] = self.defaults['branches_layers']
        for i, branch in enumerate(self.params['branches']):
            st.markdown(f'### {i+1}. {self.get_dict_key(branch, self.BRANCHES)} branch')
            self.params['no_branches_layers'][branch] = st.number_input(
                'Number of layers in the branch:', min_value=0, value=self.defaults['no_branches_layers'][branch], key=f'{branch}_no')
            for i in range(self.params['no_branches_layers'][branch]):
                if self.params_loaded and i < len(self.defaults['branches_layers'][branch]):
                    default_args = self.defaults['branches_layers'][branch][i]['args']
                    layer = copy.deepcopy(self.defaults['branches_layers'][branch][i])
                    checkbox = True
                else:
                    layer = {'name': 'Convolution layer', 'args': {'batchnorm': False, 'dropout': 0.0, 'filters': 40, 'kernel': 4}}
                    checkbox = False
                st.markdown(f'#### Layer {i + 1}')
                default_i = list(BRANCH_LAYERS.keys()).index(layer['name'])
                layer.update(dict(name=st.selectbox('Layer type', list(BRANCH_LAYERS.keys()), index=default_i, key=f'layer{branch}{i}')))
                layer = self.layer_options(layer, i, checkbox, default_args, branch)
                st.markdown('---')
                if len(self.params['branches_layers'][branch]) > i:
                    self.params['branches_layers'][branch][i] = layer
                else:
                    self.params['branches_layers'][branch].append(layer)

        st.markdown(f"### {len(self.params['branches'])+1}. Connected (after branches' concatenation)")
        self.params['common_layers'] = self.defaults['common_layers']
        self.params['no_common_layers'] = st.number_input(
            'Number of layers after concatenation of branches:', min_value=0, value=self.defaults['no_common_layers'], key=f'common_no')
        for i in range(self.params['no_common_layers']):
            if self.params_loaded:
                default_args = self.defaults['common_layers'][i]['args']
                layer = copy.deepcopy(self.defaults['common_layers'][i])
                checkbox = True
            else:
                layer = {'name': 'Dense layer', 'args': {'batchnorm': False, 'dropout': 0.0, 'units': 32}}
                checkbox = False
            default_i = list(COMMON_LAYERS.keys()).index(layer['name'])
            st.markdown(f'#### Layer {i + 1}')
            layer.update(dict(name=st.selectbox('Layer type', list(COMMON_LAYERS.keys()), index=default_i, key=f'common_layer{i}')))
            layer = self.layer_options(layer, i, checkbox, default_args)
            st.markdown('---')
            if len(self.params['common_layers']) > i:
                self.params['common_layers'][i] = layer
            else:
                self.params['common_layers'].append(layer)

        self.validate_and_run(self.validation_hash)

    # TODO adjust when stateful ops enabled
    def layer_options(self, layer, i, checkbox, defaults=None, branch=None):
        if st.checkbox('Show advanced options', value=checkbox, key=f'show{branch}{i}'):
            layer['args'].update({'batchnorm': st.checkbox(
                'Batch normalization', value=defaults['batchnorm'], key=f'batch{branch}{i}')})
            layer['args'].update({'dropout': st.slider(
                'Dropout rate', min_value=0.0, max_value=1.0, value=defaults['dropout'], key=f'do{branch}{i}', format='%.2f')})
            if layer['name'] in ['Convolution layer', 'Locally Connected 1D layer']:
                layer['args'].update({'filters': st.number_input('Number of filters:', min_value=1, value=
                defaults['filters'], key=f'filters{branch}{i}')})
                layer['args'].update({'kernel': st.number_input('Kernel size:', min_value=1, value=
                defaults['kernel'], key=f'kernel{branch}{i}')})
            elif layer['name'] in ['Dense layer', 'RNN', 'GRU', 'LSTM']:
                layer['args'].update({'units': st.number_input('Number of units:', min_value=1, value=
                defaults['units'], key=f'units{branch}{i}')})
        return layer

    @staticmethod
    def parse_data(dataset_files, branches, alphabet):
        dictionary = {}
        for file in dataset_files:
            dataset = Dataset.load_from_file(file)
            dictionary.update({dataset.category: {}})
            values = []
            for branch in branches:
                # can not use apply, as it returns wrong object with different shape
                # (because pandas dataframes and series are not able to store arrays as values)
                value = []
                for string in dataset.df[branch]:
                    value.append(dataset.sequence_from_string(string))
                values.append(np.array(value))

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
        status = st.empty()
        status.text('Initializing network...')

        candidate_files = f.list_files_in_dir(self.params['input_folder'], 'zip')
        categories = ['train', 'validation', 'test', 'blackbox']
        dataset_files = [file for file in candidate_files if any(category in os.path.basename(file) for category in categories)]

        if self.previous_param_file:
            with open(self.previous_param_file, 'r') as file:
                previous_params = yaml.safe_load(file)
                encoded_labels = seq.onehot_encode_alphabet(previous_params['Preprocess']['klasses'])
        else:
            raise UserInputError('Could not read class labels from parameters.yaml file).')
        train_x, valid_x, test_x, train_y, valid_y, test_y = self.parse_data(dataset_files, self.params['branches'], encoded_labels)
        branch_shapes = self.get_shapes(train_x, self.params['branches'])

        self.params['train_dir'] = os.path.join(self.params['output_folder'], 'training',
                                 f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(self.params['train_dir'])

        model = ModelBuilder(self.params['branches'], encoded_labels, branch_shapes, self.params['branches_layers'], self.params['common_layers']).build_model()
        optimizer = self.create_optimizer(self.params['optimizer'], self.params['lr'])
        model.compile(
            optimizer=optimizer,
            loss=[self.params['loss']],
            metrics=[self.params['metric']])

        # Training & testing the model
        status.text('Training the network...')

        progress_bar = st.progress(0)
        progress_status = st.empty()
        chart = st.altair_chart(self.initialize_altair_chart(), use_container_width=True)

        callbacks = self.create_callbacks(
            self.params['train_dir'], self.params['lr_optim'], self.params['tb'], self.params['epochs'], progress_bar, progress_status, chart, self.params['early_stop'],
            self.params['lr'], branch_shapes[self.params['branches'][0]][0])

        history = self.train(model, self.params['epochs'], self.params['batch_size'], callbacks, train_x, valid_x, train_y, valid_y).history
        # if self.params['lr_optim'] == 'lr_finder': self.params['epochs'] = 1
        # if self.params['lr_optim'] == 'lr_finder': LRFinder.plot_schedule_from_file(self.params['train_dir'])

        if self.params['early_stop']:
            early_epochs = [callback for callback in callbacks if type(callback) == EarlyStopping][0]
            if early_epochs and early_epochs.stopped_epoch != 0:
                self.params['epochs'] = early_epochs.stopped_epoch

        self.params['best_acc'] = str(round(max(history[self.params['metric']]), 4))
        self.params['best_loss'] = str(round(min(history['loss']), 4))
        self.params['best_val_acc'] = str(round(max(history[f"val_{self.params['metric']}"]), 4))
        self.params['best_val_loss'] = str(round(min(history[f"val_loss"]), 4))

        logger.info('Best achieved training ' + self.params['metric'] + ': ' + self.params['best_acc'])
        logger.info('Best achieved training loss: ' + self.params['best_loss'])
        logger.info('Best achieved ' + f"validation {self.params['metric']}" + ': ' + self.params['best_val_acc'])
        logger.info('Best achieved validation loss: ' + self.params['best_val_loss'])

        st.text(f"Best achieved training {self.params['metric']}: {self.params['best_acc']}\n"
                f"Best achieved training loss: {self.params['best_loss']}\n\n"
                f"Best achieved validation {self.params['metric']}: " + self.params['best_val_acc'] + '\n'
                'Best achieved validation loss: ' + self.params['best_val_loss'] + '\n\n')

        # Plot metrics
        # if self.params['lr_optim'] != 'lr_finder':
        self.plot_graph(history, self.params['metric'], self.params['metric'].capitalize(), self.params['train_dir'])
        self.plot_graph(history, 'loss', f"Loss: {self.params['loss'].capitalize()}", self.params['train_dir'])
        tf.keras.utils.plot_model(model, to_file=f"{self.params['train_dir']}/model.png", show_shapes=True, dpi=300)

        status.text('Testing the network...')
        test_results = self.test(model, self.params['batch_size'], test_x, test_y)

        self.params['eval_loss'] = str(round(test_results[0], 4))
        self.params['eval_acc'] = str(round(test_results[1], 4))
        logger.info('Evaluation loss: ' + self.params['eval_loss'])
        logger.info('Evaluation acc: ' + self.params['eval_acc'])
        st.text(f"Evaluation loss: {self.params['eval_loss']} \nEvaluation acc: {self.params['eval_acc']} \n")

        model_json = model.to_json()
        with open(f"{self.params['train_dir']}/model.json", 'w') as json_file:
            json_file.write(model_json)

        header = self.train_header(self.params['metric'])
        row = self.train_row(self.params)
        if 'Preprocess' in previous_params.keys():
            header += f'{self.preprocess_header()}\n'
            row += f"{self.preprocess_row(previous_params['Preprocess'])}\n"
        else:
            header += '\n'
            row += '\n'
        self.finalize_run(logger, self.params['train_dir'], self.params, header, row, self.previous_param_file)
        status.text('Finished!')
    
    @staticmethod
    def step_decay_schedule(initial_lr, drop=0.5, epochs_drop=10.0):
        def schedule(epoch):
            return initial_lr * math.pow(drop, math.floor(epoch + 1) / epochs_drop)
        return LearningRateScheduler(schedule)

    @staticmethod
    def create_callbacks(out_dir, lr_optim, tb, epochs, progress_bar, progress_status, chart, early_stop, lr, sample):
        mcp = ModelCheckpoint(filepath=out_dir + '/model.hdf5',
                              verbose=0,
                              save_best_only=True)

        csv_logger = CSVLogger(out_dir + '/log.csv',
                               append=True,
                               separator='\t')

        progress = ProgressMonitor(epochs, progress_bar, progress_status, chart)

        callbacks = [mcp, csv_logger, progress]

        if early_stop:
            earlystopper = EarlyStopping(monitor='val_loss',
                                         patience=10,
                                         min_delta=0.1,
                                         verbose=1,
                                         mode='auto')
            callbacks.append(earlystopper)

        if lr_optim != 'fixed':
            # if lr_optim == 'lr_finder':
            #     callbacks.append(LRFinder(num_samples=sample,
            #                               batch_size=sample//30,
            #                               minimum_lr=1e-5,
            #                               maximum_lr=1e0,
            #                               lr_scale='exp',
            #                               save_dir=out_dir))
            if lr_optim == 'one_cycle':
                callbacks.append(OneCycleLR(max_lr=lr,
                                            end_percentage=0.1,
                                            scale_percentage=None,
                                            maximum_momentum=0.95,
                                            minimum_momentum=0.85,
                                            verbose=True))
            elif lr_optim == 'lr_scheduler':
                callbacks.append(Train.step_decay_schedule(initial_lr=lr))

        if tb:
            callbacks.append(TensorBoard(log_dir=out_dir, histogram_freq=1, profile_batch=3))

        return callbacks

    @staticmethod
    def create_optimizer(chosen, learning_rate):
        if chosen == 'sgd':
            optimizer = SGD(
                lr=learning_rate,
                momentum=0.9,
                nesterov=True)
        elif chosen == 'rmsprop':
            optimizer = RMSprop(
                lr=learning_rate
            )
        elif chosen == 'adam':
            optimizer = Adam(
                lr=learning_rate,
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
        val_metric = f'val_{metric}'
        file_name = f'/{metric}.png'

        plt.plot(history[metric])
        plt.plot(history[val_metric])

        if metric == 'accuracy':
            plt.ylim(0.0, 1.0)
        elif metric == 'loss':
            plt.ylim(0.0, max(max(history[metric]), max(history[val_metric])))

        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(out_dir + file_name, dpi=300)
        plt.clf()

    @staticmethod
    def initialize_altair_chart():
        source = pd.DataFrame([], columns=['Metric', 'Metric value', 'Epoch'])
        nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Epoch'], empty='none')
        line = alt.Chart(source).mark_line().encode(x='Epoch:Q', y='Metric value:Q', color='Metric:N')
        selectors = alt.Chart(source).mark_point().encode(x='Epoch:Q', opacity=alt.value(0)).add_selection(nearest)
        points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Metric value:Q', alt.value(' ')))
        rules = alt.Chart(source).mark_rule(color='gray').encode(x='Epoch:Q').transform_filter(nearest)
        alt_chart = alt.layer(line, selectors, points, rules, text)
        return alt_chart

    @staticmethod
    def default_params():
        return {'batch_size': 256,
                'branches': [],
                'branches_layers': {'seq': [], 'fold': [], 'cons': []},
                'common_layers': [],
                'early_stop': True,
                'epochs': 100,
                'input_folder': '',
                'loss': 'categorical_crossentropy',
                'lr': 0.005,
                'lr_optim': 'fixed',
                'metric': 'accuracy',
                'no_branches_layers': {'seq': 1, 'fold': 1, 'cons': 1},
                'no_common_layers': 1,
                'optimizer': 'sgd',
                'output_folder': os.path.join(os.getcwd(), 'deepnet_output'),
                'tb': True}
