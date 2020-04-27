import altair as alt
import datetime
import os
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

from .callbacks import ProgressMonitor, LRFinder, OneCycleLR
from .layers import LAYERS
from .model_builder import ModelBuilder
from ..utils.dataset import Dataset
from ..utils import file_utils as f
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('root')


class Train(Subcommand):
    OPTIMIZERS = {'SGD': 'sgd',
                  'RMSprop': 'rmsprop',
                  'Adam': 'adam'}
    METRICS = {'Accuracy': 'accuracy'}
    LOSSES = {'Categorical Crossentropy': 'categorical_crossentropy'}

    def __init__(self):
        self.validation_hash = {'not_empty_branches': [],
                                'is_dataset_dir': []}

        st.markdown('# Train a Model')

        st.markdown('## General Options')
        self.add_general_options()

        self.input_folder = st.text_input('Path to folder containing all preprocessed files (train, validation, test)')
        self.validation_hash['is_dataset_dir'].append(self.input_folder)

        self.tb = st.checkbox('Output TensorBoard log files', value=False)

        st.markdown('## Training Options')
        # TODO make sure batch size is smaller than dataset size
        self.batch_size = st.number_input('Batch size', min_value=0, value=256)
        self.epochs = st.slider('No. of training epochs', min_value=0, max_value=1000, value=600)
        self.early_stop = st.checkbox('Apply early stopping', value=True)
        self.optimizer = self.OPTIMIZERS[st.selectbox('Optimizer', list(self.OPTIMIZERS.keys()))]
        self.lr = st.number_input('Learning rate', min_value=0.0, max_value=0.1, value=0.0001, step=0.0001, format='%.4f')
        if self.optimizer == 'sgd':
            # TODO move lr finder to hyperparam tuning, runs one epoch on small sample, does not need valid and test data
            lr_options = {'Use fixed learning rate (applies above defined value throughout whole training)': None,
                          'Use learning rate scheduler (gradually decreasing lr from 0.1)': 'lr_scheduler',
                          'Use learning rate finder (beta)': 'lr_finder',
                          'Apply one cycle policy on learning rate (beta; uses above defined value as max)': 'one_cycle'}
            self.lr_optim = lr_options[st.radio('Learning rate options',
                list(lr_options.keys()))]
        else:
            self.lr_optim = None
        self.metric = self.METRICS[st.selectbox('Metric', list(self.METRICS.keys()))]
        self.loss = self.LOSSES[st.selectbox('Loss function', list(self.LOSSES.keys()))]

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

        st.markdown('## Network Architecture')
        self.branches_layers = {}
        for i, branch in enumerate(self.branches):
            st.markdown(f'### {i+1}. {list(self.BRANCHES.keys())[list(self.BRANCHES.values()).index(branch)]} branch')
            self.branches_layers.update({branch: []})
            no = st.number_input('Number of layers in the branch:', min_value=0, value=1, key=f'{branch}_no')
            for i in range(no):
                layer = dict(args={})
                st.markdown(f'#### Layer {i + 1}')
                layer.update(dict(name=st.selectbox('Layer type', list(LAYERS.keys()), key=f'layer{branch}{i}')))
                layer = self.layer_options(layer, i, branch)
                st.markdown('---')
                if len(self.branches_layers[branch]) > i:
                    self.branches_layers[branch][i] = layer
                else:
                    self.branches_layers[branch].append(layer)

        st.markdown(f'### {len(self.branches)+1}. Connected (after branches\' concatenation)')
        self.common_layers = []
        allowed_common = list(LAYERS.keys())
        allowed_common.remove('Convolutional layer')
        no = st.number_input('Number of layers after concatenation of branches:', min_value=0, value=1, key=f'common_no')
        for i in range(no):
            layer = {'args': {}}
            st.markdown(f'#### Layer {i + 1}')
            layer.update(dict(name=st.selectbox('Layer type', allowed_common, key=f'common_layer{i}')))
            layer = self.layer_options(layer, i)
            st.markdown('---')
            if len(self.common_layers) > i:
                self.common_layers[i] = layer
            else:
                self.common_layers.append(layer)

        self.validate_and_run(self.validation_hash)

    @staticmethod
    # TODO adjust when stateful ops enabled
    def layer_options(layer, i, branch=None):
        if st.checkbox('Show advanced options', value=False, key=f'show{branch}{i}'):
            layer['args'].update({'batchnorm': st.checkbox('Batch normalization', value=False, key=f'batch{branch}{i}')})
            layer['args'].update({'dropout': st.slider('Dropout rate', min_value=0.0, max_value=1.0, value=0.0, key=f'do{branch}{i}')})
            if layer['name'] == 'Convolutional layer':
                layer['args'].update({'filters': st.number_input('Number of filters:', min_value=0, value=40, key=f'filters{branch}{i}')})
                layer['args'].update({'kernel': st.number_input('Kernel size:', min_value=0, value=4, key=f'kernel{branch}{i}')})
            elif layer['name'] == 'Dense layer':
                layer['args'].update(
                    {'units': st.number_input('Number of units:', min_value=0, value=32, key=f'units{branch}{i}')})
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

        candidate_files = f.list_files_in_dir(self.input_folder, 'zip')
        cateogires = ['train', 'validation', 'test']  # TODO add blackbox when in use
        self.dataset_files = [file for file in candidate_files if any(category in file for category in cateogires)]

        labels = seq.onehot_encode_alphabet(list(set(Dataset.load_from_file(self.dataset_files[0]).labels())))
        train_x, valid_x, test_x, train_y, valid_y, test_y = self.parse_data(self.dataset_files, self.branches, labels)
        branch_shapes = self.get_shapes(train_x, self.branches)

        train_dir = os.path.join(self.output_folder, 'training',
                                 f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(train_dir)

        model = ModelBuilder(self.branches, labels, branch_shapes, self.branches_layers, self.common_layers).build_model()
        optimizer = self.create_optimizer(self.optimizer, self.lr)
        model.compile(
            optimizer=optimizer,
            loss=[self.loss],
            metrics=[self.metric])

        # Training & testing the model
        status.text('Training the network...')

        progress_bar = st.progress(0)
        progress_status = st.empty()
        chart = st.altair_chart(self.initialize_altair_chart())

        callbacks = self.create_callbacks(
            train_dir, self.lr_optim, self.tb, self.epochs, progress_bar, progress_status, chart, self.early_stop,
            self.lr, branch_shapes[self.branches[0]][0])

        if self.lr_optim == 'lr_finder': self.epochs = 1
        history = self.train(model, self.epochs, self.batch_size, callbacks, train_x, valid_x, train_y, valid_y).history
        if self.lr_optim == 'lr_finder': LRFinder.plot_schedule_from_file(train_dir)

        logger.info('Best achieved ' + self.metric + ' - ' + str(round(max(history[self.metric]), 4)))
        logger.info('Best achieved ' + f'val_{self.metric}' + ' - ' + str(round(max(history[f'val_{self.metric}']), 4)))
        st.text('Best achieved ' + self.metric + ' - ' + str(round(max(history[self.metric]), 4)))
        st.text('Best achieved ' + f'val_{self.metric}' + ' - ' + str(round(max(history[f'val_{self.metric}']), 4)))

        # Plot metrics
        self.plot_graph(history, self.metric, self.metric.capitalize(), train_dir)
        self.plot_graph(history, 'loss', f'Loss: {self.loss.capitalize()}', train_dir)
        tf.keras.utils.plot_model(model, to_file=f'{train_dir}/model.png', show_shapes=True, dpi=300)

        status.text('Testing the network...')
        test_results = self.test(model, self.batch_size, test_x, test_y)

        logger.info('Evaluation loss: ' + str(round(test_results[0], 4)))
        logger.info('Evaluation acc: ' + str(round(test_results[1], 4)))
        st.text('Evaluation loss: ' + str(round(test_results[0], 4)))
        st.text('Evaluation acc: ' + str(round(test_results[1], 4)))

        model_json = model.to_json()
        with open(f'{train_dir}/model.json', 'w') as json_file:
            json_file.write(model_json)

        self.finalize_run(logger, train_dir)
        status.text('Finished!')
    
    @staticmethod
    def step_decay_schedule(initial_lr=1e-3, drop=0.5, epochs_drop=10.0):
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
                                         patience=50,
                                         min_delta=0.01,
                                         verbose=1,
                                         mode='auto')
            callbacks.append(earlystopper)

        if lr_optim:
            if lr_optim == 'lr_finder':
                callbacks.append(LRFinder(num_samples=sample,
                                          batch_size=sample//30,
                                          minimum_lr=1e-5,
                                          maximum_lr=1e0,
                                          lr_scale='exp',
                                          save_dir=out_dir))
            elif lr_optim == 'one_cycle':
                callbacks.append(OneCycleLR(max_lr=lr,
                                            end_percentage=0.1,
                                            scale_percentage=None,
                                            maximum_momentum=0.95,
                                            minimum_momentum=0.85,
                                            verbose=True))
            elif lr_optim == 'lr_scheduler':
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
