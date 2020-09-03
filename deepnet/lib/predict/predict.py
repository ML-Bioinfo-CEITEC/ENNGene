import datetime
import logging
import numpy as np
import os
import pandas as pd
import streamlit as st

# TODO export the env when releasing, check pandas == 1.1.1

from tensorflow import keras

from ..utils.dataset import Dataset
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('root')


class Predict(Subcommand):
    SEQ_TYPES = {'BED file': 'bed',
                 'FASTA file': 'fasta',
                 'Text input': 'text'}
    # TODO add option to use deepnet blackbox file (already mapped and not bed) - instead add separate section to test a model on the blackbox dataset
    # test module - either on already mapped blackbox dataset, or not encoded dataset, eg from different experiment to check transferbality of the results

    def __init__(self):
        self.params = {'task': 'Predict'}
        self.validation_hash = {'is_model_file': [],
                                'is_bed': [],
                                'is_fasta': [],
                                'is_multiline_text': []}
        self.model_folder = None
        missing_params = False
        missing_model = False

        st.markdown('# Make Predictions')
        st.markdown('## General Options')
        self.add_general_options(branches=False)

        st.markdown('## Model')
        model_options = {'Use a model trained by the deepnet app': 'from_app',
                         'Use a custom trained model': 'custom'}
        self.params['model_source'] = model_options[st.radio(
            'Select a source of the trained model:',
            list(model_options.keys()), index=self.get_dict_index(self.defaults['model_source'], model_options))]

        if self.params['model_source'] == 'from_app':
            self.model_folder = st.text_input('Path to the folder containing the trained model (hdf5 file)', value=self.defaults['model_file'])
            if self.model_folder and os.path.isdir(self.model_folder):
                model_files = [f for f in os.listdir(self.model_folder) if f.endswith('.hdf5') and
                             os.path.isfile(os.path.join(self.model_folder, f))]
                if len(model_files) == 0:
                    st.markdown('#### Sorry, there is no hdf5 file in given folder.')
                elif len(model_files) == 1:
                    self.params['model_file'] = os.path.join(self.model_folder, model_files[0])
                    st.markdown(f"###### Model file: {self.params['model_file']}")
                elif len(model_files) > 1:
                    st.markdown('#### Sorry, there is too many hdf5 files in the given folder. Please specify the model file below.')

                param_files = [f for f in os.listdir(self.model_folder) if f == 'parameters.yaml' and
                               os.path.isfile(os.path.join(self.model_folder, f))]
                if len(param_files) == 0:
                    missing_params = True
                    st.markdown('#### Sorry, could not find parameters.yaml file in the given folder. '
                                'Check the folder or specify the parameters below.')
                elif len(param_files) == 1:
                    training_params = {'win': None, 'winseed': None, 'no_klasses': 0, 'klasses': []}
                    param_file = os.path.join(self.model_folder, param_files[0])
                    # TODO read from yaml: window, winseed reuse, class labels, number of classes
                    # TODO ! CANT get those from trainig params, must save preprocess params together with the training,
                    #  and then again all of those to the prediction
                    # TODO zajistit konzistentni poradi klass names
                    if not training_params['win'] or not training_params['winseed'] \
                            or training_params['no_klasses'] == 0 or len(training_params['klasses']) == 0 \
                            or len(training_params['klasses']) != training_params['no_klasses']:
                        missing_params = True
                        st.markdown('#### Sorry, could not read the parameters from given folder. '
                                    'Check the folder or specify the parameters below.')
                    else:
                        st.markdown('##### Parameters read from given folder:'
                                    f"- window: {training_params['win']}"
                                    f"- window seed: {training_params['winseed']}"
                                    f"- no. of classes: {training_params['no_klasses']}"
                                    f"- class labels: {training_params['klasses']}")
                if len(param_files) > 1:
                    missing_params = True
                    st.markdown('#### Sorry, there is too many parameters.yaml files in the given folder. '
                                'Check the folder or specify the parameters below.')

        if self.params['model_source'] == 'custom' or missing_params or missing_model:
            if not missing_params:
                self.params['model_file'] = st.text_input('Path to the trained model (hdf5 file)', value=self.defaults['model_file'])
            if not missing_model:
                self.params['win'] = int(st.number_input('Window size used for training', min_value=3, value=self.defaults['win']))
                self.params['winseed'] = int(st.number_input('Seed for semi-random window placement upon the sequences',
                                                             value=self.defaults['winseed']))
                self.params['no_klasses'] = int(st.number_input('Number of classes used for training', min_value=2, value=self.defaults['no_klasses']))
                # mozna zachovat v nejakym meta souboru ktera trida je co a podle toho
                for i in range(self.params['no_klasses']):
                    if len(self.params['klasses']) >= i+1:
                        value = self.params['klasses'][i]
                    else:
                        value = str(i)
                        self.params['klasses'].append(value)
                    self.params['klasses'][i] = st.text_input(f'Class {i} label:', value=value)

        self.validation_hash['is_model_file'].append(self.params['model_file'])

        st.markdown('## Sequences')
        self.params['seq_type'] = self.SEQ_TYPES[st.radio(
            'Select a source of the sequences for the prediction:',
            list(self.SEQ_TYPES.keys()), index=self.get_dict_index(self.defaults['seq_type'], self.SEQ_TYPES))]

        self.params['alphabet'] = st.selectbox('Select alphabet:',
                                           list(seq.ALPHABETS.keys()),
                                               index=list(seq.ALPHABETS.keys()).index(self.defaults['alphabet']))

        if self.params['seq_type'] == 'bed':
            self.params['seq_source'] = st.text_input(
                'Path to BED file containing intervals to be classified', value=self.defaults['seq_source'])
            self.validation_hash['is_bed'].append(self.params['seq_source'])
            self.params['fasta_ref'] = st.text_input('Path to reference fasta file', value=self.defaults['fasta_ref'])
            self.params['strand'] = st.checkbox('Apply strandedness', self.defaults['strand'])
            self.validation_hash['is_fasta'].append(self.params['fasta_ref'])
        elif self.params['seq_type'] == 'fasta' or self.params['seq_type'] == 'text':
            st.markdown('###### WARNING: Sequences shorter than the window size will be padded with Ns (might affect '
                        'the prediction accuracy). Longer sequences will be cut to the length of the window.')
            if self.params['seq_type'] == 'fasta':
                self.params['seq_source'] = st.text_input(
                    'Path to FASTA file containing sequences to be classified', value=self.defaults['seq_source'])
                self.validation_hash['is_fasta'].append(self.params['seq_source'])
            elif self.params['seq_type'] == 'text':
                self.params['seq_source'] = st.text_area(
                    'One or more sequences to be classified (each sequence on a new line)', value=self.defaults['seq_source'])
                self.validation_hash['is_multiline_text'].append({'text': self.params['seq_source'],
                                                                  'alphabet': seq.ALPHABETS[self.params['alphabet']]})

        self.validate_and_run(self.validation_hash)

    def run(self):
        status = st.empty()
        status.text('Preparing sequences...')

        predict_dir = os.path.join(self.params['output_folder'], 'prediction',
                                 f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(predict_dir)

        prepared_file_path = os.path.join(predict_dir, 'mapped.tsv')
        if self.params['seq_type'] == 'bed':
            dataset = Dataset(bed_file=self.params['seq_source'], branches=['predict'], category='predict',
                              win=self.params['win'], winseed=self.params['winseed'])
            dataset.df['input'] = dataset.df['predict']
            status.text('Parsing reference fasta file...')
            fasta_dict, _ = seq.parse_fasta_reference(self.params['fasta_ref'])
            status.text('Mapping intervals to the fasta reference...')
            dataset.map_to_branches(
                {'predict': fasta_dict}, self.params['alphabet'], self.params['strand'], prepared_file_path)
        elif self.params['seq_type'] == 'fasta':
            dataset = Dataset(fasta_file=self.params['seq_source'], branches=['predict'], category='predict',
                              win=self.params['win'], winseed=self.params['winseed'])
            dataset.df['input'] = dataset.df['predict']
        elif self.params['seq_type'] == 'text':
            dataset = Dataset(text_input=self.params['seq_source'], branches=['predict'], category='predict',
                              win=self.params['win'], winseed=self.params['winseed'])
            dataset.df['input'] = dataset.df['predict']
        dataset.encode_predict(self.params['alphabet'])
        predict_x = np.array(dataset.df['predict'].to_list())  # TODO check effectiveness of the to_list on larger dataset

        status.text('Calculating predictions...')
        model = keras.models.load_model(self.params['model_file'])
        predict_y = model.predict(
            predict_x,
            verbose=1)

        dataset.df['predicted class'] = self.get_klass(predict_y, self.params['klasses'])
        dataset.df[f"raw predicted probabilities {self.params['klasses']}"] = [Dataset.sequence_to_string(y) for y in predict_y]

        status.text('Exporting results...')
        result_file = os.path.join(predict_dir, 'results.tsv')
        dataset.save_to_file(ignore_cols=['predict'], outfile_path=result_file)

        self.finalize_run(logger, predict_dir, self.params, self.csv_header(), self.csv_row(predict_dir, self.params, self.model_folder))
        status.text('Finished!')

    @staticmethod
    def get_klass(predicted, klasses):
        #TODO give the user choice of the tresshold value? - based on test results (specificity?
        # would have to be saved to the log file or similar) or user input if not available
        #TODO how to decide a klass in general?
            # the treshold itself is not enough - if we are interested in one klass, and the other serves just as a background,
            # then we want one-sided treslhold, if it's klass A with at least eg 95%, otherwise neg (independent of number of klasses)
        treshold = 0.98
        chosen = []
        for probs in predicted:
            max_prob = np.amax(probs)
            if max_prob > treshold:
                # not considering an option there would be two exactly same highest probabilities
                index = np.where(probs == max_prob)[0][0]
                value = klasses[index]
            else:
                value = 'UNCERTAIN'
            chosen.append(value)
        return chosen

    @staticmethod
    def default_params():
        return {
            'model_source': 'from_app',
            'win': 100,
            'no_klasses': 2,
            'klasses': [],
            'model_file': '',
            'seq_type': 'bed',
            'seq_source': '',
            'alphabet': 'DNA',
            'strand': True,
            'fasta_ref': '',
            'winseed': 42,
            'output_folder': os.path.join(os.getcwd(), 'deepnet_output')
        }

    @staticmethod
    def csv_header():
        return 'Folder\t'\
               'Model file\t' \
               'Window\t' \
               'Window seed\t' \
               'No. classes\t' \
               'Classes\t' \
               'Sequence source type\t' \
               'Sequence source\t' \
               'Alphabet\t' \
               'Strand\t' \
               'Fasta ref.\t' \
               'Input folder\n'

    @staticmethod
    def csv_row(folder, params, model_folder):
        return f"{os.path.basename(folder)}\t" \
               f"{params['model_file']}\t" \
               f"{params['win']}\t" \
               f"{params['winseed']}\t" \
               f"{params['no_klasses']}\t" \
               f"{params['klasses']}\t" \
               f"{Predict.get_dict_key(params['seq_type'], Predict.SEQ_TYPES)}\t" \
               f"{params['seq_source']}\t" \
               f"{params['alphabet']}\t" \
               f"{params['strand']}\t" \
               f"{params['fasta_ref']}\t" \
               f"{model_folder}\n"
