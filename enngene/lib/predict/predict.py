import datetime
import logging
import numpy as np
import os
import pandas as pd
import streamlit as st
import yaml
import csv
import tensorflow as tf


# TODO export the env when releasing, check pandas == 1.1.1

from tensorflow import keras

from . import ig
from ..utils.dataset import Dataset
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('root')


class Predict(Subcommand):
    # TODO add option to use blackbox file (already mapped and not bed) - instead add separate section to test a model on the blackbox dataset
    # test module - either on already mapped blackbox dataset, or not encoded dataset, eg from different experiment to check transferbality of the results

    def __init__(self):
        self.params = {'task': 'Predict'}
        self.validation_hash = {'is_model_file': [],
                                'is_bed': [],
                                'is_fasta': [],
                                'is_multiline_text': [],
                                'is_wig_dir': []}
        self.params['model_folder'] = None

        st.markdown('# Prediction')
        st.markdown('')
        self.general_options()

        st.markdown('## Model')
        self.model_options()

        st.markdown('## Sequences')
        if 'cons' in self.params['branches']:
            # to map to the conservation files we need the coordinates
            self.params['seq_type'] = 'bed'
            st.markdown('###### Note: Only BED files allowed when Conservation score branch is applied (the coordinates are necessary).')
        else:
            self.params['seq_type'] = self.SEQ_TYPES[st.radio(
                'Select a source of the sequences for the prediction:',
                list(self.SEQ_TYPES.keys()), index=self.get_dict_index(self.defaults['seq_type'], self.SEQ_TYPES))]

        self.references = {}
        if self.params['seq_type'] == 'bed':
            self.params['seq_source'] = st.text_input(
                'Path to the BED file containing intervals to be classified', value=self.defaults['seq_source'])
            self.validation_hash['is_bed'].append(self.params['seq_source'])
            self.params['strand'] = st.checkbox('Apply strand', self.defaults['strand'])

            if 'seq' in self.params['branches'] or 'fold' in self.params['branches']:
                self.params['fasta_ref'] = st.text_input('Path to the reference fasta file', value=self.defaults['fasta_ref'])
                self.references.update({'seq': self.params['fasta_ref'], 'fold': self.params['fasta_ref']})
                self.validation_hash['is_fasta'].append(self.params['fasta_ref'])
            if 'cons' in self.params['branches']:
                self.params['cons_dir'] = st.text_input('Path to folder containing reference conservation files',
                                                        value=self.defaults['cons_dir'])
                self.references.update({'cons': self.params['cons_dir']})
                self.validation_hash['is_wig_dir'].append(self.params['cons_dir'])
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
                self.validation_hash['is_multiline_text'].append(self.params['seq_source'])
        if 'fold' in self.params['branches']:
            # currently used only as an option for RNAfold
            max_cpu = os.cpu_count() or 1
            self.ncpu = st.slider('Number of CPUs to be used for folding (max = all available CPUs on the machine).',
                                  min_value=1, max_value=max_cpu, value=max_cpu)
        else:
            self.ncpu = 1

        self.validate_and_run(self.validation_hash)

    def run(self):
        status = st.empty()
        status.text('Preparing sequences...')

        self.params['predict_dir'] = os.path.join(self.params['output_folder'], 'prediction',
                                 f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(self.params['predict_dir'])

        prepared_file_path = os.path.join(self.params['predict_dir'], 'mapped.tsv')
        predict_x = []
        if self.params['seq_type'] == 'bed':
            dataset = Dataset(bed_file=self.params['seq_source'], branches=self.params['branches'], category='predict',
                              win=self.params['win'], winseed=self.params['winseed'])
            status.text(f"Mapping intervals to {len(self.params['branches'])} branch(es) and exporting...")

        elif self.params['seq_type'] == 'fasta' or self.params['seq_type'] == 'text':
            if self.params['seq_type'] == 'fasta':
                dataset = Dataset(fasta_file=self.params['seq_source'], branches=self.params['branches'], category='predict',
                                  win=self.params['win'], winseed=self.params['winseed'])
            elif self.params['seq_type'] == 'text':
                dataset = Dataset(text_input=self.params['seq_source'], branches=self.params['branches'], category='predict',
                                  win=self.params['win'], winseed=self.params['winseed'])

        dataset.sort_datapoints().map_to_branches(
            self.references, self.params['alphabet'], self.params['strand'], prepared_file_path, status, self.ncpu)
        for branch in self.params['branches']:
            branch_list = dataset.df[branch].to_list()
            predict_x.append(np.array([Dataset.sequence_from_string(seq_str) for seq_str in branch_list]))
            # TODO check effectiveness of the to_list on larger dataset

        status.text('Calculating predictions...')

        model = keras.models.load_model(self.params['model_file'])
        predict_y = model.predict(
            predict_x,
            verbose=1)

        status.text('Calculating Integrated Gradients...')
        for i, klass in enumerate(self.params['klasses']):
            dataset.df[klass] = [y[i] for y in predict_y]
        dataset.df['highest scoring class'] = self.get_klass(predict_y, self.params['klasses'])

        if len(self.params['branches']) == 1 and self.params['branches'][0] == 'seq':

            status.text('Calculating Integrated Gradients...')

            raw_sequence = dataset.df['input_seq']  # sekvence bez one hot enoded
            print('RAW SEQUENCE', raw_sequence)
            # predict_x  # toto by mela byt data primo ve formatu pro prediction, a tedy snad i pro tvuj ucel

            # slozka, do ktere se exportuji vysledne soubory z tohoto behu, pokud by vysledkem tveho kodu bylo vic souboru, idealne pro ne vytvor nejakou podslozku
            # self.params['predict_dir']


            # set baseline, win parameter in yaml and num 5, num of sequence
            baseline = tf.zeros(shape=(self.params['win'], 5))



            # file to write html from IG predictions
            with open(self.params['predict_dir'] + "/HTML_visualisation.csv",
                      'w', encoding='utf-8') as tabular_output:

                tabular_writer = csv.writer(tabular_output, delimiter=',')

                # need to transform to right shape:
                predict_x_np = np.array(predict_x[0])

                # take each prediction, unprocessed data and count IG
                for sample, letter_sequence in zip(predict_x_np, raw_sequence):

                    # return tensor of shape: (window width(sequence length), encoded base shape)
                    sample = tf.convert_to_tensor(sample, dtype=tf.float32)

                    # contain significance of each base in sequence
                    ig_attribution = ig.integrated_gradients(model, baseline, sample)

                    # choose attribution for specific encoded base
                    attrs = ig.choose_validation_points(ig_attribution, self.params['win'], 5)

                    # return HTML code with colored bases
                    visualisation = ig.visualize_token_attrs(letter_sequence, attrs)

                    # write each HTML code to csv file
                    tabular_writer.writerow([visualisation])



            dataset.df[f"predicted probabilities ({', '.join(self.params['klasses'])})"] = [Dataset.sequence_to_string(y) for y in predict_y]
            dataset.df['highest scoring class'] = self.get_klass(predict_y, self.params['klasses'])




        status.text('Exporting results...')
        result_file = os.path.join(self.params['predict_dir'], 'results.tsv')
        ignore = ['name', 'score'] + self.params['branches']
        dataset.save_to_file(ignore_cols=ignore, outfile_path=result_file)  #TODO ignore branches cols ??

        header = self.predict_header()
        row = self.predict_row(self.params)

        if self.previous_param_file:
            with open(self.previous_param_file, 'r') as file:
                previous_params = yaml.safe_load(file)
            if 'Train' in previous_params.keys():
                header += f"{self.train_header()}"
                row += f"{self.train_row(previous_params['Train'])}"
                if 'Preprocess' in previous_params.keys():
                    header += f'{self.preprocess_header()}\n'
                    row += f"{self.preprocess_row(previous_params['Preprocess'])}\n"
                else:
                    header += '\n'
                    row += '\n'
            else:
                header += '\n'
                row += '\n'

        self.finalize_run(logger, self.params['predict_dir'], self.params, header, row, self.previous_param_file)
        status.text('Finished!')

    @staticmethod
    def get_klass(predicted, klasses):
        #TODO give the user choice of the tresshold value? - would have to specify per each class, if the highest scoring class would be above its threshold, then we would call it, otherwise uncertain

        # treshold = 0.98
        # chosen = []
        # for probs in predicted:
        #     max_prob = np.amax(probs)
        #     if max_prob > treshold:
        #         not considering an option there would be two exactly same highest probabilities
                # index = np.where(probs == max_prob)[0][0]
                # value = klasses[index]
            # else:
            #     value = 'UNCERTAIN'
            # chosen.append(value)

        chosen = [klasses[np.argmax(probs)] for probs in predicted]

        return chosen

    @staticmethod
    def default_params():
        return {
            'model_source': 'from_app',
            'model_folder': '',
            'model_file': '',
            'branches': [],
            'win': 100,
            'no_klasses': 2,
            'klasses': [],
            'seq_type': 'bed',
            'seq_source': '',
            'alphabet': 'DNA',
            'strand': True,
            'fasta_ref': '',
            'cons_dir': '',
            'winseed': 42,
            'output_folder': os.path.join(os.path.expanduser('~'), 'enngene_output')
        }

