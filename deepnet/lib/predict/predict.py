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
                                'is_multiline_text': [],
                                'is_wig_dir': []}
        self.model_folder = None

        st.markdown('# Make Predictions')
        st.markdown('## General Options')
        self.general_options()

        st.markdown('## Model')
        self.model_options()

        st.markdown('## Sequences')
        if 'cons' in self.params['branches']:
            # to map to the conservation files we need the coordinates
            self.params['seq_type'] = 'bed'
        else:
            self.params['seq_type'] = self.SEQ_TYPES[st.radio(
                'Select a source of the sequences for the prediction:',
                list(self.SEQ_TYPES.keys()), index=self.get_dict_index(self.defaults['seq_type'], self.SEQ_TYPES))]

        self.params['alphabet'] = st.selectbox('Select alphabet:',
                                           list(seq.ALPHABETS.keys()),
                                               index=list(seq.ALPHABETS.keys()).index(self.defaults['alphabet']))

        self.references = {}
        if self.params['seq_type'] == 'bed':
            self.params['seq_source'] = st.text_input(
                'Path to BED file containing intervals to be classified', value=self.defaults['seq_source'])
            self.validation_hash['is_bed'].append(self.params['seq_source'])
            self.params['strand'] = st.checkbox('Apply strandedness', self.defaults['strand'])

            if 'seq' in self.params['branches'] or 'fold' in self.params['branches']:
                self.params['fasta_ref'] = st.text_input('Path to reference fasta file', value=self.defaults['fasta_ref'])
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
                self.validation_hash['is_multiline_text'].append({'text': self.params['seq_source'],
                                                                  'alphabet': seq.ALPHABETS[self.params['alphabet']]})
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

        predict_dir = os.path.join(self.params['output_folder'], 'prediction',
                                 f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(predict_dir)

        prepared_file_path = os.path.join(predict_dir, 'mapped.tsv')
        predict_x = []
        if self.params['seq_type'] == 'bed':
            dataset = Dataset(bed_file=self.params['seq_source'], branches=self.params['branches'], category='predict',
                              win=self.params['win'], winseed=self.params['winseed'])
            if 'seq' in self.params['branches'] or 'fold' in self.params['branches']:
                status.text('Parsing reference fasta file...')
                fasta_dict, _ = seq.parse_fasta_reference(self.params['fasta_ref'])
                self.references.update({'seq': fasta_dict, 'fold': fasta_dict})
            status.text(f"Mapping intervals to {len(self.params['branches'])} branch(es) and exporting...")

        elif self.params['seq_type'] == 'fasta' or self.params['seq_type'] == 'text':
            if self.params['seq_type'] == 'fasta':
                dataset = Dataset(fasta_file=self.params['seq_source'], branches=self.params['branches'], category='predict',
                                  win=self.params['win'], winseed=self.params['winseed'])
            elif self.params['seq_type'] == 'text':
                dataset = Dataset(text_input=self.params['seq_source'], branches=self.params['branches'], category='predict',
                                  win=self.params['win'], winseed=self.params['winseed'])

        dataset.map_to_branches(
            self.references, self.params['alphabet'], self.params['strand'], prepared_file_path)
        for branch in self.params['branches']:
            branch_list = dataset.df[branch].to_list()
            predict_x.append(np.array([Dataset.sequence_from_string(seq_str) for seq_str in branch_list]))
            # TODO check effectiveness of the to_list on larger dataset

        status.text('Calculating predictions...')
        model = keras.models.load_model(self.params['model_file'])
        predict_y = model.predict(
            predict_x,
            verbose=1)

        dataset.df['predicted class'] = self.get_klass(predict_y, self.params['klasses'])
        dataset.df[f"raw predicted probabilities ({', '.join(self.params['klasses'])})"] = [Dataset.sequence_to_string(y) for y in predict_y]

        status.text('Exporting results...')
        result_file = os.path.join(predict_dir, 'results.tsv')
        dataset.save_to_file(ignore_cols=self.params['branches'], outfile_path=result_file)  #TODO ignore branches cols ??

        self.finalize_run(logger, predict_dir, self.params, self.csv_header(),
                          self.csv_row(predict_dir, self.params, self.model_folder), self.previous_param_file)
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
            'output_folder': os.path.join(os.getcwd(), 'deepnet_output')
        }

    @staticmethod
    def csv_header():
        return 'Folder\t'\
               'Model file\t' \
               'Branches\t' \
               'Window\t' \
               'Window seed\t' \
               'No. classes\t' \
               'Classes\t' \
               'Sequence source type\t' \
               'Sequence source\t' \
               'Alphabet\t' \
               'Strand\t' \
               'Fasta ref.\t' \
               'Conservation ref\t' \
               'Input folder\n'

    @staticmethod
    def csv_row(folder, params, model_folder):
        return f"{os.path.basename(folder)}\t" \
               f"{params['model_file']}\t" \
               f"{params['branches']}\t" \
               f"{params['win']}\t" \
               f"{params['winseed']}\t" \
               f"{params['no_klasses']}\t" \
               f"{params['klasses']}\t" \
               f"{Predict.get_dict_key(params['seq_type'], Predict.SEQ_TYPES)}\t" \
               f"{params['seq_source']}\t" \
               f"{params['alphabet']}\t" \
               f"{params['strand']}\t" \
               f"{params['fasta_ref']}\t" \
               f"{params['cons_dir']}\t" \
               f"{model_folder}\n"
