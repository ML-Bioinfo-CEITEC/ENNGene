import datetime
import logging
import numpy as np
import os
import streamlit as st
import yaml
import tensorflow as tf


# TODO export the env when releasing, check pandas == 1.1.1
from . import ig
from ..utils.dataset import Dataset
from ..utils.subcommand import Subcommand

logger = logging.getLogger('root')


class Predict(Subcommand):
    SEQ_TYPES = {'BED file': 'bed',
                 'FASTA file': 'fasta',
                 'Text input': 'text'}

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
        self.sequence_options(self.SEQ_TYPES)

        if len(self.params['branches']) == 1 and self.params['branches'][0] == 'seq':
            self.params['ig'] = st.checkbox('Calculate Integrated Gradients', self.defaults['ig'])
            if self.params['ig']:
                st.markdown('###### Note: Integrated Gradients are available only for one-branched models with a sequence branch.')
                st.markdown('###### **WARNING**: Calculating the integrated gradients is a time-consuming process, '
                            'it may take several minutes up to few hours (depending on the number of sequences).')

        self.validate_and_run(self.validation_hash)

    def run(self):
        status = st.empty()
        status.text('Preparing sequences...')

        self.params['predict_dir'] = os.path.join(self.params['output_folder'], 'prediction',
                                 f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(self.params['predict_dir'])

        prepared_file_path = os.path.join(self.params['predict_dir'], 'sequences.tsv')
        predict_x = []

        if self.params['seq_type'] == 'bed':
            dataset = Dataset(bed_file=self.params['seq_source'], branches=self.params['branches'], category='predict',
                              win=self.params['win'], win_place=self.params['win_place'], winseed=self.params['winseed'])
            status.text(f"Mapping intervals to {len(self.params['branches'])} branch(es) and exporting...")

        elif self.params['seq_type'] == 'fasta' or self.params['seq_type'] == 'text':
            if self.params['seq_type'] == 'fasta':
                dataset = Dataset(fasta_file=self.params['seq_source'], branches=self.params['branches'], category='predict',
                                  win=self.params['win'], win_place=self.params['win_place'], winseed=self.params['winseed'])
            elif self.params['seq_type'] == 'text':
                dataset = Dataset(text_input=self.params['seq_source'], branches=self.params['branches'], category='predict',
                                  win=self.params['win'], win_place=self.params['win_place'], winseed=self.params['winseed'])

        dataset.sort_datapoints().map_to_branches(
            self.references, self.params['alphabet'], self.params['strand'], prepared_file_path, status, predict=True, ncpu=self.ncpu)

        for branch in self.params['branches']:
            branch_list = dataset.df[branch].to_list()
            predict_x.append(np.array([Dataset.sequence_from_string(seq_str) for seq_str in branch_list]))
            # TODO check effectiveness of the to_list on larger dataset

        status.text('Calculating predictions...')

        model = tf.keras.models.load_model(self.params['model_file'])
        predict_y = model.predict(
            predict_x,
            verbose=1)

        status.text('Calculating Integrated Gradients... \n'
                    'Note: This is rather slow process, it may take a while.')
        logger.info('Calculating Integrated Gradients...')
        for i, klass in enumerate(self.params['klasses']):
            dataset.df[klass] = [y[i] for y in predict_y]
        dataset.df['highest scoring class'] = self.get_klass(predict_y, self.params['klasses'])

        placeholder = st.empty()
        if len(self.params['branches']) == 1 and self.params['branches'][0] == 'seq' and self.params['ig']:
            status.text('Calculating Integrated Gradients...')
            raw_sequence = dataset.df['input_seq']

            # set baseline, win parameter in yaml and num 5, num of sequence
            baseline = tf.zeros(shape=(self.params['win'], 5))

            visualisations = []
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
                visualisations.append(visualisation)

            dataset.df['Integrated Gradients Visualisation'] = visualisations

            # Show ten best predictions per class in the application window
            st.markdown('---')
            st.markdown('### Integrated Gradients Visualisation')
            st.markdown('Below are ten sequences with highest predicted score per each class. \n'
                        'You can find html visualisation code for all the sequences in the results.tsv file.\n\n'
                        'The higher is the attribution of the sequence to the prediction, the more pronounced is its red color. '
                        'On the other hand, the blue color means low level of attribution.')

            best = dataset.df[self.params['klasses']+['Integrated Gradients Visualisation']]
            for klass in self.params['klasses']:
                st.markdown(f'#### {klass}')
                best.sort_values(by=klass, ascending=False, inplace=True)
                best_ten = best[:10] if (len(best) >= 10) else best

                def visualize(row):
                    st.markdown(f"{row['Integrated Gradients Visualisation']}", unsafe_allow_html=True)
                    return row
                best_ten.apply(visualize, axis=1)

        status.text('Exporting results...')
        result_file = os.path.join(self.params['predict_dir'], 'results.tsv')
        ignore = ['name', 'score'] + self.params['branches']
        dataset.save_to_file(ignore_cols=ignore, outfile_path=result_file)

        header = self.predict_header()
        row = self.predict_row(self.params)

        if self.previous_param_file:
            with open(self.previous_param_file, 'r') as file:
                previous_params = yaml.safe_load(file)
            if 'Train' in previous_params.keys():
                # Parameters missing in older versions of the code
                novel_params = {'auc': None, 'avg_precision': None}
                parameters = previous_params['Train']
                parameters.update(novel_params)
                header += f"{self.train_header()}"
                row += f"{self.train_row(parameters)}"
                if 'Preprocess' in previous_params.keys():
                    novel_params = {'win_place': 'rand'}  # It's always been 'random' for the previous versions
                    parameters = previous_params['Preprocess']
                    parameters.update(novel_params)
                    header += f'{self.preprocess_header()}\n'
                    row += f"{self.preprocess_row(parameters)}\n"
                else:
                    header += '\n'
                    row += '\n'
            else:
                header += '\n'
                row += '\n'

        self.finalize_run(logger, self.params['predict_dir'], self.params, header, row, placeholder, self.previous_param_file)
        status.text('Finished!')
        logger.info('Finished!')

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
            'win_place': 'rand',
            'winseed': 42,
            'ig': True,
            'output_folder': os.path.join(os.path.expanduser('~'), 'enngene_output')
        }

