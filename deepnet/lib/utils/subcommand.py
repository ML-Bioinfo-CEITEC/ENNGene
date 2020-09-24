from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import streamlit as st
import yaml

from . import validators
from .exceptions import UserInputError

logger = logging.getLogger('root')


# noinspection PyAttributeOutsideInit
class Subcommand:

    BRANCHES = {'Raw sequence': 'seq',
                'Conservation score': 'cons',
                'Secondary structure': 'fold'}

    def general_options(self, branches=True):
        self.params_loaded = False
        self.defaults = {}
        self.defaults.update(self.default_params())
        self.load_params = st.checkbox('Load parameters from a previous run', value=False)

        if self.load_params:
            folder = st.text_input('Path to the output folder from the previous run (must contain the parameters.yaml file)')
            if folder:
                if os.path.isdir(folder):
                    param_file = os.path.join(folder,
                                              ([file for file in os.listdir(folder) if (file == 'parameters.yaml') and
                                                (os.path.isfile(os.path.join(folder, file)))][0]))
                    if param_file:
                        with open(param_file, 'r') as file:
                            try:
                                user_params = yaml.safe_load(file)
                            except Exception as err:
                                logger.exception(f'{err.__class__.__name__}: {err}')
                                raise UserInputError('An error occurred while processing given yaml file.')
                            if self.__class__.__name__ in user_params.keys():
                                self.defaults.update(user_params[self.__class__.__name__])
                                self.params_loaded = True
                            else:
                                raise UserInputError('Found yaml file does not contain parameters for the currently selected task.')
                    else:
                        raise UserInputError('No yaml file was found in the given folder.')
                else:
                    raise UserInputError('Given folder does not exist.')
        self.params.update(self.defaults)

        self.params['output_folder'] = st.text_input(
            'Output path were result files will be exported (cwd used as default)',
            value=self.defaults['output_folder']
        )
        try:
            self.ensure_dir(self.params['output_folder'])
        except Exception:
            raise UserInputError(f"Failed to create output folder at given path: {self.params['output_folder']}.")

        if branches:
            default_branches = [self.get_dict_key(b, self.BRANCHES) for b in self.defaults['branches']]
            self.params['branches'] = list(map(lambda name: self.BRANCHES[name],
                                               st.multiselect('Branches',
                                                              list(self.BRANCHES.keys()),
                                                              default=default_branches)))
            self.validation_hash['not_empty_branches'].append(self.params['branches'])

        if (self.params['task'] == 'Preprocess') and ('fold' in self.params['branches']):
            # currently used only as an option for RNAfold
            max_cpu = os.cpu_count() or 1
            self.ncpu = st.slider('Number of CPUs to be used for folding (max = all available CPUs on the machine).',
                                  min_value=1, max_value=max_cpu, value=max_cpu)
        else:
            self.ncpu = 1

    def model_options(self, blackbox=False, warning=None):
        missing_model = False
        missing_params = False
        model_types = {'Use a model trained by the deepnet app': 'from_app',
                       'Use a custom trained model': 'custom'}

        st.markdown(warning)
        self.params['model_source'] = model_types[st.radio(
            'Select a source of the trained model:',
            list(model_types.keys()), index=self.get_dict_index(self.defaults['model_source'], model_types))]
        if self.params['model_source'] == 'from_app':
            self.model_folder = st.text_input('Path to the folder containing the trained model (hdf5 file)',
                                              value=self.defaults['model_file'])
            if self.model_folder and os.path.isdir(self.model_folder):
                model_files = [f for f in os.listdir(self.model_folder) if f.endswith('.hdf5') and
                               os.path.isfile(os.path.join(self.model_folder, f))]
                if len(model_files) == 0:
                    missing_model = True
                    st.markdown('#### Sorry, there is no hdf5 file in given folder.')
                elif len(model_files) == 1:
                    self.params['model_file'] = os.path.join(self.model_folder, model_files[0])
                    st.markdown(f"###### Model file: {self.params['model_file']}")
                elif len(model_files) > 1:
                    missing_model = True
                    st.markdown(
                        '#### Sorry, there is too many hdf5 files in the given folder. Please specify the model file below.')

                if not blackbox:
                    previous_param_files = [f for f in os.listdir(self.model_folder) if (f == 'parameters.yaml') and
                                   (os.path.isfile(os.path.join(self.model_folder, f)))]
                    if len(previous_param_files) == 0:
                        missing_params = True
                        st.markdown('#### Sorry, could not find parameters.yaml file in the given folder. '
                                    'Check the folder or specify the parameters below.')
                    elif len(previous_param_files) == 1:
                        training_params = {'win': None, 'winseed': None, 'no_klasses': 0, 'klasses': []}
                        self.previous_param_file = os.path.join(self.model_folder, previous_param_files[0])
                        with open(self.previous_param_file, 'r') as file:
                            user_params = yaml.safe_load(file)
                        try:
                            training_params['win'] = user_params['Preprocess']['win']
                            training_params['winseed'] = user_params['Preprocess']['winseed']
                            training_params['no_klasses'] = len(user_params['Preprocess']['klasses'])
                            training_params['klasses'] = user_params['Preprocess']['klasses']
                        except:
                            missing_params = True
                            st.markdown('#### Sorry, could not read the parameters from given folder. '
                                        'Check the folder or specify the parameters below.')
                        # TODO zajistit konzistentni poradi klass names
                        if not training_params['win'] or not training_params['winseed'] \
                                or training_params['no_klasses'] == 0 or len(training_params['klasses']) == 0 \
                                or len(training_params['klasses']) != training_params['no_klasses']:
                            missing_params = True
                            st.markdown('#### Sorry, could not read the parameters from given folder. '
                                        'Check the folder or specify the parameters below.')
                        else:
                            st.markdown('###### Parameters read from given folder:\n'
                                        f"* Window size: {training_params['win']}\n"
                                        f"* Window seed: {training_params['winseed']}\n"
                                        f"* No. of classes: {training_params['no_klasses']}\n"
                                        f"* Class labels: {training_params['klasses']}")
                            self.params.update(training_params)
                    if len(previous_param_files) > 1:
                        missing_params = True
                        st.markdown('#### Sorry, there is too many parameters.yaml files in the given folder. '
                                    'Check the folder or specify the parameters below.')

        if self.params['model_source'] == 'custom' or missing_params or missing_model:
            if not missing_params:
                self.params['model_file'] = st.text_input('Path to the trained model (hdf5 file)',
                                                          value=self.defaults['model_file'])
            if not blackbox and not missing_model:
                self.params['win'] = int(
                    st.number_input('Window size used for training', min_value=3, value=self.defaults['win']))
                self.params['winseed'] = int(st.number_input('Seed for semi-random window placement upon the sequences',
                                                             value=self.defaults['winseed']))
                self.params['no_klasses'] = int(st.number_input('Number of classes used for training', min_value=2,
                                                                value=self.defaults['no_klasses']))
                # mozna zachovat v nejakym meta souboru ktera trida je co a podle toho
                for i in range(self.params['no_klasses']):
                    if len(self.params['klasses']) >= i + 1:
                        value = self.params['klasses'][i]
                    else:
                        value = str(i)
                        self.params['klasses'].append(value)
                    self.params['klasses'][i] = st.text_input(f'Class {i} label:', value=value)

        self.validation_hash['is_model_file'].append(self.params['model_file'])

    def validate_and_run(self, validation_hash):
        st.markdown('---')
        if st.button('Run'):
            warnings = self.validate_input(validation_hash)
            if len(warnings) == 0:
                logger.info('\n'.join(['%s: %s' % (key, value) for (key, value) in self.params.items()]))
                self.run()
            else:
                st.warning('  \n'.join(warnings))

    @staticmethod
    def validate_input(validation_hash):
        warnings = []
        for validator, items in validation_hash.items():
            for item in items:
                if type(item) == dict:
                    warnings.append(getattr(validators, validator)(**item))
                else:
                    warnings.append(getattr(validators, validator)(item))

        return list(filter(None, warnings))

    @staticmethod
    def spent_time(time1):
        time2 = datetime.now()
        t = time2 - time1
        return t

    @staticmethod
    def ensure_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def get_dict_index(value, dictionary):
        return list(dictionary.values()).index(value)

    @staticmethod
    def get_dict_key(value, dictionary):
        index = Subcommand.get_dict_index(value, dictionary)
        return list(dictionary.keys())[index]

    @staticmethod
    def finalize_run(logger, out_dir, user_params, csv_header, csv_row, previous_param_file=None):
        st.text(f'You can find your results at {out_dir}')
        params = user_params.copy()
        task = params.pop('task')
        params = {task: user_params}
        if previous_param_file:
            with open(previous_param_file, 'r') as file:
                previous_params = yaml.safe_load(file)
            params.update(previous_params)

        with open(os.path.join(out_dir, 'parameters.yaml'), 'w') as file:
            yaml.dump(params, file)

        parent_dir = Path(out_dir).parent
        table_file = os.path.join(parent_dir, 'parameters.tsv')
        if table_file:
            write_header = not os.path.isfile(table_file)
            with open(table_file, 'a') as file:
                file.write(csv_header) if write_header else None
                file.write(csv_row)
                pass

        file_handler = [handler for handler in logger.handlers if type(handler) == logging.FileHandler]
        if file_handler:
            logfile_path = file_handler[0].baseFilename
            logfile_name = os.path.basename(logfile_path)
            shutil.move(logfile_path, os.path.join(out_dir, logfile_name))
