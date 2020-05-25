from datetime import datetime
import logging
import os
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

    def add_general_options(self, branches=True):
        self.params_loaded = False
        self.load_params = st.checkbox('Load parameters from previous run', value=False)
        if self.load_params:
            param_file = st.text_input('Path to parameters.yaml file')
            if param_file:
                if os.path.isfile(param_file):
                    with open(param_file, 'r') as file:
                        try:
                            user_params = yaml.load(file)
                            user_params['task']
                        except Exception as err:
                            logger.exception(f'{err.__class__.__name__}: {err}')
                            raise UserInputError("An error occurred while processing given yaml file.")
                        if user_params['task'] == self.__class__.__name__:
                            self.defaults = user_params
                            self.params_loaded = True
                        else:
                            raise UserInputError('Given file contains parameters from a different task then currently selected.')
                else:
                    raise UserInputError('Given yaml file does not exist.')
            else:
                self.defaults = self.default_params()
        else:
            self.defaults = self.default_params()

        self.params['output_folder'] = st.text_input(
            'Output path were result files will be exported (cwd used as default)',
            value=self.defaults['output_folder']
        )
        try:
            self.ensure_dir(self.params['output_folder'])
        except Exception:
            raise UserInputError(f"Failed to create output folder at given path: {self.params['output_folder']}.")

        if branches:
            default_branches = [list(self.BRANCHES.keys())[list(self.BRANCHES.values()).index(b)] for b in self.defaults['branches']]
            self.params['branches'] = list(map(lambda name: self.BRANCHES[name],
                                               st.multiselect('Branches',
                                                              list(self.BRANCHES.keys()),
                                                              default=default_branches)))
            self.validation_hash['not_empty_branches'].append(self.params['branches'])

        # FIXME does not work with streamlit
        # max_cpu = os.cpu_count() or 1
        # self.ncpu = st.slider('Number of CPUs to be used. 0 to use all available CPUs.',
        #                       min_value=0, max_value=max_cpu, value=0)
        # if self.ncpu == 0:
        #     self.ncpu = max_cpu
        self.ncpu = 1

        # self.verbose = self.args.verbose

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
    def finalize_run(logger, out_dir, params):
        with open(os.path.join(out_dir, 'parameters.yaml'), 'w') as file:
            yaml.dump(params, file)
        file_handler = [handler for handler in logger.handlers if type(handler) == logging.FileHandler]
        if file_handler:
            logfile_path = file_handler[0].baseFilename
            logfile_name = os.path.basename(logfile_path)
            shutil.move(logfile_path, os.path.join(out_dir, logfile_name))
