from datetime import datetime
import os
import shutil
import streamlit as st


# noinspection PyAttributeOutsideInit
class Subcommand:

    BRANCHES = {'Raw sequence': 'seq',
                'Conservation score': 'cons',
                'Secondary structure': 'fold'}

    def add_general_options(self):
        self.output_folder = st.text_input(
            'Output path were result files will be exported (cwd used as default)',
            value=os.path.join(os.getcwd(), 'deepnet_output')
        )
        self.ensure_dir(self.output_folder)
        self.branches = list(map(lambda name: self.BRANCHES[name], st.multiselect('Branches', list(self.BRANCHES.keys()))))

        # FIXME does not work with streamlit
        # max_cpu = os.cpu_count() or 1
        # self.ncpu = st.slider('Number of CPUs to be used. 0 to use all available CPUs.',
        #                       min_value=0, max_value=max_cpu, value=0)
        # if self.ncpu == 0:
        #     self.ncpu = max_cpu
        self.ncpu = 1

        # self.verbose = self.args.verbose

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
    def finalize_run(logger, out_dir):
        logfile_path = logger.handlers[0].baseFilename
        logfile_name = os.path.basename(logfile_path)
        shutil.move(logfile_path, os.path.join(out_dir, logfile_name))