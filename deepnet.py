import os
import streamlit as st
import sys
import logging

sys.path.append(os.getcwd())
logging.basicConfig(filename='app.log',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

logger = logging.getLogger('main')
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',  datefmt='%m/%d/%Y %I:%M:%S %p')
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


class DeepNet:

    def __init__(self):
        st.sidebar.title('Deepnet App')

        available_subcommands = {'Preprocess Data': 'make_datasets',
                                 'Tune hyperparameters': 'train',
                                 'Train a network': 'train'}

        subcommand = available_subcommands[st.sidebar.selectbox(
            'Select a task to be run:',
            list(available_subcommands.keys())
        )]
        logger.info(f'DeepNet started with the following subcommand: {subcommand}')

        module_path = f'lib.{subcommand}.{subcommand}'
        subcommand_class = ''.join(x.title() for x in subcommand.split('_'))
        module = __import__(module_path, fromlist=[subcommand_class])
        # use dispatch pattern to invoke object of class with same name as the subcommand
        subcommand = getattr(module, subcommand_class)
        subcommand()


if __name__ == '__main__':
    DeepNet()
