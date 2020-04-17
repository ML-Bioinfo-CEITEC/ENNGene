import streamlit as st  # TODO outside try-except, add check if installed
import logging

from lib.utils.exceptions import MyException

try:
    logger = logging.getLogger('root')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',  datefmt='%m/%d/%Y %I:%M:%S %p')

    if len(logger.handlers) == 0:  # TODO improve condition to check for the exact handler
        file_handler = logging.FileHandler('app.log', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
except Exception as err:
    st.warning(f'Failed to load logger, continuing without logging...')


def deepNet():
    st.sidebar.title('Deepnet App')

    available_subcommands = {'Preprocess Data': 'make_datasets',
                             'Train a Model': 'train'}

    subcommand = available_subcommands[st.sidebar.selectbox(
        'Select a task to be run:',
        list(available_subcommands.keys())
    )]

    st.sidebar.markdown('')
    st.sidebar.markdown('[Documentation](https://gitlab.com/RBP_Bioinformatics/deepnet/-/blob/master/README.md)')
    st.sidebar.markdown('[FAQ](https://gitlab.com/RBP_Bioinformatics/deepnet/-/blob/master/FAQ.md)')
    st.sidebar.markdown('[GitLab](https://gitlab.com/RBP_Bioinformatics/deepnet)')

    logger.debug(f'DeepNet started with the following subcommand: {subcommand}')

    module_path = f'lib.{subcommand}.{subcommand}'
    subcommand_class = ''.join(x.title() for x in subcommand.split('_'))
    module = __import__(module_path, fromlist=[subcommand_class])
    # use dispatch pattern to invoke object of class with same name as the subcommand
    subcommand = getattr(module, subcommand_class)
    subcommand()


if __name__ == '__main__':
    try:
        deepNet()
    except MyException as err:
        logger.exception(f'{err.__class__.__name__}: {err}')
        st.warning(f'{err}\n Exiting...')
    except Exception as err:
        logger.exception(f'{err.__class__.__name__}: {err}')
        st.warning(
            'Unexpected error occurred in the application. For more details check the log file in your output directory. Exiting...')
    finally:
        # move the log file to appropriate folder
        pass
