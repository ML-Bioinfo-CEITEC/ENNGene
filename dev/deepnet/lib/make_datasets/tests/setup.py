import os, sys
from os.path import dirname as parent_dir
import random as rn


BASE_DEEPNET_DIR = parent_dir(parent_dir(parent_dir(parent_dir(os.path.abspath(__file__)))))
TEST_FILES_DIR = os.path.join(parent_dir(parent_dir(parent_dir(parent_dir(parent_dir(os.getcwd()))))), 'files/test_set')
sys.path.append(BASE_DEEPNET_DIR)
sys.path.append(TEST_FILES_DIR)

from deepnet import *


def get_possible_arguments():
    return ["coord", "ref", "consdir", "branches", "onehot", "strand"]


def random_argument_generator(shuffles=10):
    coord = [os.path.join(TEST_FILES_DIR, file) for file in os.listdir(TEST_FILES_DIR) if file.endswith('.bed')]
    ref = [os.path.join(TEST_FILES_DIR, file) for file in os.listdir(TEST_FILES_DIR) if file.endswith('.fa')]
    consdir = [os.path.join(TEST_FILES_DIR, file) for file in os.listdir(TEST_FILES_DIR) if file.endswith('way')]
    branches = ['seq', 'cons']
    onehot = ['C', 'G', 'T', 'A', 'N']
    strand = ['True', 'False']
    for shuffle in range(shuffles):
        branches_sample = rn.sample(branches, rn.randint(1, len(branches)))
        rn.shuffle(onehot)
        strand_choice = rn.choice(strand)
        yield {k:v for k, v in zip(get_possible_arguments(), \
               [coord, ref, consdir, branches_sample, onehot, strand_choice])}


def set_default_args(kwargs):
    default_args = []
    for key in kwargs.keys():
        args = kwargs[key]
        args = kwargs[key]
    for key in kwargs.keys():
        args = kwargs[key]
        key = '--' + key
        default_args.append(key)
        if type(args) == list:
            for arg in args:
                default_args.append(arg)
        else:
            default_args.append(args)
    return default_args


def create_random_arguments(shuffles):
    for rand_args in random_argument_generator(shuffles):
        yield set_default_args(rand_args)


def make_datasets(default_args):
    testing_module = 'make_datasets'
    test_module_path = "lib.{}.{}".format(testing_module, testing_module)
    subcommand_class = dirname_to_class(testing_module)
    module = __import__(test_module_path, fromlist=[subcommand_class])
    subcommand = getattr(module, subcommand_class)
    make_datasets = subcommand(default_args=default_args)
    return make_datasets.run()
