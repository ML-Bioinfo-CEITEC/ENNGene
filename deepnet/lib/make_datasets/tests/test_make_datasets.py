from setup import make_datasets, create_random_arguments
import pytest


class TestMakeDatasets(object):

    datasets_names = ['train', 'test', 'validation', 'blackbox']
    arguments_shuffles = 50


    @pytest.yield_fixture
    def create_data(self):
        test_data = []
        for default_args in create_random_arguments(self.arguments_shuffles):
            datasets = make_datasets(default_args)
            test_data.append(datasets)
        yield test_data


    def test_get_data_branches_len(self, create_data):
        """Test if corresponding datasets in different branches have same length"""
        for test_data in create_data:
            if len(test_data) > 1:  # more than 1 branch
                branch_1, branch_2 = list(test_data.keys())
                for dataset in self.datasets_names:
                    if dataset == 'train':
                        continue
                    assert len(test_data[branch_1][dataset].dictionary) == \
                           len(test_data[branch_2][dataset].dictionary)


    def test_output_datasets_is_empty(self, create_data):
        """Test if any of output datasets is empty"""
        for test_data in create_data:
            for branch in test_data.keys():
                for dataset in self.datasets_names:
                    assert test_data[branch][dataset].dictionary


    @pytest.mark.parametrize('expected', [list])
    def test_datasets_data_type(self, create_data, expected):
        for test_data in create_data:
            for branch in test_data.keys():
                for dataset in self.datasets_names:
                    assert all(type(value) is expected for value \
                        in test_data[branch][dataset].dictionary.values())


    @pytest.mark.parametrize('expected', [dict])
    def test_output_data_type(self, create_data, expected):
        for test_data in create_data:
            assert type(test_data) is expected
