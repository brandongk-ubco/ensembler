from ensembler.utils import ds_combination
import torch


class TestReporter:
    def test_ds_combination_all_agree(self):
        a = torch.FloatTensor([0, 1, 0])
        b = torch.FloatTensor([0, 1, 0])

        result = ds_combination(a, b)
        expected = torch.FloatTensor([0, 1, 0])
        eps = torch.finfo(a.dtype).eps
        diff = torch.abs(result - expected)

        assert torch.max(diff) <= eps

    def test_ds_combination_all_disagree(self):
        a = torch.FloatTensor([0, 1, 0])
        b = torch.FloatTensor([1, 0, 0])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0, 0, 0])
        eps = torch.finfo(a.dtype).eps
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_uncertainty_in_two(self):
        a = torch.FloatTensor([0.5, 0.5, 0])
        b = torch.FloatTensor([0.5, 0.5, 0])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0.5, 0.5, 0])
        eps = torch.finfo(a.dtype).eps
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_all_different_uncertainty_in_two(self):
        a = torch.FloatTensor([0.3, 0.7, 0])
        b = torch.FloatTensor([0.7, 0.3, 0])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0.5, 0.5, 0])
        eps = torch.finfo(a.dtype).eps
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_all_different_uncertainty_in_two(self):
        a = torch.FloatTensor([0.3, 0.7, 0])
        b = torch.FloatTensor([0, 0.7, 0.3])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0, 1, 0])
        eps = torch.finfo(a.dtype).eps
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_all_different_uncertainty_in_two(self):
        a = torch.FloatTensor([0.3, 0.6, 0.1])
        b = torch.FloatTensor([0.1, 0.6, 0.3])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0.0714, 0.8571, 0.0714])
        eps = torch.finfo(a.dtype).eps
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps