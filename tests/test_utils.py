from ensembler.utils import ds_combination
import torch


class TestReporter:
    def test_ds_combination_all_agree(self):
        a = torch.FloatTensor([0, 1, 0])
        b = torch.FloatTensor([0, 1, 0])

        result = ds_combination(a, b)
        expected = torch.FloatTensor([0, 1, 0])
        eps = 1e-4
        diff = torch.abs(result - expected)

        assert torch.max(diff) <= eps

    def test_ds_combination_all_disagree(self):
        a = torch.FloatTensor([0, 1, 0])
        b = torch.FloatTensor([1, 0, 0])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0, 0, 0])
        eps = 1e-4
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_uncertainty_in_two(self):
        a = torch.FloatTensor([0.5, 0.5, 0])
        b = torch.FloatTensor([0.5, 0.5, 0])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0.5, 0.5, 0])
        eps = 1e-4
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_all_different_uncertainty_in_two(self):
        a = torch.FloatTensor([0.3, 0.7, 0])
        b = torch.FloatTensor([0.7, 0.3, 0])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0.5, 0.5, 0])
        eps = 1e-4
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_all_different_uncertainty_in_two(self):
        a = torch.FloatTensor([0.3, 0.7, 0])
        b = torch.FloatTensor([0, 0.7, 0.3])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0, 1, 0])
        eps = 1e-4
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_all_different_uncertainty_in_two(self):
        a = torch.FloatTensor([0.3, 0.6, 0.1])
        b = torch.FloatTensor([0.1, 0.6, 0.3])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0.0714, 0.8571, 0.0714])
        eps = 1e-4
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps

    def test_ds_combination_example_from_slide(self):
        a = torch.FloatTensor([0.35, 0.06, 0.35, 0.24])
        b = torch.FloatTensor([0.10, 0.44, 0.40, 0.06])
        result = ds_combination(a, b)
        expected = torch.FloatTensor([0.1622, 0.1223, 0.6487, 0.0667])
        eps = 1e-4
        diff = torch.abs(result - expected)
        assert torch.max(diff) <= eps
