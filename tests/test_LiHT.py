from ensembler.activations import LiHT, PWLinear
import torch


class TestLiHT():
    def test_LiHT(self):
        liht = LiHT()
        t = torch.tensor([
            -2, -1, -0.26, -0.25, -0.24, -0.1, 0, 0.1, 0.24, 0.25, 0.26, 1, 2
        ])
        result = liht(t)
        assert torch.all(result == torch.tensor([
            -1.0000, -1.0000, -1.0000, -1.0000, -0.9600, -0.4000, 0.0000,
            0.4000, 0.9600, 1.0000, 1.0000, 1.0000, 1.0000
        ]))

    def test_PWLinear(self):
        liht = PWLinear()
        t = torch.tensor([
            -2, -1, -0.26, -0.25, -0.24, -0.1, 0, 0.1, 0.24, 0.25, 0.26, 1, 2
        ])
        result = liht(t)
        assert torch.all(result == torch.tensor([
            -1.8750, -1.3750, -1.0050, -1.0000, -0.9600, -0.4000, 0.0000,
            0.4000, 0.9600, 1.0000, 1.0050, 1.3750, 1.8750
        ]))
