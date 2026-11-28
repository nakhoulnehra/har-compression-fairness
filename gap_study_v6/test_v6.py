"""Gap study v6 checks: python -m pytest gap_study_v6/test_v6.py -q"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "bit-reduction-test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from har.models import HarModel, TinyHAR  # noqa: E402
from gap_study_v6.run_v6 import slice_tinyhar  # noqa: E402
import gap_study_v6.quant_v6 as Q  # noqa: E402


def _net():
    torch.manual_seed(0)
    return HarModel("tinyhar", {"filters": 28}, TinyHAR(), torch.zeros(6), torch.ones(6)).eval()


def test_slice_full_width_is_identity():
    net = _net()
    x = torch.randn(4, 6, 128)
    assert torch.equal(net(x), slice_tinyhar(net, 28).eval()(x))


def test_slice_width_12_param_count():
    assert sum(p.numel() for p in slice_tinyhar(_net(), 12).parameters()) == 10401


def test_gptq_on_grid_and_p8():
    net = _net()
    X = torch.randn(32, 6, 128)
    q = Q.gptq_quantize(net, 2, X, protect=True)
    for row in q.backbone.lstm.weight_ih_l0.detach():
        assert len(torch.unique(row)) <= 3
    assert max(len(torch.unique(r)) for r in q.backbone.convs[0].weight.detach().reshape(28, -1)) > 3
