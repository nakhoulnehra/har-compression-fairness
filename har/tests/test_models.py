"""Tests for har/models.py and the single-output SHAP call (plan Section 2.10, test_models.py)."""

import numpy as np
import pytest
import torch

from har.attributions import shap_target_class
from har.models import DEEP_MODELS, build_model, n_params, resolve_config

MEAN = np.array([-0.9, 0.0, -0.2, -0.5, 0.0, 0.4], np.float32)
STD = np.array([0.3, 0.2, 0.3, 0.5, 0.3, 0.6], np.float32)
# Expected parameter counts at the default (paper) width, pinned so silent changes are caught.
EXPECTED_PARAMS = {"cnn1d": 1_254_919, "tinyhar": 54_273, "deepconvlstm": 458_183, "hart": 1_389_555}


def make(name, width=1.0, seed=0):
    torch.manual_seed(seed)
    return build_model(name, resolve_config(name, width), MEAN, STD)


@pytest.mark.parametrize("name", DEEP_MODELS)
@pytest.mark.parametrize("width", [1.0, 0.5])
def test_output_shape_and_backward(name, width):
    m = make(name, width)
    x = torch.randn(5, 6, 128)
    y = m(x)
    assert y.shape == (5, 7)
    y.sum().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)
    print(f"{name} width {width}: {n_params(m):,} parameters")


@pytest.mark.parametrize("name", DEEP_MODELS)
def test_parameter_count(name):
    assert n_params(make(name)) == EXPECTED_PARAMS[name]
    assert n_params(make(name, 0.5)) < EXPECTED_PARAMS[name] / 2.5


def test_hart_size_close_to_paper():
    # Ek et al. report 1,445,918 parameters for HART [HART Table 7, p13] (with sizes they do not state)
    assert abs(n_params(make("hart")) - 1_445_918) / 1_445_918 < 0.05


def test_cnn1d_follows_harth_section_4_1():
    m = make("cnn1d").backbone
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv1d)]
    assert [c.kernel_size[0] for c in convs] == [6, 12, 12, 32]           # kernel sizes per layer
    assert all(c.out_channels == 128 for c in convs)                       # 128 kernels in each layer
    assert [l.out_features for l in m.fc if isinstance(l, torch.nn.Linear)] == [512, 512]
    assert {l.p for l in m.modules() if isinstance(l, torch.nn.Dropout)} == {0.4}
    with torch.no_grad():                                                  # 'same' padding keeps T = 128
        assert m.features(torch.randn(2, 6, 128)).shape == (2, 128, 128)


def test_deepconvlstm_follows_table_1_and_section_3_3():
    m = make("deepconvlstm").backbone
    convs = [l for l in m.convs if isinstance(l, torch.nn.Conv2d)]
    assert [n_params(c) for c in convs] == [384, 20_544, 20_544, 20_544]  # [DCL Table 1]
    assert all(c.padding == (0, 0) and c.kernel_size == (1, 5) for c in convs)
    w = m.lstm.weight_hh_l0.detach()                                       # orthogonal init [DCL Sec 3.3]
    assert torch.allclose(w.T @ w, torch.eye(w.shape[1]), atol=1e-4)
    with torch.no_grad():                                                  # 4 valid convs: 128 -> 112
        assert m.convs(torch.randn(1, 1, 6, 128)).shape[-1] == 112


def test_tinyhar_follows_section_3_2():
    m = make("tinyhar").backbone
    convs = [l for l in m.convs if isinstance(l, torch.nn.Conv2d)]
    assert len(convs) == 4 and all(c.kernel_size == (5, 1) and c.stride == (2, 1) for c in convs)
    assert convs[0].out_channels == 28                                     # F = 28 [TINY Sec 4.2]
    assert m.fusion[0].out_features == 2 * 28 and m.lstm.hidden_size == 2 * 28 and m.lstm.num_layers == 1
    with torch.no_grad():                                                  # stride 2 in each layer: T* = 8
        assert m.convs(torch.randn(1, 1, 128, 6)).shape[2] == 8


@pytest.mark.parametrize("name", DEEP_MODELS)
def test_eval_mode_is_deterministic(name):
    m = make(name).eval()
    x = torch.randn(3, 6, 128)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))


@pytest.mark.parametrize("name", DEEP_MODELS)
def test_stored_normalisation_applied_unchanged(name):
    m = make(name).eval()
    assert {"norm.mean", "norm.std"} <= set(m.state_dict())
    assert not any(n.startswith("norm.") for n, _ in m.named_parameters())     # buffers, not trained
    x = torch.randn(4, 6, 128) * 0.5 + 1.0                                     # e.g. HAR70+-like input
    with torch.no_grad():
        expected = m.backbone((x - torch.tensor(MEAN).view(1, 6, 1)) / torch.tensor(STD).view(1, 6, 1))
        assert torch.allclose(m(x), expected, atol=1e-6)
    assert torch.equal(m.norm.mean.view(-1), torch.tensor(MEAN))


@pytest.mark.parametrize("name", DEEP_MODELS)
def test_head_is_the_final_linear_layer(name):
    m = make(name)
    assert isinstance(m.head, torch.nn.Linear) and m.head.out_features == 7


def test_hart_branches_see_only_their_sensor():
    m = make("hart").eval()
    x = torch.randn(2, 6, 128)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(2, 3, 128)                 # change the thigh sensor only
    with torch.no_grad():
        e1, e2 = m.backbone.embed(m.norm(x)), m.backbone.embed(m.norm(x2))
    half = e1.shape[-1] // 2
    assert torch.equal(e1[..., :half], e2[..., :half])       # back embedding unchanged
    assert not torch.equal(e1[..., half:], e2[..., half:])   # thigh embedding changed


@pytest.mark.parametrize("name", DEEP_MODELS)
def test_single_output_shap_on_two_windows(name):
    m = make(name).eval()
    bg = torch.randn(10, 6, 128)
    x = torch.randn(2, 6, 128)
    A = shap_target_class(m, bg, x, target=np.array([0, 3]), shap_seed=0, nsamples=8)
    assert A.shape == (2, 6, 128) and A.dtype == np.float32 and np.isfinite(A).all()


def test_single_output_shap_completeness_on_a_linear_model():
    torch.manual_seed(0)
    lin = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(6 * 128, 7))
    # near-constant background: GradientExplainer samples background rows, so with a spread-out
    # background the sum matches f(x) - mean f(sampled rows), not the full background mean
    bg = 0.5 + 0.001 * torch.randn(50, 6, 128)
    x = torch.randn(3, 6, 128)
    target = np.array([1, 4, 1])
    A = shap_target_class(lin, bg, x, target, nsamples=500)
    with torch.no_grad():
        fx = lin(x)[torch.arange(3), torch.as_tensor(target)]
        fb = lin(bg).mean(0)[torch.as_tensor(target)]
    assert np.allclose(A.reshape(3, -1).sum(1), (fx - fb).numpy(), atol=1e-3)
