"""Smoke tests on model output shapes."""
import torch

from src.model import GMF, MLP, NeuMF, build_model


def test_gmf_forward_shape():
    m = GMF(num_users=10, num_items=20, embed_dim=4)
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([5, 6, 7])
    out = m(u, i)
    assert out.shape == (3,)


def test_mlp_forward_shape():
    m = MLP(num_users=10, num_items=20, embed_dim=4, layers=[8, 4])
    u = torch.tensor([0, 1, 2, 3])
    i = torch.tensor([5, 6, 7, 8])
    out = m(u, i)
    assert out.shape == (4,)


def test_neumf_forward_shape():
    m = NeuMF(
        num_users=10,
        num_items=20,
        embed_dim_gmf=4,
        embed_dim_mlp=4,
        layers=[8, 4],
    )
    u = torch.tensor([0, 1])
    i = torch.tensor([5, 6])
    out = m(u, i)
    assert out.shape == (2,)


def test_build_model_factory():
    cfg = {
        "type": "neumf",
        "embed_dim_gmf": 4,
        "embed_dim_mlp": 4,
        "mlp_layers": [8, 4],
        "dropout": 0.0,
    }
    m = build_model(5, 7, cfg)
    assert isinstance(m, NeuMF)

    cfg["type"] = "gmf"
    m = build_model(5, 7, cfg)
    assert isinstance(m, GMF)

    cfg["type"] = "mlp"
    m = build_model(5, 7, cfg)
    assert isinstance(m, MLP)
