import torch

from smolbge_image_embedding import VisionProjector


def test_projector_shape_and_normalization():
    torch.manual_seed(42)
    model = VisionProjector().eval()
    output = model(torch.randn(4, 768))
    assert output.shape == (4, 384)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_trainable_parameter_count():
    model = VisionProjector()
    assert sum(parameter.numel() for parameter in model.parameters()) == 887_424

