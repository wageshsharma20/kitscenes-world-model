"""Tests for the chain_warps option and the per-frame coverage diagnostic.

`chain_warps` makes each frame a warp of the previously rendered frame instead of a warp
of t=0. Ground-truth A/B on the validation split showed it is worse (compounding resampling
blur: `hf` 0.78 -> 0.68 against real footage), so the default must stay off. These tests
exist to keep the flag plumbed correctly and to keep the coverage diagnostic populated.
"""

import numpy as np
import pytest

from kitscenes import spec
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel


@pytest.fixture
def dummy_conditioning_images():
    rng = np.random.default_rng(17)
    base = rng.integers(0, 256, size=(spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    return {cam: base for cam in spec.CAMERAS}


@pytest.fixture
def dummy_trajectory():
    """Move forward and drift right, so every frame needs a real warp."""
    return [[1.2 * (k + 1), 0.25 * (k + 1)] for k in range(spec.WAYPOINT_COUNT)]


def test_chain_warps_defaults_off():
    model = Depth3DFullFoVWorldModel(render_scale=1)
    assert model.chain_warps is False


@pytest.mark.parametrize("chain", [False, True])
def test_hole_fraction_diagnostic_is_populated(chain, dummy_conditioning_images, dummy_trajectory):
    model = Depth3DFullFoVWorldModel(render_scale=1, chain_warps=chain)
    videos = model.generate_scenario(
        "0000902", dummy_trajectory, dummy_conditioning_images, cameras=["front"]
    )

    clip = videos["front"]
    assert clip.shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
    assert clip.dtype == np.uint8

    fractions = model.last_hole_fraction["front"]
    assert len(fractions) == spec.FRAME_COUNT
    assert all(0.0 <= value <= 1.0 for value in fractions)


def test_chain_warps_stays_bounded_under_forward_motion(
    dummy_conditioning_images, dummy_trajectory
):
    """Chaining must not blow the image up: a warp of a warp has to stay in range."""
    model = Depth3DFullFoVWorldModel(render_scale=1, chain_warps=True)
    videos = model.generate_scenario(
        "0000902", dummy_trajectory, dummy_conditioning_images, cameras=["front"]
    )

    clip = videos["front"].astype(np.float32)
    assert clip.min() >= 0.0 and clip.max() <= 255.0
    # A constant-ish source must not acquire a hard black band from sampling off the edge.
    mean_per_frame = clip.mean(axis=(1, 2, 3))
    assert mean_per_frame.min() > 0.35 * mean_per_frame.max()
