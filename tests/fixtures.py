import pytest
import numpy as np
import mutyper

@pytest.fixture
def ancestor():
    return mutyper.Ancestor("tests/test_data/test.fa", k=1)

@pytest.fixture
def haplotypes():
    # sites x haps
    return np.array([
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
    ])

@pytest.fixture
def positions():
    return np.array([4, 7, 12, 18, 19])

@pytest.fixture
def distance_vec():
    return np.array([0, 0.25, 0.3, 0.05, 0.1])

@pytest.fixture
def references():
    nucs = ["A", "A", "T", "T", "A"]
    nucs = [n.encode("utf-8") for n in nucs]
    return nucs

@pytest.fixture
def alternates():
    nucs = ["T", "C", "C", "G", "T"]
    nucs = [(n.encode("utf-8"), ) for n in nucs]
    return nucs


@pytest.fixture
def fake_image():
    rng = np.random.default_rng(1833)
    # site x haps x channels
    img = rng.random(size=(5, 3, 6))
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

# @pytest.fixture
# def unprocessed_image_and_dist():
#     rng = np.random.default_rng(1833)
#     # site x haps x channels
#     return rng.poisson(lam = 0.01, size=(5, 10, 6)),

@pytest.fixture
def unpolarized_image():
    # haps x sites
    return np.array([
        [0, 0, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
    ])

@pytest.fixture
def polarized_image():
    # haps x sites
    return np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
