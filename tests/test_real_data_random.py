from real_data_random import prep_real_region
from util import process_region, sum_across_channels, reorder

import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@pytest.mark.parametrize("chrom,n_haps,n_snps", [("chr1", 4, 5)])
def test_prep_real_region(
    haplotypes,
    positions,
    references,
    alternates,
    ancestor,
    chrom,
    n_haps,
    n_snps,
):
    out_arr, dist_vec = prep_real_region(
        haplotypes,
        positions,
        references,
        alternates,
        ancestor,
        chrom,
        n_haps,
        n_snps,
    )

    assert out_arr.shape[0] == n_snps
    assert out_arr.shape[1] == n_haps
    assert out_arr.shape[2] == 6

    assert np.array_equal(dist_vec, np.array([0, 3, 5, 6, 1]) / 15)


@pytest.mark.parametrize("X", [(np.empty(shape=(10, 100, 6)))])
def test_reorder(X):
    reordered = reorder(X)

    assert reordered.shape[0] == 100
    assert reordered.shape[1] == 10
    assert reordered.shape[2] == 6


def test_sum_across_channels(fake_image):
    res = sum_across_channels(fake_image)

    assert res.shape[0] == 5
    assert res.shape[1] == 3
    assert res.shape[2] == 1

    print (res)

@pytest.mark.parametrize("neg1,n_snps,n_channels", [(True, 36, 2)])
def test_process_region(fake_image, distance_vec, neg1, n_snps, n_channels):
    processed = process_region(
        fake_image,
        distance_vec,
        neg1=neg1,
        n_snps=n_snps,
        n_channels=n_channels,
    )
    print (processed.shape)
    
    
    assert processed.shape[0] == 5
