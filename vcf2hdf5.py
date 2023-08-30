"""
Script to convert VCF to HDF5. Samples/sites should already be filtered (if
desired) in the VCF.
Author: Sara Mathieson
Date: 2/4/21
"""

import allel
import argparse
import sys

# example command line
# python3 vcf2hdf5.py -i YRI.vcf.gz -o YRI.h5

def main(args):
    convert_vcf(args.vcf_filename, args.h5_filename)

def convert_vcf(vcf_filename, h5_filename):
    """Convert vcf_filename"""
    # here we save only CHROM, GT (genotypes) and POS (SNP positions)
    # see: https://scikit-allel.readthedocs.io/en/stable/io.html
    allel.vcf_to_hdf5(vcf_filename, h5_filename, fields=['CHROM','GT','POS', 'REF', 'ALT'])

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument('-i', '--vcf_filename', type=str, \
        help='path to input VCF file')
    p.add_argument('-o', '--h5_filename', type=str, \
        help='path to output H5 file')

    args = p.parse_args()
    main(args)