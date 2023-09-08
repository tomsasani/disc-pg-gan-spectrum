import allel

CHROMS = list(map(str, range(1, 23)))
CHROMS = [f"chr{c}" for c in CHROMS]
#CHROMS = ["chr1"]

rule all:
    input:
        "data/simulated/simulated.h5",
        "data/simulated/simulated.fa"

rule make_training:
    input: "create_training_sim_data.py"
    output: 
        temp(expand("data/simulated/vcf/{chrom}.simulated.vcf", chrom=CHROMS)),
        temp(expand("data/simulated/ref/{chrom}.simulated.fa", chrom=CHROMS))
    shell:
        """
        python {input} -length 25000000
        """

rule combine_vcf:
    input:
        expand("data/simulated/vcf/{chrom}.simulated.vcf", chrom=CHROMS)
    output:
        "data/simulated/simulated.unfiltered.vcf.gz"
    shell:
        """
        bcftools concat -Oz -o {output} {input}
        """

rule filter_vcf:
    input: "data/simulated/simulated.unfiltered.vcf.gz"
    output: "data/simulated/simulated.vcf.gz"
    shell:
        """
        bcftools view -m2 -M2 -Oz -o {output} {input}
        """

rule combine_ref:
    input:
        expand("data/simulated/ref/{chrom}.simulated.fa", chrom=CHROMS)
    output:
        "data/simulated/simulated.fa"
    shell:
        """
        cat {input} > {output}
        """

rule convert_to_h5:
    input: vcf = "data/simulated/simulated.vcf.gz"
    output: hdf = "data/simulated/simulated.h5"
    run:
        allel.vcf_to_hdf5(
            input.vcf,
            output.hdf,
            fields=['CHROM', 'GT', 'POS', 'REF', 'ALT'],
            overwrite=True,
        )
