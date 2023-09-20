import allel
import pandas as pd

CHROMS = list(map(str, range(1, 23)))
CHROMS = [f"chr{c}" for c in CHROMS]

df = pd.read_csv("hg38.genome", sep="\t")
df = df[df["chrom"].isin(CHROMS)]

CHROMOSOMES, LENGTHS = df["chrom"].to_list(), df["size"].to_list()
chrom2length = dict(zip(CHROMOSOMES, LENGTHS))

rule all:
    input:
        "data/simulated/simulated.h5",
        "data/simulated/simulated.fa"

rule make_training:
    input: "create_training_sim_data.py"
    output: 
        "data/simulated/vcf/{chrom}.simulated.vcf",
        "data/simulated/ref/{chrom}.simulated.fa"
    params:
        length = lambda wcs: chrom2length[wcs.chrom]
    shell:
        """
        python {input} --chrom {wildcards.chrom} -length {params.length}
        """

rule combine_vcf:
    input:
        expand("data/simulated/vcf/{chrom}.simulated.vcf", chrom=CHROMOSOMES)
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
        bcftools view -m2 -M2 -Oz -c1 -C199 -o {output} {input}
        """

rule combine_ref:
    input:
        expand("data/simulated/ref/{chrom}.simulated.fa", chrom=CHROMOSOMES)
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
