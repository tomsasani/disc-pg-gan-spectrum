CHROMS = list(map(str, range(1, 23)))
CHROMS = [f"chr{c}" for c in CHROMS]

rule all:
    input:
        "data/simulated/simulated.vcf.gz",
        "data/simulated/simulated.fa"

rule make_training:
    input: "create_training_sim_data.py"
    output: 
        temp(expand("data/simulated/vcf/{chrom}.simulated.vcf", chrom=CHROMS)),
        temp(expand("data/simulated/ref/{chrom}.simulated.fa", chrom=CHROMS))
    shell:
        """
        python {input} -length 10000000
        """


rule combine_vcf:
    input:
        expand("data/simulated/vcf/{chrom}.simulated.vcf", chrom=CHROMS)
    output:
        "data/simulated/simulated.vcf.gz"
    shell:
        """
        bcftools concat -Oz -o {output} {input}
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
