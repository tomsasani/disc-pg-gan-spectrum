import numpy as np 
import random 

N_TRIALS = 10

seeds = [np.random.randint(1, 2**32) for _ in range(N_TRIALS)]
entropy = ["n" for _ in seeds]

seed2entropy = dict(zip(list(map(str, seeds)), entropy))

rule all:
    input: expand("saved_model/{seed}/fingerprint.pb", seed=seeds)


rule train:
    input:
        py_script = "pg_gan.py",
    output: "saved_model/{seed}/fingerprint.pb"
    params: entropy = lambda wcs: seed2entropy[wcs.seed]
    shell:
        """
        python {input.py_script} -disc {wildcards.seed} \
                                 -params rho,N_anc,T_split,mig,N1,N2 \
                                 -seed {wildcards.seed} \
                                 -entropy {params.entropy}
        """
