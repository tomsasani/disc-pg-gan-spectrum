import numpy as np 
import random 

N_TRIALS = 10

reg_seeds = [np.random.randint(1, 2**32) for _ in range(N_TRIALS)]
spectrum_seeds = [np.random.randint(1, 2**32) for _ in range(N_TRIALS)]

# entropy = ["n" for _ in seeds]

# seed2entropy = dict(zip(list(map(str, seeds)), entropy))

rule all:
    input: expand("saved_model/{seed}_standard/fingerprint.pb", seed=reg_seeds),
           expand("saved_model/{seed}_spectrum/fingerprint.pb", seed=spectrum_seeds),


rule train_standard:
    input:
        py_script = "pg_gan.py",
    output: "saved_model/{seed}_standard/fingerprint.pb"
    # params: entropy = lambda wcs: seed2entropy[wcs.seed]
    shell:
        """
        python {input.py_script} -disc {wildcards.seed}_standard \
                                 -params N1,N2,N3,N_anc,T1,T2,mig \
                                 -seed {wildcards.seed} \
        """


rule train_spectrum:
    input:
        py_script = "pg_gan.py",
    output: "saved_model/{seed}_spectrum/fingerprint.pb"
    # params: entropy = lambda wcs: seed2entropy[wcs.seed]
    shell:
        """
        python {input.py_script} -disc {wildcards.seed}_spectrum \
                                 -params N1,N2,N3,N_anc,T1,T2,mig \
                                 -seed {wildcards.seed} \
                                 -use_full_spectrum
        """
