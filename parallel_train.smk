import numpy as np 

N_TRIALS = 5

seeds = [np.random.randint(1, 2**32) for _ in range(N_TRIALS)]

rule all:
    input: expand("saved_model/{seed}/fingerprint.pb", seed=seeds)


rule train:
    input:
        py_script = "pg_gan.py",
        data = "data/simulated/simulated.h5",
        ref = "data/simulated/simulated.fa",
    output: "saved_model/{seed}/fingerprint.pb"
    shell:
        """
        python {input.py_script} --data {input.data} \
                                 --ref {input.ref} \
                                 --disc {wildcards.seed} \
                                 -params growth,N1,N2,T1,T2 \
                                 -seed {wildcards.seed} \
        """
