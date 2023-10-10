"""
Application entry point for PG-GAN.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date 9/27/22
"""

# python imports
import datetime
import numpy as np
import sys
import tensorflow as tf
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
from typing import List, Union
import wandb
import random

from collections import Counter
# our imports
import discriminator
import global_vars
import util
import real_data_random
import param_set
import generator

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# globals for simulated annealing
NUM_ITER = 300
NUM_BATCH = 100
print("NUM_ITER", NUM_ITER)
print("BATCH_SIZE", global_vars.BATCH_SIZE)
print("NUM_BATCH", NUM_BATCH)

# globals for data
NUM_CLASSES = 2     # "real" vs "simulated"
print("NUM_SNPS", global_vars.NUM_SNPS)

print("NUM_CLASSES", NUM_CLASSES)
# print("NUM_CHANNELS", global_vars.NUM_CHANNELS)

def main():
    """Parse args and run simulated annealing"""

    args = util.parse_args()

    # set up seeds
    if args.seed != None:
        #rng = np.random.default_rng(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    generator, iterator, parameters, sample_sizes, entropy = util.process_args(args)
    disc = get_discriminator(sample_sizes)

    use_entropy = entropy == "y"

    posterior, loss_lst = simulated_annealing(
        generator,
        disc,
        iterator,
        parameters,
        args.seed,
        toy=args.toy,
        use_entropy=use_entropy,
        use_full_spectrum=args.use_full_spectrum,
    )

    if args.disc is not None:
        tf.saved_model.save(disc, "saved_model/" + args.disc)
        print("discriminator saved")


################################################################################
# SIMULATED ANNEALING
################################################################################


def simulated_annealing(
    generator: generator.Generator,
    disc: discriminator.OnePopModel,
    iterator: real_data_random.RealDataRandomIterator,
    parameters: List[param_set.Parameter],
    seed: int,
    toy: bool = False,
    use_entropy: bool = False,
    use_full_spectrum: bool = False,
):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, disc, iterator, parameters, seed, use_entropy = use_entropy, use_full_spectrum = use_full_spectrum,)

    wandb.init(
        # set the wandb project where this run will be logged
        project=f"mutator-ml-ooa",

        # track hyperparameters and run metadata
        config={
            "epochs": NUM_ITER,
            "architecture": "CNN",
            "dataset": "simulated",
            "num_snps": global_vars.NUM_SNPS,
            "batch_size": global_vars.BATCH_SIZE,
            "random_seed": seed,
            "using_entropy": use_entropy,
            "using_full_mutation_spectrum": use_full_spectrum,
        })

    # find starting point through pre-training (update generator in method)
    if not toy:
        param_current = pg_gan.disc_pretraining(800, use_full_spectrum=use_full_spectrum)
    # otherwise, if this is a "toy" example for testing, just run a single
    # round of discriminator pretraining
    else:
        pg_gan.disc_pretraining(1) # for testing purposes
        param_current = [param.start() for param in pg_gan.parameters]
        pg_gan.generator.update_params(param_current)
        print ("COMPLETED DISCRIMINATOR PRETRAINING")

    # NOTE: should really generate root dists to sample from (for measuring
    # generator loss) by randomly sampling between the min
    # and max for each nucleotide as measured empirically

    # after discriminator pre-training, figure out our Generator loss.
    # specifically, generate a bunch of fake data using whatever the current
    # parameter values are, and figure out how good the Discriminator is at
    # figuring out that it's all fake.
    loss_current = pg_gan.generator_loss(param_current)

    print("Current params, Current generator loss", param_current, loss_current)

    posterior = [param_current]
    loss_lst = [loss_current]

    # simulated-annealing iterations
    num_iter = 5 if toy else NUM_ITER

    out_df = []
    # main PG-GAN loop
    for i in range(num_iter):
        print("\nITER", i)
        print("time", datetime.datetime.now().time())
        T = temperature(i, num_iter) # reduce width of proposal over time

        # keep track of the best parameter values we encounter in this iteration
        # and the lowest Generator loss we get using those values
        best_iteration_params = None
        best_iteration_loss = float('inf')
        # currently, trying all parameters!
        # NOTE: for each parameter, we test 10 possible values and measure the
        # generator loss each time. when testing each of the 10 possible parameter
        # values, we *hold all other parameters constant at the values assigned to
        # them in the previous iteration*. we then pick the single value out of the
        # 10 * len(parameters) possible trials that minimized the generator loss. as
        # mentioned in the paper, this has the effect of only modifying a single parameter
        # in each iteration.
        for k in range(len(parameters)):
            #print (f"Testing parameter {parameters[k].name}")
            # k = random.choice(range(len(parameters))) # random param
            # try 10 iterations of parameter value selection for each param
            for _ in range(10):
                # get the parameters and corresponding values that were proposed in the previous iteration
                # (i.e., that minimized the generator loss)
                param_proposal = [v for v in param_current]
                # propose a new value for this parameter, *holding all other parameter values to
                # be the values they were set at in the previous iteration
                param_proposal[k] = parameters[k].proposal(param_current[k], T)
                # figure out the Generator loss using the modified parameter values.
                # that is, using the proposed parameter values, generate some fake
                # data (with no real data paired alongside it at all) and see how
                # good our discriminator is at telling that it's fake.
                loss_proposal = pg_gan.generator_loss(param_proposal)
                # if our Generator's loss is better than the best so far *in this iteration*, keep track of
                # both the loss and current parameter values
                if loss_proposal < best_iteration_loss:
                    print (f"For parameter {parameters[k].name}, proposed value of {param_proposal[k]}, improves loss to {loss_proposal} from {best_iteration_loss}.")
                    best_iteration_loss = loss_proposal
                    best_iteration_params = param_proposal

        # figure out whether the Generator loss in this iteration is better than the best
        # loss observed so far *in any iteration*. if it is, set the "probability that we
        # should accept this set of parameters for the Generator" to be 1.
        if best_iteration_loss <= loss_current:
            p_accept = 1
        # otherwise, set the "probability that we should accept this set of parameters for the Generator"
        # to be a float that captures the degree to which the current loss compares to the best loss. basically,
        # the worse this iteration looks compared to previous iterations, the lower this probability should be.
        else:
            p_accept = (loss_current / best_iteration_loss) * T
        # draw a random float from a uniform dist [0, 1). if the float is less than the "probability" defined
        # above, we'll accept the current set of parameters.
        rand = np.random.rand()
        accept = rand < p_accept

        # if we accept the current set of parameters, let's retrain our model!
        if accept:
            print("ACCEPTED")
            param_current = best_iteration_params
            loss_current = best_iteration_loss
            # NOTE: should this be pg_gan.generator.update_params() ?
            # update the parameters of the Generator to reflect the best set of parameters in the iteration.
            generator.update_params(param_current)
            # NOTE: should we be using the previous iteration's root distribution
            # as the root distribution for this iteration's generator loss?
            real_acc, fake_acc = pg_gan.train_sa(
                NUM_BATCH,
                iteration=i,
            )

        # if we shouldn't accept the current set of parameters for the Generator, move on to the
        # next iteration.
        else:
            print("NOT ACCEPTED")

        log_dict = {
            "Generator loss": loss_current,
            "Accuracy on real data": real_acc,
            "Accuracy on fake data": fake_acc,
            "Epoch": i,
        }

        for pi, p in enumerate(parameters):
            log_dict.update({p.name: param_current[pi]})
            log_dict.update({f"{p.name} (expected)": p.value})
        wandb.log(log_dict)
        out_df.append(log_dict)

        #print("T, p_accept, rand, s_current, loss_curr", end=" ")
        print (f"Temperature: {T}, Current params: {param_current}, Generator loss: {loss_current}, Real acc: {real_acc}, Fake acc: {fake_acc}")
        posterior.append(param_current)
        loss_lst.append(loss_current)

    out_df = pd.DataFrame(out_df)
    out_df.to_csv(f"summary.csv")

    return posterior, loss_lst

def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob."""
    return 1 - i / num_iter # start at 1, end at 0

################################################################################
# TRAINING
################################################################################

class PG_GAN:

    def __init__(self, generator, disc, iterator, parameters, seed, use_entropy, use_full_spectrum):
        """Setup the model and training framework"""

        # set up generator and discriminator
        self.generator = generator
        self.discriminator = disc
        self.iterator = iterator
        self.parameters = parameters

        # this checks and prints the model (1 is for the batch size)
        num_channels = 7 if use_full_spectrum else 2
        self.discriminator.build_graph((
            1,
            iterator.num_haplotypes,
            global_vars.NUM_SNPS,
            num_channels,
        ))
        self.discriminator.summary()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.disc_optimizer = tf.keras.optimizers.legacy.Adam()
        self.use_entropy = use_entropy
        self.use_full_spectrum = use_full_spectrum

        # NOTE: we should get a sense of the dynamic lenghts of "real" regions
        # that contain the specified number of SNPs we want. in other words,
        # grab 10,000 random regions from the real data and record the 95th percentile
        # of region lengths, so that we have a sense of how many base pairs we *should*
        # need in order to reach NUM_SNPs. this way, we don't incentivize our
        # generator to increase the mutation rate to get NUM_SNPs if the L variable is
        # smaller than it should be
        # _, real_region_lens = iterator.real_batch(1, batch_size=1_000)

        # exp_region_length = int(np.max(real_region_lens) * 1.5) # heuristic to ensure right size
        # print (f"Region length that should give us {global_vars.NUM_SNPS} SNPs is: {exp_region_length}")
        # figure out the median root distribution across these regions, use for
        # generator loss. # NOTE: we should probably generate random root distributions
        # wihtin the 95th percentile of root distributions observed across these, instead
        # of a constant mean
        # exp_root_dist = np.median(real_root_dists, axis=0)
        # print (f"Root distribution that should be average is {exp_root_dist}")

    def disc_pretraining(self, num_batches: int, use_full_spectrum: bool = False):
        """
        Pretrain the discriminator. We do this in order to give the discriminator
        a bit of a "head-start," so that it starts off the main training step having
        seen a number of real and simulated regions, and is already OK at discriminating
        between the two. The pre-training will make it harder for the generator to fool
        the discriminator in the early training epochs. (I think). This is also
        because our focus is one the discriminator at the end of the day.

        NOTE: in each epoch of the discriminator pretraining, we simply choose random
        values of each parameter. we're not updating parameters using simulated annealing.
        """
        s_best = []
        max_acc = 0
        k = 0 if num_batches > 1 else 9 # limit iterations for toy/testing
        max_k = 20 if self.use_full_spectrum else 10
        # try with several random sets at first
        while max_acc < 0.9 and k < max_k:
            # choose a completely random value for each of the tweakable parameters
            s_trial = [param.start() for param in self.parameters]
            print("trial", k+1, s_trial)
            print (dict(zip([param.name for param in self.parameters], s_trial)))
            self.generator.update_params(s_trial)
            real_acc, fake_acc = self.train_sa(num_batches, iteration = k, pretraining=True)
            # wandb.log({"Pretraining accuracy on real data": real_acc, "Pretraining accuracy on fake data": fake_acc})
            avg_acc = (real_acc + fake_acc) / 2
            if avg_acc > max_acc:
                max_acc = avg_acc
                s_best = s_trial
            k += 1

        # once pretraining is complete, update the parameter values to be
        # the "best" set of random parameters that we explored during pretraining
        self.generator.update_params(s_best)
        return s_best

    def train_sa(self, num_batches: int, iteration: int = 0, pretraining: bool = False):
        """
        Main training function. Comprises a single epoch with `num_batches`.
        """

        for epoch in tqdm.tqdm(range(num_batches)):
            # sample a batch of real regions

            real_regions, real_root_dists = self.iterator.real_batch()

            # perform a training step. in a single training step,
            # we use the Generator to generate a set of corresponding
            # fake regions. we then ask the Discriminator to predict the
            # class labels for the real regions and the fake regions.

            # NOTE: should we simulate regions using the same distribution
            # of region lengths as in the real data? esp. if we're using the
            # same root distributions? maybe not! need to ensure we have
            # enough bp to count suff. number of mutations, so use max
            # length but keep true root dists. but, if we make the region length
            # too large, we'll always get enough mutations in the region, even if
            # the population parameters are not sufficient.

            outname = f"imgs/{iteration}.png" if epoch == 1 else None
            iter_epoch = iteration * 8


            # NOTE: the real question is whether we should divide the inter-SNP distances
            # by a constant, or by the actuall region length. i think we need to divide by
            # a constant, so that the inter-SNP distances are just normalized versions of
            # the actualy base pair distances.
            real_acc, fake_acc, disc_loss = self.train_step(
                real_regions,
                real_root_dists,
                outname=outname,
            )


            # every 100th epoch, print the accuracy
            if (epoch+1) % 100 == 0:
                step = (epoch + 1) / 100
                template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
                print(
                    template.format(
                        epoch + 1,
                        disc_loss,
                        real_acc / global_vars.BATCH_SIZE * 100,
                        fake_acc / global_vars.BATCH_SIZE * 100,
                    ))
                if pretraining: wandb.log({"Pretraining epoch": step + iter_epoch, "Pretraining accuracy on real data": real_acc / global_vars.BATCH_SIZE,
                                           "Pretraining accuracy on fake data": fake_acc / global_vars.BATCH_SIZE, })

        return (
            real_acc / global_vars.BATCH_SIZE,
            fake_acc / global_vars.BATCH_SIZE,
        )

    def generator_loss(
        self,
        proposed_params: List[Union[int, float]],
    ):
        """ Generator loss """
        # NOTE: how to parameterize the root distribution for the generator loss?
        # this is a tricky one, since a true test should use the root distribution of
        # a "real" region. perhaps we can use the average root distribution
        # of the "real" genome across chromosomes? in order for this to be a fiar comparison
        # (i.e., for the generator to have a chance at producing a region that is "good enough"
        # to fool the discriminator), we need to give it a root distribution to simualte from
        # that is close to a real region. to this end, we actually use the prevoius iteration's
        # distribution of root distributions gathered from the real data.
        root_dists = np.tile(np.array([0.25] * 4), (global_vars.BATCH_SIZE, 1))
        generated_regions = self.generator.simulate_batch(root_dists,
            params=proposed_params,
        )
        # not training when we use the discriminator here
        fake_output = self.discriminator(generated_regions, training=False)
        loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        return loss.numpy()

    def train_step(self, real_regions, real_root_dists, outname = None):
        """One mini-batch for the discriminator"""

        with tf.GradientTape() as disc_tape:
            # use current Generator params to create a set of fake
            # regions that corresopnd to the `real_regions` input
            generated_regions = self.generator.simulate_batch(real_root_dists)

            # if outname is not None:
            #     to_plot = np.arange(4)
            #     f, axarr = plt.subplots(2, 8, figsize=(24, 2 * 4))
            #     for channel_i in np.arange(2):
            #         for idx, plot_i in enumerate(to_plot):
            #             sns.heatmap(real_regions[plot_i][:, :, channel_i], ax=axarr[channel_i, idx * 2], cbar=True)
            #             sns.heatmap(generated_regions[plot_i][:, :, channel_i], ax=axarr[channel_i, (idx * 2) + 1], cbar=True)
            #     for idx in range(4):
            #         axarr[0, idx * 2].set_title("R")
            #         axarr[0, (idx * 2) + 1].set_title("G")
            #     f.tight_layout()
            #     f.savefig(outname)

            # predict class labels (fake or real) for the real regions
            real_output = self.discriminator(real_regions, training=True)
            # do the same for the fake/generated regions
            fake_output = self.discriminator(generated_regions, training=True)
            # measure the discriminator "loss," as well as separate measures of
            # accuracy predicting labels for the real and fake regions
            disc_loss, real_acc, fake_acc = self.discriminator_loss(
                real_output, fake_output)

        # gradient descent
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
            self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator,
            self.discriminator.trainable_variables))

        return real_acc, fake_acc, disc_loss


    def discriminator_loss(self, real_output, fake_output):
        """ Discriminator loss """
        # measure accuracy of Discriminator's class label predictions
        # on both the real and fake data
        real_acc = np.sum(real_output >= 0) # positive logit => pred 1
        fake_acc = np.sum(fake_output <  0) # negative logit => pred 0

        # use binary cross-entropy to measure Discriminator loss on both the
        # real and fake data
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        # NOTE: we want to penalize our discriminator if it always
        # picks the same class label for every input (which would achieve
        # 50% accuracy, since 50% of the data in each batch is real/fake).
        # so, we can calculate the entropy of the predicted class probabilities
        # on the real and fake datasets, and we subtract that from the Discriminator loss.
        # more variable predictions (i.e., not all 1s) should have higher entropy, so
        # they should "reward" the Discriminator by lowering its loss.
        real_entropy = scipy.stats.entropy(tf.nn.sigmoid(real_output))
        fake_entropy = scipy.stats.entropy(tf.nn.sigmoid(fake_output))
        # TODO: experiment with constant (scalar) by which we multiply
        # the total entropy. higher scalars will reward the Discriminator for
        # more variable guesses, but not sure how high we want to go.
        entropy = tf.math.scalar_mul(
            0.001 / 2,
            tf.math.add(real_entropy, fake_entropy),
        )
        if self.use_entropy:
            total_loss -= entropy

        return total_loss, real_acc, fake_acc


################################################################################
# EXTRA UTILITIES
################################################################################

def get_discriminator(sample_sizes):
    # for now, only considering one-populatoin models
    assert len(sample_sizes) == 2
    return discriminator.TwoPopModel(sample_sizes[0], sample_sizes[1])

if __name__ == "__main__":
    main()
