"""
Application entry point for PG-GAN.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date 9/27/22
"""

# python imports
import datetime
import numpy as np
import sys
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
from typing import List, Union
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx

# our imports
import gnn_discriminator
import global_vars
import util
import real_data_random
import param_set
import generator

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
        torch.manual_seed(args.seed)

    generator, iterator, parameters, sample_sizes = util.process_args(args)
    disc = get_discriminator(64)

    posterior, loss_lst = simulated_annealing(
        generator,
        disc,
        iterator,
        parameters,
        args.seed,
        toy=args.toy,
    )

    # if args.disc is not None:
    #     tf.saved_model.save(disc, "saved_model/" + args.disc)
    #     print("discriminator saved")

    #print(posterior)
    #print(loss_lst)

################################################################################
# SIMULATED ANNEALING
################################################################################


def simulated_annealing(
    generator: generator.Generator,
    disc: gnn_discriminator.GCN,
    iterator: real_data_random.RealDataRandomIterator,
    parameters: List[param_set.Parameter],
    seed: int,
    toy: bool = False,
):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, disc, iterator, parameters, seed)

    # find starting point through pre-training (update generator in method)
    if not toy:
        param_current = pg_gan.disc_pretraining(800)
    # otherwise, if this is a "toy" example for testing, just run a single
    # round of discriminator pretraining
    else:
        pg_gan.disc_pretraining(1) # for testing purposes
        param_current = [param.start() for param in pg_gan.parameters]
        pg_gan.generator.update_params(param_current)
        print ("COMPLETED DISCRIMINATOR PRETRAINING")

    prev_region_lens = np.array([pg_gan.norm_len] * global_vars.BATCH_SIZE)
    prev_root_dists = np.tile(
        pg_gan.exp_root_dist,
        (global_vars.BATCH_SIZE, 1),
    )
    # after discriminator pre-training, figure out our Generator loss.
    # specifically, generate a bunch of fake data using whatever the current
    # parameter values are, and figure out how good the Discriminator is at
    # figuring out that it's all fake.
    loss_current = pg_gan.generator_loss(param_current, prev_root_dists, prev_region_lens)
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
                loss_proposal = pg_gan.generator_loss(
                    param_proposal,
                    prev_root_dists,
                    prev_region_lens,
                )
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
        rand = np.random.uniform()
        accept = rand < p_accept

        disc_loss = -1

        # if we accept the current set of parameters, let's retrain our model!
        if accept:
            print("ACCEPTED")
            param_current = best_iteration_params
            loss_current = best_iteration_loss
            # NOTE: should this be pg_gan.generator.update_params() ?
            # update the parameters of the Generator to reflect the best set of parameters in the iteration.
            pg_gan.generator.update_params(param_current)
            # NOTE: should we be using the previous iteration's root distribution
            # as the root distribution for this iteration's generator loss?
            real_acc, fake_acc, _, disc_loss = pg_gan.train_sa(NUM_BATCH, iteration=i)

        # if we shouldn't accept the current set of parameters for the Generator, move on to the
        # next iteration.
        else:
            print("NOT ACCEPTED")

        for pi, p in enumerate(parameters):
            out_df.append({
                "epoch": i,
                "param": p.name,
                "generator_loss": best_iteration_loss,
                "discriminator_loss": disc_loss,
                "param_value": best_iteration_params[pi],
                "Real acc": real_acc,
                "Fake acc": fake_acc,
            })

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

    def __init__(self, generator, disc, iterator, parameters, seed):
        """Setup the model and training framework"""

        # set up generator and discriminator
        self.generator = generator
        self.discriminator = disc
        self.iterator = iterator
        self.parameters = parameters

        self.cross_entropy = torch.nn.BCEWithLogitsLoss()
        self.disc_optimizer = torch.optim.Adam(disc.parameters())

        # NOTE: we should get a sense of the dynamic lenghts of "real" regions
        # that contain the specified number of SNPs we want. in other words,
        # grab 10,000 random regions from the real data and record the 95th percentile
        # of region lengths, so that we have a sense of how many base pairs we *should*
        # need in order to reach NUM_SNPs. this way, we don't incentivize our
        # generator to increase the mutation rate to get NUM_SNPs if the L variable is
        # smaller than it should be
        _, real_root_dists, real_region_lens = iterator.real_batch(1, batch_size=10)
        exp_region_length = int(np.max(real_region_lens) * 1.5) # heuristic to ensure right size
        print (f"Region length that should give us {global_vars.NUM_SNPS} SNPs is: {exp_region_length}")
        # figure out the median root distribution across these regions, use for
        # generator loss
        exp_root_dist = np.median(real_root_dists, axis=0)
        print (f"Root distribution that should be average is {exp_root_dist}")

        self.norm_len = exp_region_length
        self.exp_root_dist = exp_root_dist

    def disc_pretraining(self, num_batches: int):
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

        # try with several random sets at first
        while max_acc < 0.9 and k < 10:
            # choose a completely random value for each of the tweakable parameters
            s_trial = [param.start() for param in self.parameters]
            #s_trial = [10_000, 1.25e-8, 1.25e-8, 5e-3]
            print("trial", k+1, s_trial)
            print (dict(zip([param.name for param in self.parameters], s_trial)))
            self.generator.update_params(s_trial)
            real_acc, fake_acc, _, disc_loss = self.train_sa(num_batches)
            avg_acc = (real_acc + fake_acc) / 2
            if avg_acc > max_acc:
                max_acc = avg_acc
                s_best = s_trial
            k += 1

        # once pretraining is complete, update the parameter values to be
        # the "best" set of random parameters that we explored during pretraining
        self.generator.update_params(s_best)
        return s_best

    def train_sa(self, num_batches: int, iteration: int = 0):
        """
        Main training function. Comprises a single epoch with `num_batches`.
        """

        for epoch in tqdm.tqdm(range(num_batches)):
            # in each epoch, let's train the discriminator
            self.discriminator.train()
            # sample a batch of real regions
            real_regions, real_root_dists, real_region_lens = self.iterator.real_batch(self.norm_len)
            # perform a training step. in a single training step,
            # we use the Generator to generate a set of corresponding
            # fake regions. we then ask the Discriminator to predict the
            # class labels for the real regions and the fake regions.
            region_lens_for_gen = np.array([self.norm_len] * global_vars.BATCH_SIZE)
            outname = f"imgs/{iteration}.png" if epoch == 1 else None

            # NOTE: should we simulate regions using the same distribution
            # of region lengths as in the real data? esp. if we're using the
            # same root distributions? maybe not! need to ensure we have
            # enough bp to count suff. number of mutations, so use max
            # length but keep true root dists
            real_acc, fake_acc, disc_loss = self.train_step(
                real_regions,
                real_root_dists,
                region_lens_for_gen,
                outname=outname,
            )

            # gradient descent
            disc_loss.backward()
            self.disc_optimizer.step()
            self.disc_optimizer.zero_grad()

            # every 100th epoch, print the accuracy
            #if (epoch+1) % 100 == 0:
            template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
            print(
                template.format(
                    epoch + 1,
                    disc_loss,
                    real_acc / global_vars.BATCH_SIZE * 100,
                    fake_acc / global_vars.BATCH_SIZE * 100,
                ))

        return (
            real_acc / global_vars.BATCH_SIZE,
            fake_acc / global_vars.BATCH_SIZE,
            real_region_lens,
            disc_loss,
        )

    def generator_loss(
        self,
        proposed_params: List[Union[int, float]],
        root_dists: np.ndarray,
        region_lens: np.ndarray,
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
        generated_regions = self.generator.simulate_batch(
            root_dists,
            region_lens,
            self.norm_len,
            params=proposed_params,
        )
        #print (seeds)
        # not training when we use the discriminator here
        fake_output = self.discriminator(generated_regions, training=False)
        loss = self.cross_entropy(fake_output, torch.ones_like(fake_output))

        return loss.detach().numpy()

    def train_step(self, real_regions, real_root_dists, real_region_lens, outname = None):
        """One mini-batch for the discriminator"""

        #with tf.GradientTape() as disc_tape:
        # use current Generator params to create a set of fake
        # regions that corresopnd to the `real_regions` input
        generated_regions = self.generator.simulate_batch(
            real_root_dists,
            real_region_lens,
            self.norm_len,
        )

        for real_data, gen_data in zip(real_regions, generated_regions):
            # predict class labels (fake or real) for the real regions
            real_output = self.discriminator(real_data.x, real_data.edge_index, real_data.batch, training=True)
            # do the same for the fake/generated regions
            fake_output = self.discriminator(gen_data.x, gen_data.edge_index, gen_data.batch, training=True)
            # measure the discriminator "loss," as well as separate measures of
            # accuracy predicting labels for the real and fake regions
            disc_loss, real_acc, fake_acc = self.discriminator_loss(real_output, fake_output)
            print (disc_loss)

        return real_acc, fake_acc, disc_loss


    def discriminator_loss(self, real_output, fake_output):
        """ Discriminator loss """
        # measure accuracy of Discriminator's class label predictions
        # on both the real and fake data
        real_acc = torch.sum(real_output >= 0) # positive logit => pred 1
        fake_acc = torch.sum(fake_output <  0) # negative logit => pred 0

        # use binary cross-entropy to measure Discriminator loss on both the
        # real and fake data. NOTE: with pytorch, expected values must come
        # second and predictions must come first in the call to BCELoss.
        real_loss = self.cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = self.cross_entropy(fake_output, torch.zeros_like(fake_output))
        total_loss = real_loss + fake_loss

        # NOTE: we want to penalize our discriminator if it always
        # picks the same class label for every input (which would achieve
        # 50% accuracy, since 50% of the data in each batch is real/fake).
        # so, we can calculate the entropy of the predicted class probabilities
        # on the real and fake datasets, and we subtract that from the Discriminator loss.
        # more variable predictions (i.e., not all 1s) should have higher entropy, so
        # they should "reward" the Discriminator by lowering its loss.
        #real_entropy = scipy.stats.entropy(torch.sigmoid(real_output))
        #fake_entropy = scipy.stats.entropy(torch.sigmoid(fake_output))
        # TODO: experiment with constant (scalar) by which we multiply
        # the total entropy. higher scalars will reward the Discriminator for
        # more variable guesses, but not sure how high we want to go.
        # entropy = tf.math.scalar_mul(
        #     0.001 / 2,
        #     tf.math.add(real_entropy, fake_entropy),
        # )

        return total_loss, real_acc, fake_acc


################################################################################
# EXTRA UTILITIES
################################################################################

def get_discriminator(hidden_channels):
    # for now, only considering one-populatoin models
    return gnn_discriminator.GCN(hidden_channels)

if __name__ == "__main__":
    main()
