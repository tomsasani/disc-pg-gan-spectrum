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

# our imports
import discriminator
import global_vars
import util
from real_data_random import Region

# globals for simulated annealing
NUM_ITER = 300
NUM_BATCH = 100
print("NUM_ITER", NUM_ITER)
print("BATCH_SIZE", global_vars.BATCH_SIZE)
print("NUM_BATCH", NUM_BATCH)

# globals for data
NUM_CLASSES = 2     # "real" vs "simulated"
NUM_CHANNELS = 6    # counts of derived alleles corresponding to each mutation type
print("NUM_SNPS", global_vars.NUM_SNPS)

print("L", global_vars.L)
print("NUM_CLASSES", NUM_CLASSES)
print("NUM_CHANNELS", NUM_CHANNELS)

def main():
    """Parse args and run simulated annealing"""

    args = util.parse_args()

    # set up seeds
    if args.seed != None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    generator, iterator, parameters, sample_sizes = util.process_args(args)
    disc = get_discriminator(sample_sizes)

    posterior, loss_lst = simulated_annealing(
        generator,
        disc,
        iterator,
        parameters,
        args.seed,
        toy=args.toy,
    )

    if args.disc is not None:
        tf.saved_model.save(disc, "saved_model/" + args.disc)
        print("discriminator saved")

    print(posterior)
    print(loss_lst)

################################################################################
# SIMULATED ANNEALING
################################################################################

def simulated_annealing(generator, disc, iterator, parameters, seed,
    toy=False):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, disc, iterator, parameters, seed)

    # find starting point through pre-training (update generator in method)
    if not toy:
        s_current = pg_gan.disc_pretraining(800)
    # otherwise, if this is a "toy" example for testing, just run a single
    # round of discriminator pretraining
    else:
        pg_gan.disc_pretraining(1) # for testing purposes
        s_current = [param.start() for param in pg_gan.parameters]
        pg_gan.generator.update_params(s_current)
        print ("COMPLETED DISCRIMINATOR PRETRAINING")

    # after discriminator pre-training, figure out our Generator loss.
    # specifically, generate a bunch of fake data using whatever the current
    # parameter values are, and figure out how good the Discriminator is at
    # figuring out that it's all fake.
    loss_curr = pg_gan.generator_loss(s_current)
    print("params, loss", s_current, loss_curr)

    posterior = [s_current]
    loss_lst = [loss_curr]

    # simulated-annealing iterations
    num_iter = 5 if toy else NUM_ITER

    out_df = []
    # main PG-GAN loop
    for i in range(num_iter):
        print("\nITER", i)
        print("time", datetime.datetime.now().time())
        T = temperature(i, num_iter) # reduce width of proposal over time
        # propose 10 updates per param and pick the best one
        s_best = None
        loss_best = float('inf')
        # currently, trying all parameters!
        for k in range(len(parameters)):
            # try 10 iterations of parameter value selection for each param
            for _ in range(10):
                s_proposal = [v for v in s_current]
                s_proposal[k] = parameters[k].proposal(s_current[k], T)
                # for each parameter value, figure out the Generator loss.
                # that is, using the proposed parameter values, generate some fake
                # data (with no real data paired alongside it at all) and see how 
                # good our discriminator is at telling that it's fake.
                loss_proposal = pg_gan.generator_loss(s_proposal)
                # if our Generator's loss is better than the best so far *in this iteration*, use these
                # new parameter values as the parameters for retraining
                if loss_proposal < loss_best: # minimizing loss
                    loss_best = loss_proposal
                    s_best = s_proposal

        # figure out whether the Generator loss in this iteration is better than the best
        # loss observed so far *in any iteration*. if it is, set the "probability that we
        # should accept this set of parameters for the Generator" to be 1.
        if loss_best <= loss_curr: 
            p_accept = 1
        # otherwise, set the "probability that we should accept this set of parameters for the Generator"
        # to be a float that captures the degree to which the current loss compares to the best loss. basically,
        # the worse this iteration looks compared to previous iterations, the lower this probability should be.
        else:
            p_accept = (loss_curr / loss_best) * T
        # draw a random float from a uniform dist [0, 1). if the float is less than the "probability" defined
        # above, we'll accept the current set of parameters.
        rand = np.random.rand()
        accept = rand < p_accept

        real_acc, fake_acc = 0, 0

        # if we accept the current set of parameters, let's retrain our model!
        if accept:
            print("ACCEPTED")
            s_current = s_best
            # NOTE: should this be pg_gan.generator.update_params() ?
            # update the parameters of the Generator to reflect the best set of parameters in the iteration.
            generator.update_params(s_current)
            real_acc, fake_acc = pg_gan.train_sa(NUM_BATCH)
            loss_curr = loss_best

        # if we shouldn't accept the current set of parameters for the Generator, move on to the
        # next iteration.
        else:
            print("NOT ACCEPTED")

        for pi, p in enumerate(parameters):
            out_df.append({"epoch": i, "param": p.name, "generator_loss": loss_best, "param_value": s_best[pi], "Real acc": real_acc, "Fake acc": fake_acc})

        print("T, p_accept, rand, s_current, loss_curr", end=" ")
        print (f"Temperature: {T}, Generator loss: {loss_curr}, Real acc: {real_acc}, Fake acc: {fake_acc}")
        print(T, p_accept, rand, s_current, loss_curr)
        posterior.append(s_current)
        loss_lst.append(loss_curr)

    out_df = pd.DataFrame(out_df)
    out_df.to_csv("summary.csv")
    g = sns.FacetGrid(data=out_df, row="param", sharey=False, sharex=True, height=3, aspect=4)
    g.map(sns.lineplot, "epoch", "param_value")
    g.savefig("params.png", dpi=200)

    return posterior, loss_lst

def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob."""
    return 1 - i/num_iter # start at 1, end at 0

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

        # this checks and prints the model (1 is for the batch size)
        self.discriminator.build_graph((
            1,
            iterator.num_haplotypes,
            global_vars.NUM_SNPS,
            NUM_CHANNELS,
        ))
        self.discriminator.summary()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.disc_optimizer = tf.keras.optimizers.legacy.Adam()

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

        while max_acc < 0.9 and k < 10:
            # choose a completely random value for each of the tweakable parameters
            s_trial = [param.start() for param in self.parameters]
            print("trial", k+1, s_trial)
            self.generator.update_params(s_trial)
            real_acc, fake_acc = self.train_sa(num_batches)
            avg_acc = (real_acc + fake_acc)/2
            if avg_acc > max_acc:
                max_acc = avg_acc
                s_best = s_trial
            k += 1

        # once pretraining is complete, update the parameter values to be
        # the "best" set of random parameters that we explored during pretraining
        self.generator.update_params(s_best)
        return s_best

    def train_sa(self, num_batches: int):
        """
        Main training function. Comprises a single epoch with `num_batches`.
        """

        for epoch in tqdm.tqdm(range(num_batches)):
            # sample a batch of real regions of the specified region_len
            real_regions, real_root_dists = self.iterator.real_batch(
                neg1=True,
                #region_len=global_vars.L,
            )
            # perform a training step. in a single training step,
            # we use the Generator to generate a set of corresponding
            # fake regions. we then ask the Discriminator to predict the
            # class labels for the real regions and the fake regions.
            real_acc, fake_acc, disc_loss = self.train_step(real_regions, real_root_dists)

            # every 100th epoch, print the accuracy
            if (epoch+1) % 100 == 0:
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
        )

    def generator_loss(self, proposed_params):
        """ Generator loss """
        # NOTE: we provide a static root distribution of nucleotides when measuring
        # the generator loss here.
        #root_dists = np.array([0.25, 0.25, 0.25, 0.25])
        root_dists = self.iterator.base_root_dist
        root_dists_tiled = np.tile(root_dists, (global_vars.BATCH_SIZE, 1),)
        generated_regions = self.generator.simulate_batch(root_dists_tiled, params=proposed_params)
        # not training when we use the discriminator here
        fake_output = self.discriminator(generated_regions, training=False)
        loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        return loss.numpy()

    def train_step(self, real_regions, real_root_dists):
        """One mini-batch for the discriminator"""

        with tf.GradientTape() as disc_tape:
            # use current Generator params to create a set of fake
            # regions that corresopnd to the `real_regions` input
            generated_regions = self.generator.simulate_batch(real_root_dists)
            # f, axarr = plt.subplots(6, 2, figsize=(12, 10))
            # for channel_i in np.arange(6):
            #     sns.heatmap(real_regions[0][:, :, channel_i], ax=axarr[channel_i, 0])
            #     sns.heatmap(generated_regions[0][:, :, channel_i], ax=axarr[channel_i, 1])
            # f.tight_layout()
            # f.savefig("regions.png")
            # predict class labels (fake or real) for the real regions
            real_output = self.discriminator(real_regions, training=True)
            # do the same for the fake/generated regions
            fake_output = self.discriminator(generated_regions, training=True)
            #print (f"Predicted classes on real regions: {np.sum(real_output >= 0)}\nPredicted classes on generated regions: {np.sum(fake_output > 0)}")
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

        # add on entropy regularization (small penalty)
        real_entropy = scipy.stats.entropy(tf.nn.sigmoid(real_output))
        fake_entropy = scipy.stats.entropy(tf.nn.sigmoid(fake_output))
        entropy = tf.math.scalar_mul(0.001/2, tf.math.add(real_entropy,
            fake_entropy)) # can I just use +,*? TODO experiement with constant

        return total_loss + entropy, real_acc, fake_acc


################################################################################
# EXTRA UTILITIES
################################################################################

def get_discriminator(sample_sizes):
    # for now, only considering one-populatoin models
    assert len(sample_sizes) == 1
    return discriminator.OnePopModel(sample_sizes[0])

if __name__ == "__main__":
    main()
