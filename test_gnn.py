import torch
from gnn_discriminator import GCN
import generator
import real_data_random
import simulation
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train(real, fake):
    model.train()

    for real_g, fake_g in zip(real, fake):  # Iterate in batches over the training dataset.
        real_output = model(real_g.x, real_g.edge_index, real_g.batch)  # Perform a single forward pass.
        fake_output = model(fake_g.x, fake_g.edge_index, fake_g.batch)
        real_loss = criterion(real_output, torch.ones_like(real_output))
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        loss = real_loss + fake_loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return loss

def test(fake):
    model.eval()

    correct = 0
    for fake_g in fake:  # Iterate in batches over the training/test dataset.
        output = model(fake_g.x, fake_g.edge_index, fake_g.batch)  
        pred = torch.sum(output < 0)
        correct += pred  # Check against ground-truth labels.
    print (correct, len(fake.dataset))
    return correct / len(fake.dataset)  # Derive ratio of correct predictions.

simulator = simulation.simulate_exp

gen = generator.Generator(
        simulator,
        ["mu"],
        [100],
        42,
    )
gen.update_params([1e-9])

iterator = real_data_random.RealDataRandomIterator("data/simulated/simulated.h5", "data/simulated/simulated.fa", None, 123)

losses, accs = [], []

epochs = 50
for epoch in tqdm.tqdm(range(epochs)):
    real_regions, real_root_dists, real_region_lens = iterator.real_batch(50_000)
    generated_regions = gen.simulate_batch(real_root_dists, real_region_lens, 50_000)
    loss_ = train(real_regions, generated_regions)
    losses.append(loss_.detach().numpy())
    accuracy = test(generated_regions)
    accs.append(accuracy)
    #train_acc = test(real_regions)
    #test_acc = test(generated_regions)
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

f, (ax1, ax2) = plt.subplots(2)
ax1.plot(range(epochs), losses)
ax2.plot(range(epochs), accs)

f.savefig('loss.png')