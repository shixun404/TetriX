import matplotlib.pyplot as plt

file_name = 'logs/N_100_bs_32.txt'
train_loss = []
test_diameter = []
tmp_test = []
epoch = 0
test_x = []
train_x = []
with open(file_name, 'r') as f:
    for line in f:
        if 'Train Epoch' in line:
            epoch += 1
            if epoch % 10 == 0:
                train_loss.append( float(line.split('=')[-1]))
            
                train_x.append(epoch)
        if 'Test Graph' in line:
            tmp_test.append(float(line.split('=')[-1]))
        if len(tmp_test) == 10:
            test_diameter.append(sum(tmp_test) / 10)
            test_x.append(epoch)
            tmp_test = []

        if len(test_x) > 100:
            break


# Creating the plot
fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

# Plotting the training loss curve
ax[0].plot(train_x[:6000], train_loss[:6000], label='Training Loss', marker='o', ms=1, linewidth=1)


# Plotting the test diameter curve
ax[1].plot(test_x[:6000], test_diameter[:6000], label='Test Diameter', marker='o', ms=1, linewidth=1)


# Adding title and labels
ax[0].set_title('Training Loss vs #Epoch', weight='bold')
ax[1].set_title('Test Diameter vs #Epoch', weight='bold')
ax[0].set_xlabel('#Epoch')
ax[1].set_xlabel('#Epoch')
ax[0].set_ylabel('Training Loss')
ax[1].set_ylabel('Test Diameter')
ax[0].grid(True)
ax[1].grid(True)
ax[0].set_xlim(0, 6e3)
ax[1].set_xlim(0, 6e3)
# ax[0].set_ylim(-1, 1050)
ax[1].set_ylim(0, 26)
ax[1].set_yticks([5, 10, 15, 20, 25])
fig.subplots_adjust( wspace = 0.2)
plt.legend()

# Show the plot
fig.savefig('training_curve.pdf', bbox_inches='tight')