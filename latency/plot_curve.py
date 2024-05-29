import matplotlib.pyplot as plt

file_name = 'N_100_bs_32.txt'
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
fig, ax = plt.subplots(nrows=2, figsize=(10, 6))

# Plotting the training loss curve
ax[0].plot(train_x, train_loss, label='Training Loss', marker='o', ms=1, linewidth=1)


# Plotting the test diameter curve
ax[1].plot(test_x, test_diameter, label='Test Diameter', marker='o', ms=1, linewidth=1)


# Adding title and labels
ax[0].set_title('Training Loss vs #Epoch')
ax[1].set_title('Test Diameter vs #Epoch')
ax[0].set_xlabel('#Epoch')
ax[1].set_xlabel('#Epoch')
ax[0].set_ylabel('Training Loss')
ax[1].set_ylabel('Test Diameter')
ax[0].grid(True)
ax[1].grid(True)
ax[0].set_xlim(0, 1e4)
ax[1].set_xlim(0, 1e4)
# ax[0].set_ylim(-1, 1050)
ax[1].set_ylim(7, 26)
ax[1].set_yticks([7, 10, 15, 20, 25])
fig.subplots_adjust(hspace=0.5, wspace = 0)
plt.legend()

# Show the plot
fig.savefig('training_curve.pdf', bbox_inches='tight')