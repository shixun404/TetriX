import re
import os
import matplotlib.pyplot as plt

# Function to parse log file and extract epoch and min test diameter
def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    epoch_list = []
    min_diameter_list = []

    for line in lines:
        match_epoch = re.search(r"Train Epoch (\d+)", line)
        match_min_diameter = re.search(r"Min Test Diameter\[(\d+\.?\d*)\]", line)

        if match_epoch and match_min_diameter:
            epoch = int(match_epoch.group(1))
            min_diameter = float(match_min_diameter.group(1))
            epoch_list.append(epoch)
            min_diameter_list.append(min_diameter)

    return epoch_list, min_diameter_list

# Parse log from the provided path
log_path_1 = "/pscratch/sd/s/swu264/SWARM/model/20250609_153331/output"
log_path_2 = "/pscratch/sd/s/swu264/SWARM/model/20250609_153334/output"
log_path_4 = "/pscratch/sd/s/swu264/SWARM/model/20250609_153410/output"
log_path_8 = "/pscratch/sd/s/swu264/SWARM/model/20250609_153420/output"

# Dummy placeholders (user will upload files)
log_data = {
    'source=1': ([], []),
    'source=2': ([], []),
    'source=4': ([], []),
    'source=8': ([], [])
}

# We will now wait for user to upload the actual log files
log_data['source=1'] = parse_log_file(log_path_1)
log_data['source=2'] = parse_log_file(log_path_2)
log_data['source=4'] = parse_log_file(log_path_4)
log_data['source=8'] = parse_log_file(log_path_8)

# Plotting
plt.figure(figsize=(10, 6))
for label, (epochs, diameters) in log_data.items():
    plt.plot(epochs, diameters, label=label)

plt.xlabel("Train Epoch")
plt.ylabel("Min Test Diameter")
plt.title("Train Epoch vs. Min Test Diameter")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig('train_epoch_vs_min_test_diameter.png')
