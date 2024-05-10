import matplotlib.pyplot as plt
import os
import torch as th

def read_files_in_folder(folder_path):
    # 确保提供的路径存在
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    x = []
    y = []
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # print(f"Reading {file_path}...")
            tmp = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if " INFO [main] (StandaloneAgent." in line and "cluster size" in line:
                        time = line[52:60]
                        time = time.split(':')
                        # print(time)
                        time = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
                        # print(time)
                        cluster_size = int(line.split(' ')[-1])
                        # print(line)
                        # if time - 48474 > 70 and cluster_size == 100:
                        #     print(file_path, time, cluster_size)
                            # assert 0
                        x.append(time)
                        y.append(cluster_size)
                        tmp.append(cluster_size)
                        
                        
            # if 196 not in tmp:
            #     print(file_path)
            #     print(tmp[-1])
                # assert 0
                        
    return x, y

# 调用函数，'test_log'是目标文件夹的名称，请根据实际路径进行调整



plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=30, weight='bold')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["hatch.color"] = 'white'
plt.rcParams['hatch.linewidth'] = 2.0
fig, ax = plt.subplots(ncols=3, figsize =(32, 10))
l = [55]
for i in range(1):
    x, y = read_files_in_folder(f"test_log_{l[i]}")
    # print(min(x))
    # assert 0
    x = th.as_tensor(x) - min(x)

    ax[i].scatter(x, y,  alpha=0.5)
    ax[i].set_title(f"#Node = 100")
    ax[i].set_xlabel('Time Stamp (s)')
    ax[i].set_ylabel('Cluster Size View')
    ax[i].grid(True)
    # Saving the plot as a PNG file
plt.savefig('chord_plot.png', format='png', bbox_inches='tight')
