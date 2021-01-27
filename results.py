import numpy as np
import sys
import os
import matplotlib.pyplot as plt

### preliminary results plotting

block_num_list = [1,2,3,4,5,6]
step_list = [0,1,2,3,4,5,6]
seeds = 1
# let's just test max first
max_losses = np.zeros((len(block_num_list), len(step_list)))
max_accs = np.zeros((len(block_num_list), len(step_list)))
max_test_accs = np.zeros((len(block_num_list), len(step_list)))
for b_n, block_num in enumerate(block_num_list):
    for s_n, step in enumerate(step_list):
        for s in range(seeds):
            spath = "prelim_experiments" +"_block_" + str(block_num) + "_steps_" + str(step) + "/" + str(s)
            try:
                print("NUM BLOCKS: ", b_n)
                print("NUM ITERS: ", s_n)
                train_loss = np.load(spath + "/losses.npy")
                train_accs = np.load(spath + "/accs.npy")
                test_accs = np.load(spath+"/test_accs.npy")
                plt.plot(train_accs)
                plt.title("Num blocks " + str(b_n) + " num iters " + str(s_n))
                plt.show()
                max_losses[b_n, s_n] = np.max(train_loss)
                max_accs[b_n, s_n] = np.max(train_accs)
                max_test_accs[b_n, s_n] = np.max(test_accs)
            except:
                print("Something failed on file : ", spath)
                pass


for b_n in range(len(block_num_list)):
    plt.plot(max_accs[b_n, :])
    plt.show()

for s_n in range(len(step_list)):
    plt.plot(max_accs[:, s_n])
    plt.show()


