
import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 1
block_num = 2
steps = 1
block_num_list = [1,2,3,4,5,6]
step_list = [0,1,2,3,4,5,6]
for block_num in block_num_list:
    for step in step_list:
        for s in range(seeds):
            lpath = log_path + "/"+str(exp_name) +"_block_" + str(block_num) + "_steps_" + str(step) + "/" + str(s)
            spath = save_path + "/"+str(exp_name) +"_block_" + str(block_num) + "_steps_" + str(step) + "/" + str(s)
            final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --num_blocks " + str(block_num) + " --cls " + str(step)
            print(final_call)
            print(final_call, file=output_file)