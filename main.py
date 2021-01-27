'''Train CIFAR with PyTorch.'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import *
from model import *
import argparse
from torch.autograd import Variable
import subprocess
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def save_logs(logdir, savedir, losses, accs, test_losses, test_accs, net, epoch):
    # this is a totally stupid thing to have to do to be honest, just because the edinburgh computers only run for 8 hours (! WHAT!?)
    train_loss_name = logdir+"/losses.npy"
    train_accs_name = logdir + "/accs.npy"
    test_loss_name = logdir+"/test_losses.npy"
    test_accs_name = logdir+"/test_accs.npy"
    if os.path.isfile(savedir + "/losses.npy"):
        l = np.load(savedir + "/losses.npy")
        l = np.concatenate((l, np.array(losses)))
    else:
        l = np.array(losses)
    np.save(train_loss_name, l)

    if os.path.isfile(savedir + "/accs.npy"):
        l = np.load(savedir + "/accs.npy")
        l = np.concatenate((l, np.array(accs)))
    else:
        l = np.array(accs)
    np.save(train_accs_name, l)

    if os.path.isfile(savedir + "/test_losses.npy"):
        l = np.load(savedir + "/test_losses.npy")
        l = np.concatenate((l, np.array(test_losses)))
    else:
        l = np.array(test_losses)
    np.save(test_loss_name, l)

    if os.path.isfile(savedir + "/test_accs.npy"):
        l = np.load(savedir + "/test_accs.npy")
        l = np.concatenate((l, np.array(test_accs)))
    else:
        l = np.array(test_accs)
    np.save(test_accs_name, l)
    net.save_model(logdir +"/model_checkpoint", epoch)
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])

def train_prednet(logdir,savedir,model='PredNetTied',dataset="cifar10", cls=6, gpunum=4, lr=0.01,num_blocks=3,use_cuda = False,use_rate_params=True,num_epochs = 50):
    #use_cuda = torch.cuda.is_available() # choose to use gpu if possible
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batchsize = 128 #batch size
    root = './'
    rep = 1 #intial repitition is 1
    
    models = {'PredNet': PredNet, 'PredNetTied':PredNetTied}
    modelname = model+'_'+str(lr)+'LR_'+str(cls)+'CLS_'+str(rep)+'REP'

    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    if dataset == "cifar100":
      trainset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True, transform=transform_train)
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
      testset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False, download=True, transform=transform_test)
      testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset == "cifar10":
      trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transform_train)
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
      testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=transform_test)
      testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    else:
      raise ValueError("dataset name: " + str(dataset) + " is not supported. Try cifar10 or cifar100")
      
    # Define objective function
    criterion = nn.CrossEntropyLoss()

    # Model
    print('==> Building model..')
    net = models[model](num_classes=100,cls=cls,num_blocks = num_blocks,use_rate_params = use_rate_params)
    # see if there is a checkpoint
    checkpoint_name = savedir + "/model_checkpoint"
    if os.path.isfile(checkpoint_name):
        print("MODEL FOUND")
        checkpoint = torch.load(checkpoint_name)
        current_epoch = checkpoint["epoch"]
        if current_epoch >= num_epochs:
            return # we are done!
        state_dict = checkpoint["state_dict"]
        net.load_state_dict(state_dict)
    # try to load the model
    else: 
        print("NO CHECKPOINT EXISTS")
        current_epoch = 0
        
    if use_cuda:
      net = net.cuda()
       
    #set up optimizer
    if model=='PredNetTied':
        convparas = [p for p in net.conv.parameters()]+\
                    [p for p in net.linear.parameters()]
    else:
        convparas = [p for p in net.FFconv.parameters()]+\
                    [p for p in net.linear.parameters()]
        if cls > 0:
            convparas += [p for p in net.FBconv.parameters()]

      
    if use_rate_params:
      rateparas = [p for p in net.a0.parameters()]+\
                  [p for p in net.b0.parameters()]
      optimizer = optim.SGD([
                  {'params': convparas},
                  {'params': rateparas, 'weight_decay': 0},
                  ], lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
      optimizer = optim.SGD([
                  {'params': convparas},
                  ], lr=lr, momentum=0.9, weight_decay=5e-4)
      

    # Parallel computing using mutiple gpu
    if use_cuda and gpunum>1:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(gpunum))
        cudnn.benchmark = True


   # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        training_setting = 'batchsize=%d | epoch=%d | lr=%.1e ' % (batchsize, epoch, optimizer.param_groups[0]['lr'])
        #statfile.write('\nTraining Setting: '+training_setting+'\n')
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                #print("In use cuda")
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets) # why do we need to make these variable?
            #print("made variables")
            outputs = net(inputs)
            #print("Computed output")
            loss = criterion(outputs, targets)
            #print("computed loss")
            loss.backward()
            #print("backpropped")
            optimizer.step()
            #print("updated params")
            if batch_idx % 10 == 0:
              print("Up to batch, ", batch_idx)
    
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum() # this all makes sense and is fine. Above all we need to understand the MODEL FILE!
            
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #% (train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
        #writing training record 
        statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best acc: %.3f' \
                  % (epoch, train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)  
        #statfile.write(statstr+'\n')  
        print(statstr) 
        return train_loss / (batch_idx +1), 100.*(float)(correct)/(float)(total)
    
    
    # Testing
    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                #print("In use cuda")
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            #print("make variables")
            outputs = net(inputs)
            #print("Computed output")
            loss = criterion(outputs, targets)
            #print("Computed loss")
            if batch_idx % 10 == 0:
              print("Up to batch, ", batch_idx)
    
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        
        acc = 100.*correct/total
        return test_loss / (batch_idx + 1), acc
        
    # Set adaptive learning rates
    def decrease_learning_rate():
        """Decay the previous learning rate by 10"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    # initialize arrays of loss and accuracies
    #train_losses = []
    #train_accs = []
    #test_losses = []
    #test_accs = []
      
    #train network
    for epoch in range(current_epoch, current_epoch + num_epochs):
        if epoch==80 or epoch==140 or epoch==200:
            decrease_learning_rate()       
        train_loss, train_acc = train(epoch)
        #test_loss, test_acc = test(epoch)
        #train_losses.append(train_loss)
        #train_accs.append(train_acc)
        #test_losses.append(test_loss)
        #test_accs.append(test_acc)
        print("SAVING LOGS!")
        save_logs(logdir, savedir, [train_loss], [train_acc], [test_loss], [test_acc], net, epoch)
    print("DONE!!!")


def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument('--cls', default=0, type=int, help='number of cycles')
    parser.add_argument('--model', default='PredNet', help= 'models to train')
    #parser.add_argument('--gpunum', default=2, type=int, help='number of gpu used to train the model')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument("--num_blocks",default=1,type=int, help="depth in blocks of the network")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset to use")
    parser.add_argument("--use_rate_params", default="False", type=boolcheck, help="Learn a_0, b_0 params via gradient descent?")
    parser.add_argument("--num_epochs", default=50, type=int, help="number of epochs to run for")
    args = parser.parse_args()

    if args.savedir != "":
            subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    use_cuda = torch.cuda.is_available()
    gpunum = 0
    if use_cuda:
        # check GPU has enough memory to actually run the thing
        MEM_LIMIT = 10
        total_mem = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1e9
        print("TOTAL MEM: ", total_mem)
        if total_mem < MEM_LIMIT:
            gpunum = 0
            use_cuda = False
        else:
            gpunum = torch.cuda.device_count()

    print("USING CUDA? ", use_cuda)
    print("GPU NUM: ", gpunum)
    torch.cuda.empty_cache()


    train_prednet(logdir = args.logdir, savedir = args.savedir,model=args.model, cls=args.cls, gpunum=gpunum, num_blocks= args.num_blocks,lr=args.lr,dataset=args.dataset,use_cuda = use_cuda, num_epochs = args.num_epochs)