'''Train CIFAR with PyTorch.'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
import subprocess
import sys

def train_prednet(model='PredNetTied',dataset="cifar10", cls=6, gpunum=4, lr=0.01,num_blocks=3, dataset="cifar10"):
    use_cuda = torch.cuda.is_available() # choose to use gpu if possible
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batchsize = 128 #batch size
    root = './'
    rep = 1 #intial repitition is 1
    
    models = {'PredNet': PredNet, 'PredNetTied':PredNetTied}
    modelname = model+'_'+str(lr)+'LR_'+str(cls)+'CLS_'+str(rep)+'REP'
    
    # clearn folder
    checkpointpath = root+'checkpoint/'
    logpath = root+'log/'

    if not os.path.isdir(checkpointpath):
        os.mkdir(checkpointpath)
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    while(os.path.isfile(checkpointpath + modelname + '_last_ckpt.t7')): 
        rep += 1
        modelname = model+'_'+str(lr)+'LR_'+str(cls)+'CLS_'+str(rep)+'REP'
        
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
      trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
      testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
      testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset == "cifar10":
      trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
      testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
      testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    else:
      raise ValueError("dataset name: " + str(dataset) + " is not supported. Try cifar10 or cifar100")
      
    # Define objective function
    criterion = nn.CrossEntropyLoss()

    # Model
    print('==> Building model..')
    net = models[model](num_classes=100,cls=cls)
    if use_cuda:
      net = net.cuda()
       
    #set up optimizer
    if model=='PredNetTied':
        convparas = [p for p in net.conv.parameters()]+\
                    [p for p in net.linear.parameters()]
    else:
        convparas = [p for p in net.FFconv.parameters()]+\
                    [p for p in net.FBconv.parameters()]+\
                    [p for p in net.linear.parameters()]

    rateparas = [p for p in net.a0.parameters()]+\
                [p for p in net.b0.parameters()]
    optimizer = optim.SGD([
                {'params': convparas},
                {'params': rateparas, 'weight_decay': 0},
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
        statfile.write('\nTraining Setting: '+training_setting+'\n')
        
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
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
        #writing training record 
        statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best acc: %.3f' \
                  % (epoch, train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)  
        statfile.write(statstr+'\n')  
        print(statstr) 
    
    
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
    
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
        statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best_acc: %.3f' \
                  % (epoch, test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)
        statfile.write(statstr+'\n')
        print(statstr)
        
        # Save checkpoint.
        acc = 100.*correct/total
        print("Accuracy: ", acc)
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,           
        }

        torch.save(state, checkpointpath + modelname + '_last_ckpt.t7')

        #check if current accuarcy is the best
        if acc >= best_acc:  
            print('Saving..')
            torch.save(state, checkpointpath + modelname + '_best_ckpt.t7')
            best_acc = acc
        
    # Set adaptive learning rates
    def decrease_learning_rate():
        """Decay the previous learning rate by 10"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    #train network
    for epoch in range(start_epoch, start_epoch+250):
        statfile = open(logpath+'training_stats_'+modelname+'.txt', 'a+')  #open file for writing
        if epoch==80 or epoch==140 or epoch==200:
            decrease_learning_rate()       
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--cls', default=6, type=int, help='number of cycles')
    parser.add_argument('--model', default='PredNet', help= 'models to train')
    parser.add_argument('--gpunum', default=2, type=int, help='number of gpu used to train the model')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument("--num_blocks",default=3,type=int, help="depth in blocks of the network")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset to use")
    args = parser.parse_args()

    train_prednet(model=args.model, cls=args.cls, gpunum=args.gpunum, num_blocsk= args.num_blocsk,lr=args.lr,dataset=dataset)