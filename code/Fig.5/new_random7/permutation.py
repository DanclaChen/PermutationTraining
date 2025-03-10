import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter

import numpy as np

import os

import random


from para_init import setup_seed, weights_init, bias_init, coefficient_init_random


class NN(nn.Module):
    def __init__(self, layers):
        super(NN, self).__init__()

        self.layer1 = nn.Linear(in_features=layers[0], out_features=layers[1])
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=layers[1], out_features=layers[2], bias=False)
        self.layer3 = nn.Linear(in_features=layers[2], out_features=layers[3], bias=True)

    def forward(self, x):
        unacti = self.layer1(x)
        hidden1 = self.relu(unacti)
        hidden2 = self.layer2(hidden1)
        logits = self.layer3(hidden2)

        return logits


def train_by_permu_adjust(dataloader, model, loss_fn, optimizer, epoch, epoch_0, k_num, device, params, scheduler, writer, weight_0):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Move the data to the computational device
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # laerning rate decay exponienally
    if params.Is_lr_decay == True:
        scheduler.step()

    if params.adjust_scale == 0:
        k_adjust = params.k_period

    else:
        k_adjust = int(np.ceil(params.k_period * np.power(2-params.gamma, epoch/params.adjust_scale)))

    writer.add_scalar('Train/k', k_adjust, epoch)

    # Permutation
    with torch.no_grad():

        # decide whether to permute
        if (epoch + 1 - epoch_0) == k_adjust:

            epoch_0 = epoch
            k_num += 1

            for name,param in model.named_parameters():
                if 'layer2.weight' in name:

                    # get the trained weight
                    weight_t = param.cpu().detach().numpy()

                    # get the ranking idx
                    rank = np.argsort(np.argsort(weight_t))

                    # take the untrained weight with the ranking idx
                    weight_permu = np.take(weight_0, rank)

                    # assign the value back
                    param.copy_(torch.from_numpy(weight_permu))

    return epoch_0, k_num


def test(dataloader, model, loss_fn, epoch, epoch_0, k_num, writer, device, params, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_loss_inf = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            test_loss_inf += torch.linalg.vector_norm(pred-y, ord=float('inf')).item()

    test_loss /= num_batches
    test_loss_inf /= num_batches

    # loss with regularization
    if params.weight_decay > 0:
        loss = test_loss + reg_loss(model)
        loss_inf = test_loss_inf + reg_loss(model)

        total_loss = loss.item()
        total_loss_inf = loss_inf.item()
    else:
        total_loss = test_loss
        total_loss_inf = test_loss_inf


    # write
#     write to show loss/acc/lr on tensorboard
    writer.add_scalar('Train/Loss',total_loss,epoch)
    writer.add_scalar('Train/Loss_inf',total_loss_inf,epoch)
    writer.add_scalar('Train/Learning Rate',optimizer.state_dict()['param_groups'][0]['lr'],epoch)

    if epoch == epoch_0:
        writer.add_scalar('Test/Permutation', total_loss, k_num)
        writer.add_scalar('Test/Permutation_inf', total_loss_inf, k_num)




def output_final(dataloader, model, loss_fn, device, params):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_loss_inf = 0, 0


    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            test_loss_inf += torch.linalg.vector_norm(pred-y, ord=float('inf')).item()

    test_loss /= num_batches
    test_loss_inf /= num_batches

    # loss with regularization
    if params.weight_decay > 0:
        loss = test_loss + reg_loss(model)
        loss_inf = test_loss_inf + reg_loss(model)

        total_loss = loss.item()
        total_loss_inf = loss_inf.item()
    else:
        total_loss = test_loss
        total_loss_inf = test_loss_inf


    return total_loss, total_loss_inf





def permutation_train(j, params, results_queue):
    
    setup_seed(int(2022+ (1000*j)))
    
    actual_width = 2*params.width
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    
    model = NN(layers = [1, actual_width, 1, 1]).to(device)
    
    loss_fn = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params.gamma)

    writer = SummaryWriter(comment='Sin-bs'+str(params.batch_size)+'-adjust'+str(params.adjust_scale)+'-'+str(params.times_n)+'Xpall-k'+str(params.k_period)+'-W'+str(params.width)+'-iter'+str(j))


    # product data
    # to ensure the training data is the same
    setup_seed(2022)
    # x \in [-1, 1]
    x_train = torch.rand(params.num_train,1)*2-1

    # certain function expression: y = sin(2 \pi x)
    y_train = - torch.sin(2 * torch.pi * x_train)

    # generate uniformly distributed test data
    temp = torch.linspace(-1, 1, params.num_test)

    # for consistency of tensor sizes
    x_test = torch.unsqueeze(temp,1)
    
    # certain function expression: y = sin(\pi x)
    y_test = - torch.sin(2 * torch.pi * x_test)


    # create data loaders.
    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    test_set = torch.utils.data.TensorDataset(x_test, y_test)

    train_dataloader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=params.num_test, shuffle=False)

    

    setup_seed(int(2022+ (1000*j)))
    
    '''
    set the first layers weight as a series of ReLU
        and the second layer to be uniformed distributed
    '''

    weight_scale = 1


    # initialize the weights and bias
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'layer1.weight' in name:
                param.copy_(weights_init([actual_width, 1]))
                param.requires_grad = False  # freeze the weights of first layer

            if 'layer1.bias' in name:
                param.copy_(bias_init(actual_width, j, params.times_w, params.merge))
                param.requires_grad = False  # freeze the weights of first layer

            if 'layer2.weight' in name:
                param.copy_(coefficient_init_random([1, actual_width], j, params.times_w, params.merge, weight_scale))

            if 'layer3.weight' in name:
                param.copy_(torch.tensor(weight_scale))

    parm = {}
    for name,param in model.named_parameters():
        parm[name]=param.cpu().detach().numpy()

    weight1 = parm['layer1.weight']
    #     print('layer1.weight:\n', weight1.T)

    bias1 = parm['layer1.bias']
    #     print('layer1.bias:\n', bias1)

    weight2 = parm['layer2.weight']
    #     print('layer2.weight:\n',weight2)

    weight3 = parm['layer3.weight']
    #     print('layer3.weight:\n',weight3)

    weight_0 = np.sort(weight2)

    
    


    epoch_0 = 0
    k_num = 0

    for epoch in range(params.epoches):
        [epoch_0, k_num] = train_by_permu_adjust(train_dataloader, model, loss_fn, optimizer, epoch, epoch_0, k_num, device, params, scheduler, writer, weight_0)
        torch.use_deterministic_algorithms(True)
        test(test_dataloader, model, loss_fn, epoch, epoch_0, k_num, writer, device, params, optimizer)


    torch.save(model.state_dict(), './trained_model/Sin-'+str(params.times_n)+'Xweight-bs'+str(params.batch_size)+'-k'+str(params.k_period)+'-adjust'+str(params.adjust_scale)+'-W'+str(params.width)+'-iter'+str(j)+'.pth')

    print("Hallelujah!")
    
    writer.close()
    
    results_queue.put((j, output_final(test_dataloader, model, loss_fn, device, params)))