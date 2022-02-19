import os
import argparse
import torch
import math
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import attacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

valid_size = 1024 
batch_size = 32 

def model_kl_divergence_loss(model, kl_weight=1.):
    """
    Compute the KL loss of the model
    """

    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl_sum = torch.Tensor([0]).to(device)

    # get the layers as a list
    model_children = list(model.children())

    for layer in model_children:
        if hasattr(layer, 'kl_'):
            kl_sum += layer.kl_

    return kl_weight * kl_sum[0]


def concrete_sample(a, temperature, hard=False, eps=1e-8, axis=-1):
    """
    Sample from the concrete relaxation.
    :param probs: torch tensor: probabilities of the concrete relaxation
    :param temperature: float: the temperature of the relaxation
    :param hard: boolean: flag to draw hard samples from the concrete distribution
    :param eps: float: eps to stabilize the computations
    :param axis: int: axis to perform the softmax of the gumbel-softmax trick
    :return: a sample from the concrete relaxation with given parameters
    """

    U = torch.rand(a.shape, device = a.device)
    G = - torch.log(- torch.log(U + eps) + eps)
    t = (a + G) / temperature

    y_soft = F.softmax(t, axis)

    if hard:
        _, k = y_soft.data.max(axis)
        shape = y_soft.size()

        if len(a.shape) == 2:
            y_hard = y_soft.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        else:
            y_hard = y_soft.new(*shape).zero_().scatter_(-1, k.view(-1, 1, a.size(1), a.size(2), a.size(3)), 1.0)

        y = (y_hard - y_soft).detach() + y_soft

    else:
        y = y_soft

    return y


class LWTA(nn.Module):
    """
    A simple implementation of the LWTA activation to be used as a standalone approach for various models.
    """

    def __init__(self, inplace = True, U=2):
        super(LWTA, self).__init__()
        self.inplace = inplace
        self.temperature = -.01
        self.temp_test = 0.01
        self.kl_ = 0.
        self.U = U
        self.temp = nn.Parameter(torch.tensor(self.temperature))

    def forward(self, input):
        out, kl = lwta_activation(input, U = self.U, training = self.training,
                                  temperature = F.softplus(self.temp),
                                  temp_test = self.temp_test)
        self.kl_ = kl

        return out

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def lwta_activation(input, U = 2, training = True, temperature = 0.67, temp_test = 0.01):
    """
    The general LWTA activation function.
    Can be either deterministic or stochastic depending on the input.
    Deals with both FC and Conv Layers.
    """
    out = input.clone()
    kl= 0.

    # case of a fully connected layer
    if len(out.shape) == 2:
        logits = torch.reshape(out, [-1, out.size(1)//U, U])
        
        if training:
            mask = concrete_sample( logits, 0.67)
        else:
            mask = concrete_sample(logits, temp_test  )
        mask_r = mask.reshape(input.shape)
    else:
        x = torch.reshape(out, [-1, out.size(1)//U, U, out.size(-2), out.size(-1)])
        
        logits = x
        if training:
            mask = concrete_sample(logits, temperature, axis = 2)
        else:
            mask = concrete_sample(logits , temp_test, axis = 2)

        mask_r = mask.reshape(input.shape)

    if training:
        q = mask
        log_q = torch.log(q + 1e-8)
        log_p = torch.log(torch.tensor(1.0 / U))

        kl = torch.sum(q * (log_q - log_p), 1)
        kl = torch.mean(kl) / 1000.

    input *= mask_r

    return input, kl

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.lwta1 = LWTA()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.lwta2 = LWTA()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.lwta1(self.bn1(x))
        else:
            out = self.lwta1(self.bn1(x))
        out = self.lwta2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"  # The name of the saved model_file can de changed for each network to iterate designs
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''
    
    def __init__(self, depth=34, num_classes=10, widen_factor=1, extra_stride=1, dropRate=0.0):
        super(Net, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                                padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, extra_stride, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.lwta = LWTA()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
        self.eval()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.lwta(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def save(self, model_file):
        # Function used to save the model once trained to './models/default_model.pth'
        # The name of the saved model_file can de changed for each network to iterate designs
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)



    def load(self, model_file):#function used to load the saved model in 'test_project.py' and by the teachers online tester
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))


        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
        project, and will load the model weights from the file
        specify in Net.model_file.
        
        You must not change the prototype of this function. You may
        add extra code in its body if you feel it is necessary, but
        beware that paths of files used in this function should be
        refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, self.model_file))


def train_model(net, train_loader, pth_filename, num_epochs, ): #pth_filename : path to save the mode
                                                            #train_loader : result of the function get_train_loader 
    print("Starting training")
    
    '''Defining loss'''
    criterion = nn.CrossEntropyLoss()
    '''defining loss function'''
    optimizer = optim.Adam(net.parameters()) 

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0): #loop over the dataset

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs_adv, _ = attacks.pgd_attack(net, inputs, labels, nn.CrossEntropyLoss(), norm='inf')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs_adv)
            
            loss = criterion(outputs, labels)
            # backward 
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()
        print("Training epoch {} ;   Running Loss = {}".format(epoch+1,running_loss))

    net.save(pth_filename) #save the trained model to the specified model_file="models/..."
    print('Model saved in {}'.format(pth_filename))


def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train


def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid


def main():
    
    # Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                            "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    # Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net = net.to(device)
    

    # Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model : ",args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        train_model(net, train_loader, args.model_file, args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    # Model testing
    print("Testing with model from '{}'. : ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {} %".format(acc))

    fgsm_acc = attacks.test_fgsm_attack(net, valid_loader)
    print("Model FGSM-attacked accuracy (test): {} %".format(fgsm_acc))

    pgd_acc = attacks.test_pgd_attack(net, valid_loader)
    print("Model PGD-attacked accuracy (test): {} %".format(pgd_acc))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "              "it will not be the one used for testing your project. "              "If this is your best model, "              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))


if __name__ == "__main__":
    main()


# In[ ]:




