import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FGSM Attack

def test_fgsm_attack(net, test_loader):
    '''FGSM attack testing function.'''

    correct = 0
    total = 0

    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        images_adv, labels_adv = fgsm_attack(net,images,labels)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(labels_adv.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


def fgsm_attack(net,images, labels,criterion=nn.CrossEntropyLoss(),eps=0.03):
    '''
    takes as inputs an image or a batch
    returns (a) perturbed image(s) and the associated preturbed prediction(s)"
    '''

    # Set requires_grad attribute of tensor
    images.requires_grad = True

    # Forward pass
    outputs = net(images)

    # Calculate the loss
    cost = criterion(outputs, labels)

    # Retain the gradient, important for non leaf tensors
    cost.retain_grad() 

    #Zero the gradient 
    net.zero_grad()

    # Backwords pass
    cost.backward()

    # Collect the gradient of the cost wrt images
    image_grad = images.grad.data

    # Create adversarial example
    images_adv = images + eps * image_grad.sign()

    # Re-classify the perturbed images
    labels_adv = net(images_adv)
    
    return images_adv, labels_adv


# PGD Attack

def test_pgd_attack(net, test_loader):
    '''FGSM attack testing function.'''

    correct = 0
    total = 0

    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        images_adv, labels_adv = pgd_attack(net, images, labels, nn.CrossEntropyLoss(), norm='inf')
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(labels_adv.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


def pgd_attack(net, x, y, loss_fn,norm, n_iter=20, step_size=0.01, eps=0.03, clamp=(0,1)):
    '''Take a model , data x and labels y and return '''

    x_adv = x.clone().detach().requires_grad_(True).to(device)
    for i in range(n_iter):
        x_adv_temp = x_adv.clone().detach().requires_grad_(True).to(device)
        pred = net(x_adv_temp)

        loss = loss_fn(pred,y)
        loss.backward()

        with torch.no_grad():

            if norm == 'inf' :
                grad = x_adv_temp.grad.sign() * step_size

            else :
                grad = x_adv_temp.grad * step_size / x_adv_temp.grad.view(x_adv_temp.shape[0], -1).norm(norm, dim=-1).view(-1, x.shape[1], 1, 1)

            x_adv += grad

            if norm == 'inf':
                x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

            else : 
                delta = x_adv - x

                mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
                scaling_factor[mask] = eps

                # .view() assumes batched images as a 4D Tensor
                delta *= eps / scaling_factor.view(-1, 1, 1, 1)

                x_adv = x + delta

        x_adv = x_adv.clamp(*clamp)

    y_adv = net(x_adv)

    return x_adv.detach(), y_adv.detach()


class PGDlinf():

    def __init__(self, model, eps=0.03, alpha=2/255, steps=40, random_init=True):
        self.model = model
        self.device = next(model.parameters()).device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_init = random_init

    def attack(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach().requires_grad_(True).to(self.device)
        loss_fn = nn.CrossEntropyLoss()

        # If random start pertub our data with a noise eps
        if self.random_init :
            adv_images = adv_images + torch.zeros_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = loss_fn(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images,
                                        retain_graph=False, create_graph=False)[0]
            adv_images = adv_images + self.alpha * grad.sign()
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        labels_adv = self.model(adv_images)

        return adv_images, labels_adv
        

class PGDl2():

    def __init__(self, model, eps=0.5, alpha=0.01, steps=40, random_init=True, eps_for_division=1e-10):
        self.model = model
        self.device = next(model.parameters()).device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_init = random_init
        self.eps_for_division = eps_for_division

    def attack(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss_fn = nn.CrossEntropyLoss()

        adv_images = images.clone().detach().to(self.device)
        batch_size = len(adv_images)

        if self.random_init:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            delta_flat = delta.view(adv_images.size(0),-1)
            norm_delta = delta_flat.norm(p=2,dim=1).view(adv_images.size(0),1,1,1)
            random_noise = torch.zeros_like(norm_delta).uniform_(0, 1)
            delta *= random_noise/norm_delta*self.eps
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = loss_fn(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        adv_labels = self.model(adv_images)
        return adv_images, adv_labels
    
    
'''def plot_attacks(net, test_loader, eps=0.03):
    print("start plotting")

    batch = next(iter(test_loader))
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device) 

    images_adv_fgsm, labels_adv_fgsm = fgsm_attack(net,images, labels,criterion=nn.CrossEntropyLoss(),eps=eps)
    images_adv_pgd, labels_adv_pgd = pgd_attack(net, images, labels, nn.CrossEntropyLoss(), norm='inf', eps=eps)
    
    images = images.cpu()
    images_adv_fgsm = images_adv_fgsm.cpu()
    images_adv_pgd = images_adv_pgd.cpu()
    
    print("Original image : ")
    plt.imshow(images[0].detach().permute(1, 2, 0))
    plt.show()
    print("FGSM Adversarial image  : ")
    plt.imshow(images_adv_fgsm[0].detach().permute(1, 2, 0))
    plt.show()
    print("PGD Adversarial image : ")
    plt.imshow(images_adv_pgd[0].detach().permute(1, 2, 0))
    plt.show()'''





