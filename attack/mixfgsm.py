import torch
import torch.nn as nn
from torchattacks.attack import Attack


class MIXFGSM(Attack):
    r"""
    altered from torchattack
    """
    def __init__(self, model, forward_function_list=None, eps=0.007, T=None, **kwargs):
        super().__init__("MIXFGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.forward_function_list = forward_function_list
        self.T = T

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        # images.unsqueeze_(1)
        # images = images.repeat(8, 1, 1, 1, 1)
        images = images.clone().detach().to(self.device)
        
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True

        if self.forward_function_list is not None:
            tot_grad = 0.
            for ff in self.forward_function_list:
                outputs = ff(self.model, images, self.T)
            
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
          
                # Update adversarial images
                grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
                tot_grad += grad / len(self.forward_function_list)
                #images.grad.zero_()
      
            adv_images = images + self.eps*tot_grad.sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()            

        else:
            outputs = self.model(images)
      
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
      
            # Update adversarial images
            grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
      
            adv_images = images + self.eps*grad.sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images