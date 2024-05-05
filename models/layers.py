import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyFloor(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        # x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self, X):
        return normalizex(X,self.mean,self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

'''
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        #tmp = (1 / gama) * ((gama - input.abs())>=0).float()
        grad_input = grad_output * tmp
        #grad_input = grad_output
        return grad_input, None
'''


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, lam):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L, torch.tensor([lam]))
        return out
        #return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others, lam) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        #tmp = (1 / (2 * gama)) * ((gama - input.abs())>=0).float()
        tmp = tmp * lam.item()
        
        grad_input = grad_output * tmp

        return grad_input, None, None


# log(tau)/(log(gamma*log(1+exp((tau-1+x)/gamma))/x))

class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau, reset_type):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem*tau + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            if reset_type is 'hard':
                mem = (1 - spike) * mem
            else:
                mem -= spike
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, out, tau = ctx.saved_tensors
        x = x.mean(0, keepdim=True)
        '''
        gamma = 0.01
        exp1 = torch.exp((tau-1+x)/gamma)
        g1 = exp1/(1+exp1)
        log1 = gamma*torch.log(1+exp1)
        h = torch.log(log1/x)
        grad = (log1-x*g1)*torch.log(tau)/(x*log1*h*h)
        grad = grad*(x>0).float()*(x<1).float()
        '''
        # out = out.mean(0, keepdim=True)
        # grad_input = grad_output * ((x > (1-tau)).float() * (x < 1).float()*1 + (x > 0).float()*(x< 1-tau).float()*.1)
        # grad_input = grad_output * (x>1-tau).float()*(x<1).float()
        # grad = ((1-tau)/(x*(tau-1+x))).clamp(0, 1) * (x>1-tau).float()*(x<1).float() + (x<1-tau).float()*(x>0).float()*0.1
        gamma = 0.2
        ext = 1 #
        des = 1
        grad = (x>=1-tau).float()*(x<=1+ext).float()*(des-gamma+gamma*tau)/(tau+ext) + (x<=1-tau).float()*(x>=0).float()*gamma
        grad_input = grad_output * grad
        return grad_input, None, None
        
        
class RateBp2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau, reset_type):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem*tau + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            if reset_type is 'hard':
                mem = (1 - spike) * mem
            else:
                mem -= spike
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        T = out.shape[0]
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()
        return grad_input, None, None
        

class RateBp3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau, reset_type):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem*tau + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            if reset_type is 'hard':
                mem = (1 - spike) * mem
            else:
                mem -= spike
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, out, tau = ctx.saved_tensors
        T = out.shape[0]
        x = x.mean(0).unsqueeze(0)
        out = out.mean(0).unsqueeze(0)
        mask1 = (x > 0.)
        grad_rate1 = torch.where(mask1, out / x, torch.zeros_like(x))
        grad_rate1 = grad_rate1.sum().item() / (mask1.float().sum().item()+0.001)
        grad_rate = mask1.float() * grad_rate1
        grad_input = grad_output * grad_rate
        return grad_input, None, None

        

class GradAvg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.mean(dim=0,keepdim=True).expand_as(grad_output)        
        return grad_input


class STNE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mem = torch.rand_like(x[0])
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            mem = mem - spike
            spike_pot.append(spike)
            # random.shuffle(spike_pot)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        x = x.mean(0, keepdim=True)
        # grad = (x>0).float()*(x<1).float()
        # grad_input = grad_output*grad
        return grad_output


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

floor = STE.apply

class qcfs(nn.Module):
    def __init__(self, up=8., t=8):
        super().__init__()
        # self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        # x = x / self.up
        x = torch.clamp(x, 0, 1)
        x = floor(x*self.t+0.5)/self.t
        # x = x * self.up
        return x

class LIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.act2 = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.relu = qcfs()
        self.ratebp = RateBp.apply
        self.ratebp2 = RateBp2.apply
        self.ratebp3 = RateBp3.apply
        self.mode = 'bptt'
        self.T = T
        self.up = 1
        self.gamma = 1. #3.257 #2.848 #3.257 #2.848 #3.257 0.8
        self.grad_avg = GradAvg.apply
        self.reset_type = 'hard' #'hard'

    def forward(self, x):
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x, self.tau, self.reset_type)
            x = self.merge(x)
        elif self.mode == 'bptr2' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp2(x, self.tau, self.reset_type)
            x = self.merge(x)    
        elif self.mode == 'bptr3' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp3(x, self.tau, self.reset_type)
            x = self.merge(x)                
        elif self.mode == 'avg' and self.T > 0:
            x = self.expand(x)
            x = self.grad_avg(x)
            mem = torch.zeros_like(x[0])
            spike_pot = []
            for t in range(self.T):
                mem = mem.detach() * self.tau + x[t, ...]
                spike = self.act(mem - self.thresh, self.gamma, 1.)    
                if self.reset_type is 'hard':             
                    mem = (1 - spike) * mem
                else:
                    mem -= spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            #x = self.grad_avg(x)
            x = self.merge(x)
        elif self.mode == 'bptt' and self.T > 0:
            x = self.expand(x)
            
            #idx = torch.randperm(x.shape[0])
            #x = x[idx,:].view(x.size())
            
            mem = torch.zeros_like(x[0])
            #mem = 0.
            spike_pot = []
            for t in range(self.T):
                mem = mem * self.tau + x[t, ...]
                spike = self.act2(mem - self.thresh, 1., 1.)  
                if self.reset_type is 'hard':               
                    mem = (1 - spike) * mem
                else:
                    mem -= spike
                spike_pot.append(spike)
            
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)            
        else:
            x = self.relu(x)
        # x = x*self.up
        return x

def add_dimention(x, T):
    x = x.unsqueeze(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

'''
class Poi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rand = torch.rand_like(x)
        out = (x>=rand).float()
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        # x = x.mean(0, keepdim=True)
        # out = out.mean(0, keepdim=True)
        return grad_output

poi = Poi.apply

class Poisson(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = poi(x)
        return out
'''


class Poi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rand):
        out = (x>=rand).float()
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        # x = x.mean(0, keepdim=True)
        # out = out.mean(0, keepdim=True)
        return grad_output, None

poi = Poi.apply

class Poisson(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        if self.mask is None:
            self.mask = torch.rand_like(x)
        out = poi(x, self.mask)
        return out


class Atk_Poisson(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = STNE.apply

    def forward(self, x):
        x = self.act(x)
        return x
