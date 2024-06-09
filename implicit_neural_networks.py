#taken from https://github.com/ykasten/layered-neural-atlases
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output

class IMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=256,
            use_positional=True,
            positional_dim=10,
            skip_layers=[4, 6],
            scale_output=1.0,
            num_layers=8,  # includes the output layer
            verbose=True,use_tanh=False,apply_softmax=False,geometric_init=False):
        super(IMLP, self).__init__()
        self.scale_output=scale_output
        self.verbose = verbose
        self.use_tanh = use_tanh
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax= nn.Softmax()
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim + input_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(positional_dim)],requires_grad = False)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim
            
            if i == num_layers - 1:
                # last layer
                lin=nn.Linear(input_dims, output_dim, bias=True)
        
                if geometric_init:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(input_dims), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -0.5)
                self.hidden.append(lin)
            else:
                lin = nn.Linear(input_dims, hidden_dim, bias=True)

                if geometric_init:
                    if  i==0:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                        torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(hidden_dim))
                    elif i in skip_layers:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.constant_(lin.weight[:, (hidden_dim+3):], 0.0)
                        torch.nn.init.normal_(lin.weight[:, :(hidden_dim+3)], 0.0, np.sqrt(2) / np.sqrt(hidden_dim))
                    else:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden_dim))
                self.hidden.append(lin)

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

        if self.verbose:
            print(f'Model has {count_parameters(self)} params')

    def forward(self, x):
        if self.use_positional:
            if self.b.device!=x.device:
                self.b=self.b.to(x.device)
            pos = positionalEncoding_vec(x,self.b)
            x = torch.cat((x,pos),dim=-1)

        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)
        x=x*self.scale_output
        if self.use_tanh:
            x = torch.tanh(x)

        if self.apply_softmax:
            x = self.softmax(x)
        return x
    def gradient(self, x,relevant_outputs_dim=0):
        x.requires_grad_(True)
        y = self.forward(x)[:,relevant_outputs_dim]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)



def apply_gradient(model,x,relevant_outputs_dim=0):
    x.requires_grad_(True)
    y = model(x)[:,relevant_outputs_dim]
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients.unsqueeze(1)


