import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
    
class RelLSTMLayer(nn.Module):
    def __init__(self, input_size, bridge_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super(RelLSTMLayer, self).__init__()
        self.input_size = input_size
        self.bridge_size = bridge_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        self.layers = self.initialize_layers()
    
    def initialize_layers(self):
        pass
    
    def forward(self, x, b, hidden=None):
        if self.batch_first:
            x = x.transpose(0, 1)
            b = b.transpose(0, 1)
        
        seq_len, batch_size, _ = x.size()
        
        if hidden is None:
            hidden = tuple([torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(2)])
        
        # Initialize a tensor to hold the outputs at each time step
        outputs = []
    
        for layer in range(self.num_layers):
            # Get the cell corresponding to this layer
            cell = self.layers[layer]
            
            # Initialize the hidden state for this layer
            h = hidden[0][layer]
            c = hidden[1][layer]
            
            for t in range(seq_len):
                # Get the input at time step t for layer 0, and the output of the previous layer at time step t for other layers
                if layer == 0:
                    input_x = x[t]
                    input_b = b[t]  # Assuming b also has time steps
                    new_h, new_c = cell(input_x, input_b, (h, c))
                else:
                    input_x = outputs[t]
                    new_h, new_c = cell(input_x, (h, c))
                
                # Store the output of this layer at time step t
                if layer == 0:
                    outputs.append(new_h)
                else:
                    outputs[t] = new_h
                h = new_h
                c = new_c
            
            # Update the hidden state for this layer
            hidden[0][layer] = h
            hidden[1][layer] = c
        
        outputs = torch.stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
            
        return outputs, hidden
    
class PropagationUnitCell(nn.Module):
    # LSTM + Forget applied to bridge and hidden by input
    def __init__(self, input_size, bridge_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.bridge_size = bridge_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2f = nn.Linear(input_size, hidden_size + bridge_size, bias=bias)
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        
        if bridge_size != 0:
            self.b2h = nn.Linear(bridge_size, 4 * hidden_size, bias=bias)
        
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, b, hidden):
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        # Additional forget gate to overwrite the previous hidden state and bridge
        bh_forget = self.x2f(x)
        hx = torch.sigmoid(bh_forget[..., :self.hidden_size]) * hx.clone()
        b = torch.sigmoid(bh_forget[..., self.hidden_size:]) * b.clone()
        
        gates = self.x2h(x) + self.h2h(hx)
        
        if self.bridge_size != 0:
            gates = gates.clone() + self.b2h(b)
        
        forgetgate, ingate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = cx.clone() * forgetgate + ingate * cellgate

        hy = outgate * torch.tanh(cy)
        
        return (hy, cy)
    
class PropagationUnit(RelLSTMLayer):
    def __init__(self, input_size, bridge_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super(PropagationUnit, self).__init__(input_size, bridge_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first)
    
    def initialize_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                layers.append(
                    PropagationUnitCell(
                        input_size=self.input_size, 
                        bridge_size=self.bridge_size,
                        hidden_size=self.hidden_size,
                        bias=self.bias
                    )
                )
            else:
                layers.append(
                    PropagationUnitCell(
                        input_size=self.hidden_size, 
                        bridge_size=0,
                        hidden_size=self.hidden_size,
                        bias=self.bias
                    )
                )
        return layers
    
    def forward(self, x, b, hidden=None):
        if self.batch_first:
            x = x.transpose(0, 1)
            b = b.transpose(0, 1)
        
        seq_len, batch_size, _ = x.size()
        
        if hidden is None:
            hidden = tuple([torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(2)])
        
        # Initialize a tensor to hold the outputs at each time step
        outputs = []
    
        for layer in range(self.num_layers):
            # Get the cell corresponding to this layer
            cell = self.layers[layer]
            
            # Initialize the hidden state for this layer
            h = hidden[0][layer]
            c = hidden[1][layer]
            
            for t in range(seq_len):
                # Get the input at time step t for layer 0, and the output of the previous layer at time step t for other layers
                if layer == 0:
                    input_x = x[t]
                    input_b = b[t]  # Assuming b also has time steps
                    new_h, new_c = cell(input_x, input_b, (h, c))
                else:
                    input_x = outputs[t]
                    input_b = input_x[..., []]
                    new_h, new_c = cell(input_x, input_b, (h, c))
                
                # Store the output of this layer at time step t
                if layer == 0:
                    outputs.append(new_h)
                else:
                    outputs[t] = new_h
                h = new_h
                c = new_c
            
            # Update the hidden state for this layer
            hidden[0][layer] = h
            hidden[1][layer] = c
        
        outputs = torch.stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
            
        return outputs, hidden
