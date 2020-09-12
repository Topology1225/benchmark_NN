
import torch
import torch.nn as nn


class CNN28(nn.Module):
    """Note
    h_out = int((H_in + 2*padding[0] - dilation[0] * (kernel_size[0]-1)-1)/stride[0]+1) 
    w_out = int((W_in + 2*padding[1] - dilation[1] * (kernel_size[1]-1)-1)/stride[1]+1)
    """

    def __init__(self):
        super().__init__()
        self.name  = "cnn28" 
        self.epoch = 0 
        self.h = 28
        self.w = 28 
        self.in_channel = 1
        self.build(
            in_channel=self.in_channel,
            h=self.h,
            w=self.w
        )


    def build(self, in_channel, h, w):
        layers = list()
        out_channel = in_channel*2  
        H_in = h
        W_in = w
        padding     = (0, 0)
        stride      = (1, 1) 
        kernel_size = (3, 3)
        dilation    = (1, 1)
        
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=1,
                    bias=True,
                    padding_mode="zeros"
                ),
                nn.ReLU()
            )
            layers.append(layer)
            tmp = in_channel
            in_channel = out_channel
            out_channel = 2*tmp 
            H_in = self.cal_h(H_in, padding, dilation, kernel_size, stride)
            W_in = self.cal_w(W_in, padding, dilation, kernel_size, stride)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x

    def save(self, save_file_path=None):
        assert save_file_path is not None
        torch.save(self.state_dict(), save_file_path)

    def load(self, load_file_path=None):
        assert load_file_path is not None
        self.load_state_dict(torch.load(load_file_path))

    @staticmethod
    def cal_h(H_in, padding, dilation, kernel_size, stride):
        h_out = int((H_in + 2*padding[0] - dilation[0] * (kernel_size[0]-1)-1)/stride[0]+1) 
        return h_out
    
    @staticmethod
    def cal_w(W_in, padding, dilation, kernel_size, stride):
        w_out = int((W_in + 2*padding[1] - dilation[1] * (kernel_size[1]-1)-1)/stride[1]+1)
        return w_out

if __name__=="__main__":
    model = CNN28()
    x = torch.ones((1,1,28,28))
    breakpoint()
    x = model(x)
    breakpoint()
    
