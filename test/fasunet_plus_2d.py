import torch
import torch.nn as nn


class New_iter(nn.Module):


    def __init__(self, ker_f, ker_fi):
        super(New_iter, self).__init__()
        self.F = ker_f
        self.Fi = ker_fi
        channels = ker_f.conv.weight.size(1)
        self.bn_out = nn.BatchNorm2d(channels)

        self.bn_u = nn.BatchNorm2d(channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, u_b):
        u, b = u_b
        out = b - self.bn_u(self.act(self.F(u)))
        out = u + self.Fi(out)
        out = self.bn_out(out)
        out = (out, b)

        return out



class Ker_fi(nn.Module):

    def __init__(self, channels, ks=3):
        super(Ker_fi, self).__init__()

        self.B1 = nn.Conv2d(channels, channels//2, kernel_size=ks, stride=1, 
                                    padding=ks//2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels//2)
        self.act = nn.ReLU6(inplace=True)

        self.B2 = nn.Conv2d(channels//2, channels//2, kernel_size=ks, 
                                    stride=1, padding=ks//2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels//2)
        # self.act2 = nn.ReLU6(inplace=True)
        # 
        self.B3 = nn.Conv2d(channels//2, channels, kernel_size=ks, 
                                    stride=1, padding=ks//2, bias=False)
        
        # self.act3 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.bn1(self.act(self.B1(x)))
        x = self.bn2(self.act(self.B2(x)))
        x = self.act(self.B3(x))

        return x


class Ker_f(nn.Module):
    def __init__(self, channels, ks=3):
        super(Ker_f, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=ks, 
                                stride=1, padding=ks//2, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x


class Restriction(nn.Module):


    def __init__(self, f_pre, f_next,in_channels, out_channels, ks=3):
        super(Restriction, self).__init__()
        self.R_u = nn.Conv2d(in_channels, out_channels, ks, 2, ks//2, bias=False)
        self.f_previous = f_pre
        self.f_next = f_next

        self.R_bu = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)
        self.act = nn.ReLU6(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

        # if channels[i] != channels[i+1]:
            #     t1 =channels[i]
            #     t2 = channels[i+1]
            #     setattr(self, 'b_short_'+str(i+1), Channels_fun(t1, t2))
            #     setattr(self, 'u_short_'+str(i+1), Channels_fun(t1, t2))
        
    def forward(self, out):
        u, b = out
        del out

        
        u_next = self.R_u(u)

        # update f follow:  pool(f) - pool( a_now(u) ) + a_next( pool(u_next) )
        b = self.R_bu(b-self.f_previous(u)) + self.f_next(u_next)
        del u
        b = self.act(b)
        b = self.bn(b)

        # if self.channels[i-1] != self.channels[i]:
        #         u = getattr(self, 'u_short_'+str(i))(out[0])
        #         b = getattr(self, 'b_short_'+str(i))(out[1])
        #         out = (u, b)

        return (u_next, b)


class Channels_fun(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Channels_fun, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1,
                            stride=1, padding=0, 
                            bias=False, 
                            # groups=16
                            )
    def forward(self, x):
        x = self.conv(x)
        return x 


class Coarse_layer(nn.Module):

    def __init__(self, ker_f, channels, iter=7, Fi_ks=3):
        super(Coarse_layer, self).__init__()
        layers = []
        
        for _ in range(iter):
            ker_fi = Ker_fi(channels, ks=Fi_ks)
            layers.append(New_iter(ker_f, ker_fi))

        # setattr(self, 'coarse', nn.Sequential(*layers))
        self.coarse = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.coarse(x)

        return x[0]


class Correction_u(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels, ks=3):
        super(Correction_u, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 
                                kernel_size=ks, stride=2, 
                                padding=ks//2, output_padding=1)
    
    def forward(self, CUb_CUa_Ub):
        # error
        out = CUb_CUa_Ub[0] - CUb_CUa_Ub[1]
        # print(out.size(), self.upsample.weight.size())
        out = self.upsample(out)

        # u + u_0
        out = CUb_CUa_Ub[2] + out

        return out


class FAS_Unet_plus_2d(nn.Module):
    '''
        channels = [32, 64, 128, 256, 512] or
        channels = [64, 64, 64, 64, 64]
        比 iter_num[0] 
    '''
    def __init__(self, in_channel, channels, num_classes,
                    iter_num = [[3,3,3,3], [7], [4,4,4,4]], 
                    F_ks=[3, 3, 3, 3,3], 
                    Fi_ks= [3, 3, 3, 3,3]):
            # item_num[0], iter_num[2] 
        super(FAS_Unet_plus_2d, self).__init__()

        self.channels = channels
        self.conv_start = nn.Conv2d(in_channel, channels[0], 3,
                            stride=1, padding=3//2, bias=False)
        self.bn_start = nn.BatchNorm2d(channels[0])



        ker_f = Ker_f(channels[0])
        for i in range(len(channels)-1):  # 0, 1, 2, 3

        
            layers = []
            for _ in range(iter_num[0][i]):
                ker_fi = Ker_fi(channels=channels[i], ks=Fi_ks[i])
                layers.append(New_iter(ker_f, ker_fi))


                                                                  
            setattr(self, 'left_layer_'+ str(i+1), nn.Sequential(*layers))
            Ker_f_next = Ker_f(channels[i+1])
            setattr(self, 'pool_'+str(i+1), 
                    Restriction(ker_f, Ker_f_next, channels[i], channels[i+1]))

            ker_f = Ker_f_next


        self.coarse = Coarse_layer(ker_f, channels[i+1], iter_num[1][0], Fi_ks=Fi_ks[i])


        
        for i in range(len(channels)-2, -1, -1):  # 3, 2, 1, 0
            # print(i)
            ker_f = Ker_f(channels[i])  
            setattr(self, 'correction_u_'+ str(i+1), Correction_u(channels[i+1], channels[i]))
            # print(i, channels[i+1], channels[i])

            layers = []
            for _ in range(iter_num[2][i]):
                ker_fi = Ker_fi(channels=channels[i], ks=Fi_ks[i])
                layers.append(New_iter(ker_f, ker_fi))

            setattr(self, 'right_layer_'+ str(i+1), nn.Sequential(*layers))
        

        self.final_1 = nn.Conv2d(channels[0], num_classes, 3, 1, 1, bias=False)


    def forward(self, b):
        b = self.bn_start(self.conv_start(b))
        out = (b, b)   # (u_1a, b)


        # out = self.layer_1(out)  
        #out = getattr(self, 'layer_'+str(1))(out)

        b_list, u_a_list, u_b_list = [], [], []
        for i in range(1, len(self.channels)):
            
            b_list.append(out[1])
            u_a_list.append(out[0])

            out = getattr(self, 'left_layer_'+str(i))(out)
            
            u_b_list.append(out[0])

            out = getattr(self, 'pool_'+str(i))(out)

        u_a_list.append(out[0])
        out = self.coarse(out)

        for i in range(len(self.channels)-2, -1, -1):
            # print(i, out.size(), u_a_list[i+1].size(), u_b_list[i].size())

            out = (out, u_a_list[i+1], u_b_list[i])
            # print(out[0].size(), out[1].size(), i,  '\n')
            out = getattr(self, 'correction_u_' + str(i+1))(out)
            
            out = (out, b_list[i])
            out = getattr(self, 'right_layer_' + str(i+1))(out)[0]
            # out = out[0]
            del u_a_list[i+1]
            del u_b_list[i]
            del b_list[i]

        out = self.final_1(out)


        return out




if __name__ == "__main__":

    in_channel  = 1
    num_classes = 5
    channels = [64, 64, 64, 64, 64]
    # channels = [16, 32, 64, 96, 128]
    iter_num = [[3, 3, 3, 3], [7], [4, 4, 4, 4]]

    f = torch.randn((1, in_channel, 32, 32))

    model = FAS_Unet_plus_2d(in_channel, channels, num_classes, iter_num=iter_num)
    # output1= model(f)
    # from torchsummaryX import summary
    
    # import pandas as pd
    

    # pd.set_option('display.max_columns', None)

    # pd.set_option('display.max_rows', None)

    # pd.set_option('max_colwidth',100)
    # data = open("model.txt",'w',encoding="utf-8")
    # # print(summary(model, f), file=data)
    # print(model, file=data)

    # print(output1.size())

    def count_parameters(model):
        """
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    print("number of parameters ：", count_parameters(model))
    print('finish')
