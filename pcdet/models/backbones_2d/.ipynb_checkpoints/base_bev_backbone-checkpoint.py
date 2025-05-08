import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###改的
# from .SB_module import Spatial_bias
# try:
#     from pcdet.ops.DeformableConvolutionV2PyTorch.modules.mdeformable_conv_block import MdeformConvBlock
# except:
#     print("Deformable Convolution not built!")

###########我加的
# class CRU(nn.Module):
#     '''
#     alpha: 0<alpha<1
#     '''
#     def __init__(self, 
#                  op_channel:int,
#                  alpha:float = 1/2,
#                  squeeze_radio:int = 2 ,
#                  group_size:int = 2,
#                  group_kernel_size:int = 3,
#                  ):
#         super().__init__()
#         self.up_channel     = up_channel   =   int(alpha*op_channel)
#         self.low_channel    = low_channel  =   op_channel-up_channel
#         self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
#         self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
#         #up
#         self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
#         self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
#         #low
#         self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
#         self.advavg         = nn.AdaptiveAvgPool2d(1)

#     def forward(self,x):
#         # Split
#         up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
#         up,low  = self.squeeze1(up),self.squeeze2(low)
#         # Transform
#         Y1      = self.GWC(up) + self.PWC1(up)
#         Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
#         # Fuse
#         out     = torch.cat( [Y1,Y2], dim= 1 )
#         out     = F.softmax( self.advavg(out), dim=1 ) * out
#         out1,out2 = torch.split(out,out.size(1)//2,dim=1)
#         return out1+out2


#ASPP类是我新加的
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        feature_map_size = x.size()[2:]
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.pool(x)
        out5 = self.conv5(out5)
        out5 = nn.functional.interpolate(out5, size=feature_map_size, mode='bilinear', align_corners=True)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        # out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.relu(out)
        out = self.dropout(out)
        return out

########################我加的ADF模块，BSH-Det里的
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ADF(nn.Module):
    def __init__(self, planes):
        super(ADF, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
################################

#自适应多尺度注意力卷积塔
############我加的EMA
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        # print(channels)
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


##############我加的模块
class GloablConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GloablConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.ema = EMA(in_channels)
#         self.cru = CRU(in_channels)
        # self.ca = CoordAtt(in_channels,out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # self.SB = Spatial_bias(128)

    def forward(self, x):
        ###3.1版本
        x1 = self.conv(x)
        # x2 = self.ca(x)
        x2 = self.ema(x)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.conv1(x)
        x = x1 + x2
        #3.2 版本
#         x1 = x #残差连接
#         # x2 = self.ca(x)
#         x2 = self.ema(x) #注意力
#         x3 = self.cru(x) #通道重组单元
#         # x = torch.cat([x1, x2], dim=1)
#         # x = self.conv1(x)
#         x = x1 + x2 +x3

        return x



###我加的模块
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 将所有输入小于0的值都设置为0，同时对所有大于6的输入取值6

    def forward(self, x):
        return self.relu(x + 3) / 6


###我加的模块
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


###我加的坐标注意力模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)  # C//r

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()  # 激活函数

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n, c, 1, w) -> (n, c, w ,1)

        y = torch.cat([x_h, x_w], dim=2)  # (n, c, w+h, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # (n, c//r, h, 1), (n, c//r, w, 1)
        x_h, x_w = torch.split(y, [h, w], dim=2)  # 在第3个维度上进行拆分，得到水平方向和垂直方向上的特征向量
        x_w = x_w.permute(0, 1, 3, 2)  # (n, c//r, 1, w)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, num_frames, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_frames = num_frames

        ###我加的
        # self.mymodel = MyModel()
        ##我加的
        self.coordatt = CoordAtt(384, 384)
        self.conv1 = nn.Conv2d(384, 384, kernel_size=3, padding=1) #我加的
        self.adf = ADF(384)  #我加的
        self.aspp = ASPP(384, 96)  #我加的

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(
                self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            # assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        use_dcn = self.model_cfg.get('USE_DCN', False)

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        ###加的
        # self.SB = Spatial_bias(256)
        # self.num_sb = 3
        ###############加的
        # self.globconv2d = GloablConv2d(1, 128 ,3 , 1)  不加
        #################

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                ############原卷积块代码
                # cur_layers.extend([
                #     nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                #     nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                #     nn.ReLU()
                # ])
                ########我加的新卷积模块，暂时叫全局卷积
                cur_layers.extend([
                    GloablConv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    deblocks_list = []
                    if use_dcn:
                        deblocks_list.extend([
                            MdeformConvBlock(num_filters[idx], num_filters[idx],
                                             deformable_groups=1),
                            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ])
                    deblocks_list.extend([
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
                    self.deblocks.append(nn.Sequential(*deblocks_list))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        # more deblocks for higher resolusion if it is needed.
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
#         print('==> spatial_features in backbone2d')
#         print(spatial_features.size()) # [1, 256, 200, 176], bs=1, pvrcnn

        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
            ####我加的 暂时叫做Adaptive Multi-scale Semantic-Spatial Feature Fusion Module，简称AMSF模块
            x_c = self.conv1(x)
            x_weight = torch.sigmoid(x_c)
            x1 = x_weight[:, :128, :, :] * ups[0]
            x2 = x_weight[:, 128:256, :, :] * ups[1]
            x3 = x_weight[:, 256:384, :, :] * ups[2]
            x = torch.cat((x1, x2, x3), dim=1)
            # 借鉴cia-ssd的融合模块，自适应地融合丰富的浅层、中层空间特征和高层语义特征
        elif len(ups) == 1:
            x = ups[0]

#         out = self.coordatt(x)  # 我加的CA
#         x = x + out  ###我加的CA
        ###########################我加的下面
#         pred_hm = data_dict['pred_hm_1']
#         pred_hm2 = data_dict['pred_hm_2']
#         predhm = torch.cat((pred_hm2, pred_hm), dim=1)
# #         print('pred_hm',pred_hm.shape)
# #         print('pred_hm2',pred_hm2.shape)
#         predhm = predhm[:,0:35200,:]
#         predhm = predhm.view(1,1,200,176)
#         predhm = torch.cat((predhm, predhm,predhm,predhm), dim=0)
# #         print('predhm',predhm.shape)
# #         print('x',x.shape)
#         #x = torch.cat((x, predhm), dim=1)
#         x = x + predhm
        x = self.adf(x)
        ##########################上面我加的

#         x = self.aspp(x) ###我加的
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['st_features_2d'] = x
        # print('==> x in backbone2d')
        # print(x.size()) #[1, 512, 200, 176], bs=1, pvrcnn

        return data_dict
