from copy import deepcopy
import math
import os
import shutil
import struct
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import os

config_path = os.path.join(os.path.dirname(__file__), '..', "config", "SingleExp", "zjc.yaml")
config = OmegaConf.load(config_path)

def get_nnmodule_param_count(module: nn.Module):
    param_count = 0
    for param in module.state_dict().values():
        param_count += int(np.prod(param.shape))
    return param_count


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def gate_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            # m.weight.kaiming_uniform_(-1 / num_input, 1 / num_input)
            try:
                torch.nn.init.kaiming_uniform_(m.weight) 
            except:
                print("this_layer cannot init")

class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)


class SIREN(nn.Module):
    """
    V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein,
    “Implicit Neural Representations with Periodic Activation Functions,”
    arXiv:2006.09661 [cs, eess], Jun. 2020, Accessed: May 04, 2021. [Online].
    Available: http://arxiv.org/abs/2006.09661

    Y. Lu, K. Jiang, J. A. Levine, and M. Berger,
    “Compressive Neural Representations,”
    Computer Graphics Forum, p. 12, 2021.
    """

    def __init__(
        self,
        coords_channel=3,
        data_channel=1,
        features=256,
        layers=5,
        w0=30,
        output_act=False,
        **kwargs,
    ):
        super().__init__()
        self.net = []
        self.net.append(nn.Sequential(nn.Linear(coords_channel, features), Sine(w0)))
        for i in range(layers - 2):
            self.net.append(nn.Sequential(nn.Linear(features, features), Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features, data_channel), Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features, data_channel)))
        self.net = nn.Sequential(*self.net)
        
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        # print(self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    @staticmethod
    def calc_param_count(coords_channel, data_channel, features, layers, **kwargs):
        param_count = (
            coords_channel * features
            + features
            + (layers - 2) * (features**2 + features)
            + features * data_channel
            + data_channel
        )
        return int(param_count)
    
    

    @staticmethod
    def calc_features(param_count, coords_channel, data_channel, layers, **kwargs):
        a = layers - 2
        b = coords_channel + 1 + layers - 2 + data_channel
        c = -param_count + data_channel

        if a == 0:
            features = round(-c / b)
        else:
            features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
        return features

class SIREN_zcq(nn.Module):
    def __init__(
        self,
        coords_channel=3,
        data_channel=1,
        features = None,
        expert_layers=config.network_structure.expert_layers,
        w0=30,
        output_act=False,
        **kwargs,
    ):
        super().__init__()
        self.net = []
        self.net.append(nn.Sequential(nn.Linear(coords_channel, features), Sine(w0)))
        for i in range(expert_layers - 2):
            self.net.append(nn.Sequential(nn.Linear(features, features), Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features, data_channel), Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features, data_channel)))
        self.net = nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)

    def forward(self, coords):
        output = self.net(coords)
        return output
    
#ourmodel          
class MoE(nn.Module):
    def __init__(
        self,
        n_network_features,
        coords_channel,
        bandwidth=45,
        experts_num=None, 
        gating_feature=None,
        topk=None,
        gating_layers = 2,
        expert_layers=None,
        **kwargs,
    ):
        super().__init__()
        self.coords_channel = coords_channel
        self.experts_num = experts_num
        self.gating_feature = gating_feature
        self.topk = topk
        self.expert_num = experts_num
        self.expert_layers = expert_layers
        self.n_network_features = n_network_features
        # #计算SIREN模型的features
        # n_network_features = MoE.calc_features_ourmodel(
        #     param_count=ideal_network_parameters_count, **config.network_structure,
        # )
        
        ### First Linear ###
        self.first_linear = nn.Sequential(nn.Linear(coords_channel, gating_feature))
        ### Gating Network ###
        self.gate = []
        for i in range(gating_layers-2):
            self.gate.append(nn.Sequential(nn.Linear(gating_feature, gating_feature), nn.ReLU()))
        self.gate.append(nn.Sequential(nn.Linear(gating_feature, gating_feature), nn.LayerNorm([gating_feature])))
        self.gate.append(nn.Sequential(nn.Linear(gating_feature, experts_num)))
        self.gate.append(nn.Sequential(nn.Softmax(dim=1)))
        self.gate = nn.Sequential(*self.gate)
        ### Experts Network ###
        self.expert = []
        for i in range(experts_num):
            freq = bandwidth/2.0 + i*bandwidth #指定每一个SIREN模型里用到的激活函数sine的w0
            self.expert.append(nn.Sequential( \
                SIREN_zcq(coords_channel = self.gating_feature, layers=self.expert_layers, \
                features= self.n_network_features, data_channel = 1, w0 = freq, output_act=False)))
        self.expert = nn.ModuleList(self.expert)
        # self.expert.cuda()
        #初始化first_linear和gating
        self.first_linear.apply(first_layer_sine_init)
        self.gate.apply(gate_init)

    def forward(self, coords):
        feature = self.first_linear(coords)
        index = self.gate(feature)
        x = None
        for idx, m in enumerate(self.expert):
            if x == None:
                x = m(feature)
            else:
                x = torch.cat((x, m(feature)), dim = 1)

        indices_to_remove = ~(index < torch.topk(index, self.topk)[0][..., -1, None])
        # index_v = indices_to_remove    
        output = torch.sum(x * indices_to_remove, dim = 1, keepdim=True)
        
        # Auxiliary loss
        c = (indices_to_remove == True).sum(dim = 0) * 1.0
        m = (indices_to_remove == True).sum(dim = 0) * 1.0

        N = x.shape[0]
        n = self.expert_num
        aux_loss = n/N/N*(torch.dot(c,m))
        
        # q = torch.ones_like(m)
        # q = q*(N/n)
        
        # kld_loss = torch.sum(m*torch.log(m) - m*torch.log(q)) /N/n
        
        return output, aux_loss
    
    # cal_param_count_ourmodel, 计算有多少个参数
    @staticmethod
    def calc_param_count_ourmodel(coords_channel, data_channel, features, expert_layers, experts_num, gating_feature, **kwargs):
        param_count = (
            (gating_feature * features + features + (expert_layers-2) * (features**2 + features) + features * data_channel + data_channel) * experts_num
            + 3 * (gating_feature**2 + gating_feature) + gating_feature * experts_num + experts_num
            + coords_channel * gating_feature + gating_feature
        )
        return int(param_count)
    
    # cal_features_ourmodel， 计算需要网络需要的features数（模型宽度）
    @staticmethod
    def calc_features_ourmodel(param_count, coords_channel, data_channel, expert_layers, experts_num, gating_feature,**kwargs):
        a = (expert_layers - 2) * experts_num
        b = (gating_feature+1+expert_layers-2+data_channel) * experts_num
        c = (-param_count + data_channel * experts_num + 3 * (gating_feature**2 + gating_feature) + gating_feature * experts_num + experts_num
            + coords_channel * gating_feature + gating_feature)
        if a == 0:
            features = round(-c / b)
        else:
            features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
        return features

class PosEncodingNeRF(nn.Module):
    def __init__(self, in_channel, frequencies=10):
        super().__init__()

        self.in_channel = in_channel
        self.frequencies = frequencies
        self.out_channel = in_channel + 2 * in_channel * self.frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_channel)
        coords_pos_enc = coords
        for i in range(self.frequencies):
            for j in range(self.in_channel):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * math.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * math.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_channel).squeeze(1)


class NeRF(nn.Module):
    """
    B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
    and R. Ng, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,”
    arXiv:2003.08934 [cs], Aug. 2020, Accessed: May 04, 2021. [Online].
    Available: http://arxiv.org/abs/2003.08934
    """

    def __init__(
        self,
        coords_channel=3,
        data_channel=1,
        frequencies=10,
        features=256,
        layers=5,
    ):
        super().__init__()
        self.positional_encoding = PosEncodingNeRF(
            in_channel=coords_channel, frequencies=frequencies
        )
        in_channel = self.positional_encoding.out_channel
        self.net = []
        self.net.append(
            nn.Sequential(nn.Linear(in_channel, features), nn.ReLU(inplace=True))
        )
        for i in range(layers - 2):
            self.net.append(
                nn.Sequential(nn.Linear(features, features), nn.ReLU(inplace=True))
            )
        self.net.append(nn.Sequential(nn.Linear(features, data_channel)))
        self.net = nn.ModuleList(self.net)

    def forward(self, coords):
        codings = self.positional_encoding(coords)
        output = codings
        for idx, model in enumerate(self.net):
            output = model(output)
        return output

    @staticmethod
    def calc_param_count(
        coords_channel, data_channel, features, frequencies, layers, **kwargs
    ):
        d = coords_channel + 2 * coords_channel * frequencies
        param_count = (
            d * features
            + features
            + (layers - 2) * (features**2 + features)
            + features * data_channel
            + data_channel
        )
        return int(param_count)

    @staticmethod
    def calc_features(
        param_count, coords_channel, data_channel, frequencies, layers, **kwargs
    ):
        d = coords_channel + 2 * coords_channel * frequencies
        a = layers - 2
        b = d + 1 + layers - 2 + data_channel
        c = -param_count + data_channel
        features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
        return features


def configure_optimizer(parameters, optimizer_opt) -> torch.optim.Optimizer:
    optimizer_opt = deepcopy(optimizer_opt)
    optimizer_name = optimizer_opt.pop("name")
    if optimizer_name == "Adam":
        Optimizer = torch.optim.Adam(parameters, **optimizer_opt)
    elif optimizer_name == "Adamax":
        Optimizer = torch.optim.Adamax(parameters, **optimizer_opt)
    elif optimizer_name == "SGD":
        Optimizer = torch.optim.SGD(parameters, **optimizer_opt)
    else:
        raise NotImplementedError
    return Optimizer

class NeRF_zcq(nn.Module):
    def __init__(
        self,
        coords_channel=3,
        data_channel=1,
        # frequencies=10,
        features=256,
        layers = 5,
    ):
        super().__init__()
        # self.positional_encoding = PosEncodingNeRF(
        #     in_channel=coords_channel, frequencies=frequencies
        # )
        # in_channel = self.positional_encoding.out_channel
        self.net = []
        # self.net.append(
        #     nn.Sequential(nn.Linear(in_channel, features), nn.ReLU(inplace=True))
        # )
        self.net.append(
            nn.Sequential(nn.Linear(coords_channel, features), nn.ReLU(inplace=True))
        )
        for i in range(layers - 2):
            self.net.append(
                nn.Sequential(nn.Linear(features, features), nn.ReLU(inplace=True))
            )
        self.net.append(nn.Sequential(nn.Linear(features, data_channel)))
        self.net = nn.ModuleList(self.net)

    def forward(self, feature):
        # codings = self.positional_encoding(coords)
        # output = codings
        for idx, model in enumerate(self.net):
            feature = model(feature)
        return feature

    @staticmethod
    def calc_param_count(
        coords_channel, data_channel, features, frequencies, layers, **kwargs
    ):
        d = coords_channel + 2 * coords_channel * frequencies
        param_count = (
            d * features
            + features
            + (layers - 2) * (features**2 + features)
            + features * data_channel
            + data_channel
        )
        return int(param_count)

    @staticmethod
    def calc_features(
        param_count, coords_channel, data_channel, frequencies, layers, **kwargs
    ):
        d = coords_channel + 2 * coords_channel * frequencies
        a = layers - 2
        b = d + 1 + layers - 2 + data_channel
        c = -param_count + data_channel
        features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
        return features


def configure_optimizer(parameters, optimizer_opt) -> torch.optim.Optimizer:
    optimizer_opt = deepcopy(optimizer_opt)
    optimizer_name = optimizer_opt.pop("name")
    if optimizer_name == "Adam":
        Optimizer = torch.optim.Adam(parameters, **optimizer_opt)
    elif optimizer_name == "Adamax":
        Optimizer = torch.optim.Adamax(parameters, **optimizer_opt)
    elif optimizer_name == "SGD":
        Optimizer = torch.optim.SGD(parameters, **optimizer_opt)
    else:
        raise NotImplementedError
    return Optimizer


def configure_lr_scheduler(optimizer, lr_scheduler_opt):
    lr_scheduler_opt = deepcopy(lr_scheduler_opt)
    lr_scheduler_name = lr_scheduler_opt.pop("name")
    if lr_scheduler_name == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, **lr_scheduler_opt
        )
    elif lr_scheduler_name == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **lr_scheduler_opt)
    elif lr_scheduler_name == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_opt)
    elif lr_scheduler_name == "None":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100000000000]
        )
    else:
        raise NotImplementedError
    return lr_scheduler


def l2_loss(gt, predicted, weight_map) -> torch.Tensor:
    loss = F.mse_loss(gt, predicted, reduction="none")
    loss = loss * weight_map
    loss = loss.mean()
    return loss

def l2_loss_2(gt, predicted) -> torch.Tensor:
    loss = F.mse_loss(gt, predicted, reduction="none")
    loss = loss
    loss = loss.mean()
    return loss

def load_model(model,model_path,device:str='cuda'):
    if hasattr(model, "net"):
        files = os.listdir(model_path)
        for file in files:
            file_path = os.path.join(model_path,file)
            with open(file_path, 'rb') as data_file:
                if 'weight' in file:
                    _,l,shape0,shape1 = file.split('-')
                    l,shape0,shape1 = int(l),int(shape0),int(shape1)
                    weight = np.array(struct.unpack('f'*shape0*shape1, data_file.read())).astype(np.float32).reshape(shape0,shape1)
                    weight = torch.tensor(weight).to(device)
                    with torch.no_grad():
                        model.net[l][0].weight.data = weight
                elif 'bias' in file:
                    _,l,length = file.split('-')
                    l,length = int(l),int(length)
                    bias = np.array(struct.unpack('f'*length, data_file.read())).astype(np.float32)
                    bias = torch.tensor(bias).to(device)
                    with torch.no_grad():
                        model.net[l][0].bias.data = bias
    else:
        model.load_state_dict(torch.load(model_path))
    return model

def save_model(model,save_path,devive:str='cuda'):
    if hasattr(model, "net"):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        for l in range(len(model.net)):
            weight = model.net[l][0].weight.data.to('cpu')
            bias = model.net[l][0].bias.data.to('cpu')
            # weight = copy.deepcopy(weight).to('cpu')
            # bias = copy.deepcopy(bias).to('cpu')
            weight_save_path = os.path.join(save_path,'weight-{}-{}-{}'.format(l,weight.shape[0],weight.shape[1]))
            weight = np.array(weight).reshape(-1)
            with open(weight_save_path, 'wb') as data_file:
                data_file.write(struct.pack('f'*len(weight), *weight))
            bias_save_path = os.path.join(save_path,'bias-{}-{}'.format(l,len(bias)))
            with open(bias_save_path, 'wb') as data_file:
                data_file.write(struct.pack('f'*len(bias), *bias))
            weight = model.net[l][0].weight.data.to(devive)
            bias = model.net[l][0].bias.data.to(devive)
    else:
        model = torch.save(model.state_dict(), save_path)

def CopyDir(old_dir,new_dir):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    files = os.listdir(old_dir)
    for file in files:
        old_path = os.path.join(old_dir,file)
        new_path = os.path.join(new_dir,file)
        shutil.copy(old_path, new_path)
        
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)