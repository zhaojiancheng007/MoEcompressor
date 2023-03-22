from cmath import exp
import csv
from curses import noecho
from dataclasses import dataclass
from datetime import datetime
import math
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import argparse
import time
from tkinter import E
from xml.sax.handler import feature_external_ges
from einops import rearrange
import numpy as np

from omegaconf import OmegaConf
import tifffile
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *
from utils.metrics import calc_psnr, calc_ssim, get_folder_size, parse_checkpoints

from utils.networks import (
    SIREN,
    configure_lr_scheduler,
    configure_optimizer,
    get_nnmodule_param_count,
    l2_loss,
    load_model,
    save_model,
    SIREN_zcq,
    sine_init,
    first_layer_sine_init
)
from utils.samplers import RandomPointSampler3D


EXPERIMENTAL_CONDITIONS = ["data_name", "data_type", "data_shape", "actual_ratio"]
METRICS = [
    "psnr",
    "ssim",
    "compression_time_seconds",
    "decompression_time_seconds",
    "original_data_path",
    "decompressed_data_path",
]
EXPERIMENTAL_RESULTS_KEYS = (
    ["algorithm_name", "exp_time"] + EXPERIMENTAL_CONDITIONS + METRICS + ["config_path"]
)
timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S.%f")[:-3]

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
            
class MoE(nn.Module):
    def __init__(
        self,
        ideal_network_parameters_count,
        coords_channel=3,
        bandwidth=45,
        experts_num = 6,
        gating_feature = 16,
        topk = 1,
        gating_layers = 4,
        expert_layers = 5,
        **kwargs,
    ):
        super().__init__()
        self.topk = topk
        n_network_features = SIREN.calc_features(
            param_count=ideal_network_parameters_count//experts_num, **config.network_structure,
            )
        # actual_network_parameters_count = SIREN.calc_param_count(
        #     features=n_network_features, **config.network_structure
        #     )
        # actual_network_size_bytes = actual_network_parameters_count * 4.0
        
        # initialize network
        self.gate = []
        self.expert_num = experts_num
        self.first_linear = nn.Sequential(nn.Linear(coords_channel, gating_feature))
        
        ### Gating Network ###
        self.gate = []
        for i in range(gating_layers-2):
            self.gate.append(nn.Sequential(nn.Linear(gating_feature, gating_feature), nn.ReLU()))
        self.gate.append(nn.Sequential(nn.Linear(gating_feature, gating_feature), nn.LayerNorm([gating_feature])))
        self.gate.append(nn.Sequential(nn.Linear(gating_feature, experts_num)))
        self.gate.append(nn.Sequential(nn.Softmax()))
        self.gate = nn.Sequential(*self.gate)
        
        ### Experts Network ###
        self.expert = []
        for i in range(experts_num):
            freq = bandwidth/2.0 + i*bandwidth
            self.expert.append(nn.Sequential( \
                SIREN_zcq(coords_channel = gating_feature, layers=expert_layers, \
                features= n_network_features, data_channel = 1, w0 = freq, output_act=False)))
            # self.expert[i].cuda()
        
        self.expert = nn.ModuleList(self.expert)
        self.expert.cuda()
        self.first_linear.apply(first_layer_sine_init)
        self.gate.apply(gate_init)

    def forward(self, coords):
        feature = self.first_linear(coords)
        index = self.gate(feature)
        
        # #### Solution 1 ####
        # index = torch.topk(index, self.topk).indices
        # for i in range(len(index)):
        #     if i == 0:
        #         temp = 0.0
        #         for m in index[i]:
        #             temp += self.expert[m](feature[i])
        #         output = torch.tensor([[temp]])
        #     else:
        #         temp = 0.0
        #         for m in index[i]:
        #             temp += self.expert[m](feature[i])
        #         print("i =",i, ",   value =", temp.data)
        #         output = torch.cat((output, torch.tensor([[temp]])), dim = 0)
        #         1
           
        #### Solution 2 ####
        x = None
        for idx, m in enumerate(self.expert):
            if x == None:
                x = m(feature)
            else:
                x = torch.cat((x, m(feature)), dim = 1)
        # print(torch.topk(index, self.topk)[1])
        indices_to_remove = ~(index < torch.topk(index, self.topk)[0][..., -1, None])    
        output = torch.zeros_like(x)
        output.masked_scatter_(indices_to_remove, x)
        output = torch.sum(output, dim = 1, keepdim=True)
        
        # output = torch.mul(x,index)
        c = (indices_to_remove == True).sum(dim = 0)
        c = c*1.0
        m = index.sum(dim = 0)
        
        N = x.shape[0]
        n = self.expert_num
        aux_loss = n/N/N*(torch.dot(c,m))
        
        q = torch.ones_like(m)
        q = q*(N/n)
        
        kld_loss = torch.sum(m*torch.log(m) - m*torch.log(q)) /N/n
        
        return output, kld_loss


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="single task for datacompress")
    parser.add_argument(
        "-c",
        type=str,
        default=opj(opd(__file__), "config", "SingleExp", "zcq.yaml"),
        help="yaml file path",
    )
    parser.add_argument("-g", type=str, default="6", help="gpu index")
    args = parser.parse_args()
    config_path = os.path.abspath(args.c)
    # Make the gpu index used by CUDA_VISIBLE_DEVICES consistent with the gpu index shown in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Specify the gpu index to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    ###########################
    # 1. load config
    config = OmegaConf.load(config_path)
    output_dir = opj(opd(__file__), "outputs", config.output_dir_name + timestamp)
    os.makedirs(output_dir)
    print(f"All results wll be saved in {output_dir}")
    OmegaConf.save(config, opj(output_dir, "config.yaml"))
    reproduc(config.seed)
    n_training_samples_upper_limit = config.n_training_samples_upper_limit
    n_random_training_samples_percent = config.n_random_training_samples_percent
    n_training_steps = config.n_training_steps
    tblogger = SummaryWriter(output_dir)
    ###########################
    # 2. prepare data, weight_map
    sideinfos = SideInfos3D()
    # parse name and extension
    data_path = config.data.path
    data_name = ops(opb(data_path))[0]
    data_extension = ops(opb(data_path))[-1]
    # read original data
    data = tifffile.imread(data_path)
    if len(data.shape) == 3:
        data = data[..., None]
    assert (
        len(data.shape) == 4
    ), "Only DHWC data is allowed. Current data shape is {}.".format(data.shape)
    data_shape = ",".join([str(i) for i in data.shape])
    sideinfos.depth, sideinfos.height, sideinfos.width, _ = data.shape
    n_samples = sideinfos.depth * sideinfos.width * sideinfos.height
    # denoise data
    denoised_data = denoise(data, config.data.denoise_level, config.data.denoise_close)
    tifffile.imwrite(
        opj(output_dir, data_name + "_denoised" + data_extension),
        denoised_data,
    )
    # normalize data
    sideinfos.normalized_min = config.data.normalized_min
    sideinfos.normalized_max = config.data.normalized_max
    normalized_data = normalize(denoised_data, sideinfos)
    # move data to device
    normalized_data = torch.tensor(normalized_data, dtype=torch.float, device="cuda")
    # generate weight_map
    weight_map = generate_weight_map(denoised_data, config.data.weight_map_rules)
    # move weight_map to device
    weight_map = torch.tensor(weight_map, dtype=torch.float, device="cuda")
    ###########################
    # 3. prepare network
    # calculate network structure
    ideal_network_size_bytes = os.path.getsize(data_path) / config.compression_ratio
    ideal_network_parameters_count = ideal_network_size_bytes / 4.0
    # n_network_features = SIREN.calc_features(
    #     param_count=ideal_network_parameters_count, **config.network_structure
    # )
    # actual_network_parameters_count = SIREN.calc_param_count(
    #     features=n_network_features, **config.network_structure
    # )
    # actual_network_size_bytes = actual_network_parameters_count * 4.0
    # initialize network
    # network = SIREN(features=n_network_features, **config.network_structure)
    # network = MoE(ideal_network_parameters_count = ideal_network_parameters_count, bandwidth=30, experts_num = 4, **config.network_structure)
    network = MoE(ideal_network_parameters_count = ideal_network_parameters_count, **config.network_structure)
    # assert(
    #     get_nnmodule_param_count(network) == actual_network_parameters_count
    # ), "The calculated network structure mismatch the actual_network_parameters_count!"
    # (optional) load pretrained network
    if config.pretrained_network_path is not None:
        load_model(network, config.pretrained_network_path, "cuda")
    # move network to device
    network.cuda()
    model_structure(network)
    # print(network)
    ###########################
    # 4. prepare coordinates
    # shape:(d*h*w,3)
    coord_normalized_min = config.coord_normalized_min
    coord_normalized_max = config.coord_normalized_max
    coordinates = torch.stack(
        torch.meshgrid(
            torch.linspace(coord_normalized_min, coord_normalized_max, sideinfos.depth),
            torch.linspace(
                coord_normalized_min, coord_normalized_max, sideinfos.height
            ),
            torch.linspace(coord_normalized_min, coord_normalized_max, sideinfos.width),
            indexing="ij",
        ),
        axis=-1,
    )
    coordinates = coordinates.cuda()
    ###########################
    # 5. prepare optimizer lr_scheduler
    optimizer = configure_optimizer(network.parameters(), config.optimizer)
    lr_scheduler = configure_lr_scheduler(optimizer, config.lr_scheduler)
    ###########################
    # 6. prepare sampler
    sampling_required = True
    if n_random_training_samples_percent == 0:
        if n_samples <= n_training_samples_upper_limit:
            sampling_required = False
        else:
            sampling_required = True
            n_random_training_samples = int(n_training_samples_upper_limit)
    else:
        sampling_required = True
        n_random_training_samples = int(
            min(
                n_training_samples_upper_limit,
                n_random_training_samples_percent * n_samples,
            )
        )
    if sampling_required:
        sampler = RandomPointSampler3D(
            coordinates, normalized_data, weight_map, n_random_training_samples
        )
    else:
        coords_batch = rearrange(coordinates, "d h w c-> (d h w) c")
        gt_batch = rearrange(normalized_data, "d h w c-> (d h w) c")
        weight_map_batch = rearrange(weight_map, "d h w c-> (d h w) c")
    if sampling_required:
        print(f"Use mini-batch training with batch-size={n_random_training_samples}")
    else:
        print(f"Use batch training with batch-size={n_samples}")
    ###########################
    # 7. optimizing
    checkpoints = parse_checkpoints(config.checkpoints, n_training_steps)
    n_print_loss_interval = config.n_print_loss_interval
    print(f"Beginning optimization with {n_training_steps} training steps.")
    # pbar = trange(1, n_training_steps + 1, desc="Compressing", file=sys.stdout)
    
    compression_time_seconds = 0
    compression_time_start = time.time()
    
    
    for steps in range(1, n_training_steps + 1):
        if sampling_required:
            coords_batch, gt_batch, weight_map_batch = sampler.next()
        optimizer.zero_grad()
            
        predicted_batch, aux_loss = network(coords_batch)
        torch.autograd.set_detect_anomaly(True)
        loss = l2_loss(predicted_batch, gt_batch, weight_map_batch) + 0.5*aux_loss
        loss.backward()
        # print(network.first_linear[0].weight.grad)
        optimizer.step()
        lr_scheduler.step()
        if steps % n_print_loss_interval == 0:
            compression_time_end = time.time()
            compression_time_seconds += compression_time_end - compression_time_start
            # pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
            print(
                f"#Steps:{steps} Loss:{loss.item()} ElapsedTime:{compression_time_seconds}s",
                flush=True,
            )
            tblogger.add_scalar("loss", loss.item(), steps)
            compression_time_start = time.time()
        
        if steps in checkpoints:
            compression_time_end = time.time()
            compression_time_seconds += compression_time_end - compression_time_start
            # save network and evaluate performance
            curr_steps_dir = opj(output_dir, "checkpoints", f"steps_{steps}")
            os.makedirs(curr_steps_dir)
            compressed_data_save_dir = opj(curr_steps_dir, "compressed")
            os.makedirs(compressed_data_save_dir)
            network_parameters_save_dir = opj(
                compressed_data_save_dir, "network_parameters"
            )
            sideinfos_save_path = opj(compressed_data_save_dir, "sideinfos.yaml")
            OmegaConf.save(sideinfos.__dict__, sideinfos_save_path)
            save_model(network, network_parameters_save_dir, "cuda")
            
            
            
            # decompress data
            with torch.no_grad():
                flattened_coords = rearrange(coordinates, "d h w c-> (d h w) c")
                flattened_decompressed_data = torch.zeros(
                    (n_samples, 1),
                    device="cuda",
                )
                n_inference_batch_size = config.n_inference_batch_size
                n_inference_batchs = math.ceil(n_samples / n_inference_batch_size)
                decompression_time_start = time.time()
                for batch_idx in range(n_inference_batchs):
                    start_sample_idx = batch_idx * n_inference_batch_size
                    end_sample_idx = min(
                        (batch_idx + 1) * n_inference_batch_size, n_samples
                    )
                    flattened_decompressed_data[
                        start_sample_idx:end_sample_idx
                    ], l = network(flattened_coords[start_sample_idx:end_sample_idx])
                decompression_time_end = time.time()
                decompression_time_seconds = (
                    decompression_time_end - decompression_time_start
                )
                decompressed_data = rearrange(
                    flattened_decompressed_data,
                    "(d h w) c -> d h w c",
                    d=sideinfos.depth,
                    h=sideinfos.height,
                    w=sideinfos.width,
                )
                decompressed_data = decompressed_data.cpu().numpy()
                decompressed_data = inv_normalize(decompressed_data, sideinfos)
                
                
                
                
            # save decompressed data
            decompressed_data_save_dir = opj(curr_steps_dir, "decompressed")
            os.makedirs(decompressed_data_save_dir)
            decompressed_data_save_path = opj(
                decompressed_data_save_dir,
                data_name + "_decompressed" + data_extension,
            )
            tifffile.imwrite(decompressed_data_save_path, decompressed_data)
            
            
            
            
            # calculate metrics
            psnr = calc_psnr(data[..., 0], decompressed_data[..., 0])
            ssim = calc_ssim(data[..., 0], decompressed_data[..., 0])
            print("ssim:", ssim, ", psnr:", psnr)
            
            
            
            # record results
            results = {k: None for k in EXPERIMENTAL_RESULTS_KEYS}
            results["algorithm_name"] = "SCI"
            results["exp_time"] = timestamp
            results["original_data_path"] = data_path
            results["config_path"] = config_path
            results["decompressed_data_path"] = decompressed_data_save_path
            results["data_name"] = data_name
            results["data_type"] = config.data.get("type")
            results["data_shape"] = data_shape
            results["actual_ratio"] = os.path.getsize(data_path) / get_folder_size(
                network_parameters_save_dir
            )
            results["psnr"] = psnr
            results["ssim"] = ssim
            results["compression_time_seconds"] = compression_time_seconds
            results["decompression_time_seconds"] = decompression_time_seconds
            csv_path = os.path.join(output_dir, "results.csv")
            if not os.path.exists(csv_path):
                f = open(csv_path, "a")
                csv_writer = csv.writer(f, dialect="excel")
                csv_writer.writerow(results.keys())
            row = [results[key] for key in results.keys()]
            csv_writer.writerow(row)
            f.flush()
            compression_time_start = time.time()
            
            # # visualization 
            # flattened_gating_volumn = torch.zeros(
            #         (n_samples, 1),
            #         device="cuda",
            #     )
            # for batch_idx in range(n_inference_batchs):
            #     start_sample_idx = batch_idx * n_inference_batch_size
            #     end_sample_idx = min(
            #         (batch_idx + 1) * n_inference_batch_size, n_samples
            #     )
            #     flattened_gating_volumn[
            #         start_sample_idx:end_sample_idx
            #     ] = network.gate(flattened_coords[start_sample_idx:end_sample_idx])
            
            
            
    print("Finish!", flush=True)
