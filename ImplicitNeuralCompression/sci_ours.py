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
from matplotlib.pyplot import scatter
import numpy as np

from omegaconf import OmegaConf
import tifffile
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *
from utils.metrics import calc_psnr, calc_ssim, get_folder_size, parse_checkpoints

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.networks import (
    SIREN,
    configure_lr_scheduler,
    configure_optimizer,
    get_nnmodule_param_count,
    l2_loss,
    l2_loss_2,
    load_model,
    save_model,
    SIREN_zcq,
    sine_init,
    first_layer_sine_init,
    MoE,
    model_structure
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="single task for datacompress")
    parser.add_argument(
        "-c",
        type=str,
        default=opj(opd(__file__), "config", "SingleExp", "zjc.yaml"),
        help="yaml file path",
    )
    parser.add_argument("-g", type=str, default="2,3,4,6", help="gpu index")
    args = parser.parse_args()
    config_path = os.path.abspath(args.c)
    
    ###########################
    # Make the gpu index used by CUDA_VISIBLE_DEVICES consistent with the gpu index shown in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Specify the gpu index to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
   
    ###########################
    # 1. load config
    config = OmegaConf.load(config_path)
    output_dir = opj('/data/sci', 'outputs', config.output_dir_name + timestamp)
    os.makedirs(output_dir)
    print(f"All results wll be saved in {output_dir}")
    OmegaConf.save(config, opj(output_dir, "config.yaml"))
    reproduc(config.seed)
    n_training_samples_upper_limit = config.n_training_samples_upper_limit
    n_random_training_samples_percent = config.n_random_training_samples_percent
    n_training_steps = config.n_training_steps
    # writer = SummaryWriter(output_dir)
    
    ###########################
    # 2. prepare data
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
    sideinfos = SideInfos3D()
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
    normalized_data = torch.tensor(normalized_data, dtype=torch.float)

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
    ###########################
    # 准备sampler
    sampling_required = True
    if n_random_training_samples_percent == 0:
        if n_samples <= n_training_samples_upper_limit:
            sampling_required = False
        else:
            sampling_required = True
            n_random_training_samples = int(n_training_samples_upper_limit)
    else:
        sampling_required = False
        n_random_training_samples = int(
            min(
                n_training_samples_upper_limit,
                n_random_training_samples_percent * n_samples,
            )
        )
    if sampling_required:
        sampler = RandomPointSampler3D(
            coordinates, normalized_data, n_random_training_samples
        )
    else:
        coords_batch = rearrange(coordinates, "d h w c-> (d h w) c")
        gt_batch = rearrange(normalized_data, "d h w c-> (d h w) c")
    if sampling_required:
        print(f"Use mini-batch training with batch-size={n_random_training_samples}")
    else:
        print(f"Use batch training with batch-size={n_samples}")
    
    # #准备data_loader
    # coords_batch = rearrange(coordinates, "d h w c-> (d h w) c")
    # gt_batch = rearrange(normalized_data, "d h w c-> (d h w) c")
    
    # data = torch.cat((coords_batch, gt_batch), dim=1)
    # dataset = torch.utils.data.TensorDataset(data[:,:3], data[:,3:])
    
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size= config.batch_size)
    
    ###########################
    # 3. prepare network
    # calculate network structure
    ideal_network_size_bytes = os.path.getsize(data_path) / config.compression_ratio
    ideal_network_parameters_count = ideal_network_size_bytes / 4.0
    n_network_features = MoE.calc_features_ourmodel(
        param_count=ideal_network_parameters_count, **config.network_structure
    )
    actual_network_parameters_count = MoE.calc_param_count_ourmodel(
        features=n_network_features, **config.network_structure
    )
    actual_network_size_bytes = actual_network_parameters_count * 4.0
    
    # initialize network
    network = MoE(n_network_features = n_network_features, **config.network_structure)
    # network = nn.DataParallel(network, device_ids=devices, dim=0)
    # assert(
    #     get_nnmodule_param_count(network) == actual_network_parameters_count
    # ), "The calculated network structure mismatch the actual_network_parameters_count!"

    # move network to device
    network.cuda()
    model_structure(network)
    print(n_network_features)
    
    ###########################
    # 5. prepare optimizer lr_scheduler
    optimizer = configure_optimizer(network.parameters(), config.optimizer)
    lr_scheduler = configure_lr_scheduler(optimizer, config.lr_scheduler)

    ###########################
    # 7. 准备checkpoint
    checkpoints = parse_checkpoints(config.checkpoints, n_training_steps)
    n_print_loss_interval = config.n_print_loss_interval
    print(f"Beginning optimization with {n_training_steps} training steps.")
    
    compression_time_seconds = 0
    compression_time_start = time.time()
    for steps in range(1, n_training_steps + 1):
        if sampling_required:
            coords_batch, gt_batch = sampler.next()
        coords, gt = coords_batch.to('cuda'), gt_batch.to('cuda')
        optimizer.zero_grad()
        pred, aux_loss, index_v = network(coords)
        torch.autograd.set_detect_anomaly(True)
        loss = l2_loss_2(gt, pred) + 0*aux_loss.mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
            
        if steps % n_print_loss_interval == 0:
            compression_time_end = time.time()
            compression_time_seconds += compression_time_end - compression_time_start
            # pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
            print(
                f"#Steps:{steps} Loss:{loss} ElapsedTime:{compression_time_seconds}s",
                flush=True,
            )
            # writer.add_scalar("loss", loss.item(), steps)
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
        
            # decompress data zjc:训完再解压缩
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
                    ], l , v= network(flattened_coords[start_sample_idx:end_sample_idx].cuda())
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

                # #visulization， save for every checkpoint, index_v in checkpoint
                # for i in range(index_v.shape[1]):
                #     c = torch.zeros_like(index_v)
                #     c[:, i] = index_v[:, i]
                #     # :coord_select --> [batch]
                #     # :unsqueeze --> [batch,1]
                #     coord_select = c[:,i]
                #     # coord_select = coord_select.unsqueeze(1)
                #     coord_select = coord_select.rearrange(coord_select, "(d h w) c -> d h w c",
                #     d=sideinfos.depth,
                #     h=sideinfos.height,
                #     w=sideinfos.width)
                #     # coord_select = coord_select.reshape(sideinfos.depth,sideinfos.height,sideinfos.width)
                #     coord_select = coord_select.cpu().numpy()
                #     original_data = tifffile.imread(data_path)
                #     expert = coord_select * original_data
                    
                #     # save visualization data
                #     visualization_save_dir = opj(output_dir, "checkpoints", f"steps_{steps}", 'visualization', f'expert_{i}')
                #     os.makedirs(visualization_save_dir, exist_ok=True)
                #     visualization_save_path = opj(
                #         visualization_save_dir,
                #         data_name + "_visualization" + data_extension,
                #     )
                #     tifffile.imwrite(visualization_save_path, expert)           
    print("Finish!", flush=True)
