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
from random import shuffle
import time
from tkinter import E
from xml.sax.handler import feature_external_ges
from einops import rearrange
import numpy as np

from omegaconf import OmegaConf
import tifffile
import torch
from torch import _pin_memory, nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
    MoE
)
from utils.samplers import RandomPointSampler3D

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tempfile

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
    parser.add_argument("-g", type=str, default="0", help="gpu_id")
    #添加多gpu训练参数
    parser.add_argument('-b', '--batch_size', default=10000, type=int, metavar='N', help='number of batchsize')
    parser.add_argument('--init_method', default='env://', help='init method')
    parser.add_argument('-r', '--rank', default=1, type=int, help='rank of current process')
    parser.add_argument('--world_size', default=2, type=int, help='world size')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    config_path = os.path.abspath(args.c)
    
    ###########################
    
    #初始化进程环境
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.g = int(os.environ['LOCAL_RANK'])
    
    args.distributed = True
    torch.cuda.set_device(args.g)
    
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank, init_method='env://')
    
    dist.barrier()
   
    ###########################

    # 1. load config(yaml)
    config = OmegaConf.load(config_path)
    # output_dir = opj(opd(__file__), "outputs", config.output_dir_name + timestamp)
    output_dir = opj('/data/sci', 'outputs', config.output_dir_name + timestamp)
    rank = args.rank
    if dist.get_rank() == 0:
        os.makedirs(output_dir)
    print(f"All results wll be saved in {output_dir}")
    # OmegaConf.save(config, opj(output_dir, "config.yaml"))
    reproduc(config.seed)
    n_training_samples_upper_limit = config.n_training_samples_upper_limit
    n_random_training_samples_percent = config.n_random_training_samples_percent
    n_training_steps = config.n_training_steps
    # tblogger = SummaryWriter(output_dir)
    
    ###########################
    #准备数据
    if dist.get_rank() == 0:
        rank = args.rank
        device = torch.device(args.device)
        batch_size = args.batch_size
        
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
        
        # tifffile.imwrite(
        #     opj(output_dir, data_name + "_denoised" + data_extension),
        #     denoised_data,
        # )
        # normalize data
        sideinfos.normalized_min = config.data.normalized_min
        sideinfos.normalized_max = config.data.normalized_max
        normalized_data = normalize(denoised_data, sideinfos)
        # prepare coordinates
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
        print('ok!!!')
        # move data to device
        normalized_data = rearrange(torch.tensor(normalized_data, dtype=torch.float), "d h w c-> (d h w) c")
        coordinates = rearrange(torch.tensor(coordinates), "d h w c-> (d h w) c")
        data = torch.cat((coordinates, normalized_data), dim=1)
        
        dataset = torch.utils.data.TensorDataset(data[:,:3], data[:,3:])
    
    #切分数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)#确保每个进程处理不同数据，返回一个采样器
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last = False)#指定batch size
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=train_batch_sampler)
    
    ###########################
    # 3. prepare network
    
    # calculate network structure
    ideal_network_size_bytes = os.path.getsize(data_path) / config.compression_ratio
    #设置压缩比，来控制网络size
    ideal_network_parameters_count = ideal_network_size_bytes / 4.0
    n_network_features = MoE.calc_features_ourmodel(
        param_count=ideal_network_parameters_count, **config.network_structure
    )
    actual_network_parameters_count = MoE.calc_param_count_ourmodel(
        features=n_network_features, **config.network_structure
    )
    actual_network_size_bytes = actual_network_parameters_count * 4.0
    
    # initialize network
    network = MoE(ideal_network_parameters_count = ideal_network_parameters_count, **config.network_structure).to(device)
    # assert(
    #     get_nnmodule_param_count(network) == actual_network_parameters_count
    # ), "The calculated network structure mismatch the actual_network_parameters_count!"

    # #是否需要将gpu1模型的权重复制到其他模型上
    # checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # if rank == 0:
    #     torch.load(network.state_dict(), checkpoint_path)
        
    # dist.barrier()
    # network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # move network to device
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=args.g)

    ###########################
    # 5. prepare optimizer lr_scheduler
    optimizer = configure_optimizer(network.parameters(), config.optimizer)
    lr_scheduler = configure_lr_scheduler(optimizer, config.lr_scheduler)

    # checkpoint设置
    checkpoints = parse_checkpoints(config.checkpoints, n_training_steps)
    n_print_loss_interval = config.n_print_loss_interval
    print(f"Beginning optimization with {n_training_steps} training steps.")
    
    ###########################
    compression_time_seconds = 0
    compression_time_start = time.time() 
    print('training start!')
    #开始训练
    for steps in range(1, n_training_steps + 1):
        # if sampling_required:
        #     coords_batch, gt_batch = sampler.next()
        
        for idx, (coords, gt) in enumerate(data_loader):  
            optimizer.zero_grad()
            pred, aux_loss, index_v= network(coords)
            torch.autograd.set_detect_anomaly(True)
            loss = l2_loss_2(pred, gt) + 0.5*aux_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            dist.barrier()
            
        if steps % n_print_loss_interval == 0 and dist.get_rank() == 0:
            compression_time_end = time.time()
            compression_time_seconds += compression_time_end - compression_time_start
            print(
                f"#Steps:{steps} Loss:{loss.item()} ElapsedTime:{compression_time_seconds}s",
                flush=True,
            )
            # tblogger.add_scalar("loss", loss.item(), steps)
            compression_time_start = time.time()
        
        if steps in checkpoints and dist.get_rank() == 0:
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
    print("Finish!", flush=True)
    
    #python -m torch.distributed.launch --nproc_per_node=6 --use_env train_multigpu.py
    
        # 6. prepare sampler
    # sampling_required = True
    # if n_random_training_samples_percent == 0:
    #     if n_samples <= n_training_samples_upper_limit:
    #         sampling_required = False
    #     else:
    #         sampling_required = True
    #         n_random_training_samples = int(n_training_samples_upper_limit)
    # else:
    #     sampling_required = False
    #     n_random_training_samples = int(
    #         min(
    #             n_training_samples_upper_limit,
    #             n_random_training_samples_percent * n_samples,
    #         )
    #     )
    # if sampling_required:
    #     sampler = RandomPointSampler3D(
    #         coordinates, normalized_data, n_random_training_samples
    #     )
    # else:
    #     coords_batch = rearrange(coordinates, "d h w c-> (d h w) c")
    #     gt_batch = rearrange(normalized_data, "d h w c-> (d h w) c")
    #     # weight_map_batch = rearrange(weight_map, "d h w c-> (d h w) c")
    # if sampling_required:
    #     print(f"Use mini-batch training with batch-size={n_random_training_samples}")
    # else:
    #     print(f"Use batch training with batch-size={n_samples}")