# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm
import time
import torch
import argparse
from collections import deque
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import joycent.params as params
from joycent.model.tts_conformer_gstloss_whisper3_qwen2_facodec3_acc_rmllm import (
    GradTTSConformerGSTWhisper3Qwen2facodec3accrmllm,
)
from joycent.data_grl import TextMelSpeakerAccentQwenDataset, TextMelSpeakerAccentQwenBatchCollate
from joycent.utils import plot_tensor, save_plot
from joycent.text.zhdict import ZHDict
import os
from joycent.data_utils import DistributedBucketSampler
torch.autograd.set_detect_anomaly(False)

train_filelist_path = 'resources/Joycent/zh_all/train.txt'
valid_filelist_path = 'resources/Joycent/zh_all/valid.txt'
data_root = ''
cmudict_path = params.cmudict_path
zhdict_path = params.zhdict_path
add_blank = True
cln = True
n_spks = 222  # remove others
n_accents = 4 # 3 remove others
spk_emb_dim = params.spk_emb_dim

log_dir = 'logs/joycent'
n_epochs = params.n_epochs
params.batch_size = 32
out_size = params.out_size
learning_rate = 0.0001
random_seed = params.seed

nsymbols = ZHDict(zhdict_path).__len__() + 1 if add_blank == True else ZHDict(zhdict_path).__len__()
n_enc_channels = 256
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale


n_mels = params.n_mels
n_audio_ctx = params.n_audio_ctx
n_audio_state = params.n_audio_state
n_audio_head = params.n_audio_head
n_audio_layer = params.n_audio_layer
acc_layers = params.acc_layers
spk_layers = params.spk_layers
n_acc = params.n_acc
n_spk = params.n_spk
model_name = params.model_name
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
pretrained_model = ''
accent_embeddings_path = ''

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-filelist-path", default=train_filelist_path)
    parser.add_argument("--valid-filelist-path", default=valid_filelist_path)
    parser.add_argument(
        "--data-root",
        default=data_root,
        help="Root directory prepended to relative wav paths in TTS filelists",
    )
    parser.add_argument("--log-dir", default=log_dir)
    parser.add_argument("--pretrained-model", default=pretrained_model)
    parser.add_argument("--batch-size", type=int, default=params.batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--n-epochs", type=int, default=n_epochs)
    parser.add_argument("--n-spks", type=int, default=n_spks)
    parser.add_argument("--n-accents", type=int, default=n_accents)
    parser.add_argument("--master-port", default="60003")
    parser.add_argument("--accent-embeddings-path", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--profile-batches", type=int, default=5)
    parser.add_argument("--validation-samples", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many steps (0 means no limit; useful for profiling)")
    return parser.parse_args()


def apply_args(args):
    global train_filelist_path, valid_filelist_path, data_root
    global log_dir, pretrained_model, learning_rate, n_epochs, n_spks, n_accents, accent_embeddings_path
    train_filelist_path = args.train_filelist_path
    valid_filelist_path = args.valid_filelist_path
    data_root = args.data_root
    log_dir = args.log_dir
    pretrained_model = args.pretrained_model
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    n_spks = args.n_spks
    n_accents = args.n_accents
    accent_embeddings_path = args.accent_embeddings_path
    params.batch_size = args.batch_size
    os.environ['MASTER_PORT'] = args.master_port

def main(params, runtime_args):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.makedirs(log_dir, exist_ok=True)
    os.environ.setdefault('MASTER_PORT', '60003')

    print('Total batch size:', params.batch_size)
    print('Batch size per GPU :', params.batch_size // n_gpus)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, runtime_args))


def run(rank, n_gpus, runtime_args):
    apply_args(argparse.Namespace(**runtime_args))
    # Select the rank-local GPU before NCCL creates a CUDA context.  Otherwise
    # every spawned rank briefly initializes the default cuda:0 device.
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    if rank == 0:
        print('Set devices ...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if n_gpus > 1:
        device = torch.device("cuda:{:d}".format(rank))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print('Initializing logger...')
    
    logger = SummaryWriter(log_dir=log_dir) if rank == 0 else None
    
    if rank == 0:
        print('Initializing data loaders...')
    train_dataset = TextMelSpeakerAccentQwenDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max, zhdict_path, train=True, whisper_flag=False,
                                          data_root=data_root, accent_embeddings_path=accent_embeddings_path)
    batch_collate = TextMelSpeakerAccentQwenBatchCollate()
    train_sampler = DistributedBucketSampler(
        logger,
        train_dataset,
        params.batch_size // n_gpus,
        [0, 30, 50, 100, 200, 250, 300, 400, 500, 600, 700, 1200],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    loader = DataLoader(dataset=train_dataset,
                        collate_fn=batch_collate,
                        num_workers=runtime_args["num_workers"], shuffle=False, batch_sampler=train_sampler,
                        pin_memory=True, persistent_workers=runtime_args["num_workers"] > 0)
    
    if rank == 0 and runtime_args["validation_samples"] > 0:
        test_dataset = TextMelSpeakerAccentQwenDataset(valid_filelist_path, cmudict_path, add_blank,
                                            n_fft, n_feats, sample_rate, hop_length,
                                            win_length, f_min, f_max, zhdict_path, whisper_flag=True,
                                            data_root=data_root, accent_embeddings_path=accent_embeddings_path)
        test_sampler = DistributedSampler(dataset=test_dataset, num_replicas=n_gpus,
                                        rank=rank,
                                        shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=1,
                                    shuffle=False, batch_size=1,drop_last=True,
                                    pin_memory=False,sampler=test_sampler, collate_fn=TextMelSpeakerAccentQwenBatchCollate())            

    print('Initializing model...')
    model = GradTTSConformerGSTWhisper3Qwen2facodec3accrmllm(nsymbols, n_spks, spk_emb_dim, n_enc_channels,
                    filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale, 
                    n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, acc_layers, spk_layers, n_accents, n_spk, model_name,
                    acc_cln_layer=0, spk_cln_layer=5, spk_dec=False, acc_dec=False).cuda(rank)
    
    print("finish initializing model")
    # print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    # print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))
    # print('Number of GST parameters = %.2fm' % (sum(p.numel() for p in model.gst.parameters()) / 1e6))
    # print('Number of non-trainable parameters = %.2fm' % (sum(p.numel() for p in model.gst.model.parameters() if not p.requires_grad) / 1e6))
    # print('Trainable parameters:', [name for name, p in model.gst.model.named_parameters() if p.requires_grad])



    print('Initializing optimizer...')
    # 优化器 1：主要模型（encoder/decoder/acc head）
    main_params = []
    main_params += list(model.encoder.parameters())
    main_params += list(model.decoder.parameters())

    # if hasattr(model, "spk_emb"):
    #     main_params += list(model.spk_emb.parameters())

    # if hasattr(model, "acc_emb"):
    #     main_params += list(model.acc_emb.parameters())

    # if hasattr(model, "accent_embedding"):
    #     main_params += list(model.accent_embedding.parameters())

    # if hasattr(model, "gst") and hasattr(model.gst, "parameters"):
    #     main_params += list(model.gst.parameters())

    # llm related
    # main_params += list(model.llm_projector.parameters()) 
    # main_params += list(model.gate_layer.parameters())
    main_params += list(model.spk_projector.parameters())
    main_params += list(model.acc_projector.parameters())
    
    optimizer_main = torch.optim.Adam(params=main_params, lr=learning_rate)

    # 优化器 2：说话人对抗分类器
    # optimizer_spk = torch.optim.Adam(model.spk_grl.parameters(), lr=learning_rate)

    # optimizer_E = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # optimizer_D = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    print('finish Initializing optimizer...')
    # net_d = DDP(net_d, device_ids=[rank],find_unused_parameters=True)
    # resume
    try:
        if not pretrained_model:
            raise FileNotFoundError("No pretrained model was provided.")
        checkpoint = torch.load(pretrained_model)
        
        # Restore the model and optimizer state
        model.load_state_dict(checkpoint['model'])
        optimizer_state = checkpoint.get('optim_main') or checkpoint.get('optim')
        if optimizer_state is not None:
            optimizer_main.load_state_dict(optimizer_state)

        # Optionally, restore the epoch and loss if needed
        start_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        iteration = 0
        if optimizer_main.state_dict()['state']:
            first_state = next(iter(optimizer_main.state_dict()['state'].values()))
            step = first_state.get('step', 0)
            iteration = int(step.item()) if hasattr(step, 'item') else int(step)
        # iteration = 53495 
    except:
        start_epoch = 0
        iteration = 0
    model = DDP(model, device_ids=[rank])
    # print("finish initializing DDP model")
    if rank == 0 and runtime_args["validation_samples"] > 0:
        
        print('Logging test batch...')
        for i, item in enumerate(test_loader):
            mel, spk = item['y'], item['spk']
            i = int(spk.cpu())
            logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                            global_step=0, dataformats='HWC')
            save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')
            if i + 1 >= runtime_args["validation_samples"]:
                break

    print('Start training...', len(loader), "batches")
    total_steps = n_epochs * len(loader)
    global_step = 0
    start_time = time.time()
    recent_step_times = deque(maxlen=200)
    stop_training = False
    for epoch in range(start_epoch + 1, n_epochs + 1):
        if rank == 0 and runtime_args["validation_samples"] > 0:
            model.eval()
            print('Synthesis...')
            with torch.no_grad():
                for i, item in enumerate(test_loader):
                    # print(item)
                    x = item['x'].to(torch.long).cuda(rank)
                    x_lengths = torch.LongTensor([x.shape[-1]]).cuda(rank)
                    y, y_lengths = item['y'].cuda(rank), item['y_lengths'].cuda(rank)
                    spk = item['spk'].to(torch.long).cuda(rank)
                    acc = item['acc'].to(torch.long).cuda(rank)
                    # llm_embs = item['llm_embs'].cuda(rank)
                    filepath = item['filepath']
                    spk_embs = item['spk_embs'].cuda(rank)
                    acc_embs = item['acc_embs'].cuda(rank)
                    i = int(spk.cpu())
                    
                    y_enc, y_dec, attn = model(x, x_lengths, y, y_lengths, spk_embs, acc_embs, n_timesteps=10, spk=spk, acc=acc)
                    logger.add_image(f'image_{i}/generated_enc',
                                    plot_tensor(y_enc.squeeze().cpu()),
                                    global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/generated_dec',
                                    plot_tensor(y_dec.squeeze().cpu()),
                                    global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/alignment',
                                    plot_tensor(attn.squeeze().cpu()),
                                    global_step=iteration, dataformats='HWC')
                    save_plot(y_enc.squeeze().cpu(), 
                            f'{log_dir}/generated_enc_{i}.png')
                    save_plot(y_dec.squeeze().cpu(), 
                            f'{log_dir}/generated_dec_{i}.png')
                    save_plot(attn.squeeze().cpu(), 
                            f'{log_dir}/alignment_{i}.png')
                    if i + 1 >= runtime_args["validation_samples"]:
                        break

                    
                    # y_dec = y_dec.squeeze(0).transpose(0, 1).cpu().numpy()
                    
                    # write_hdf5(
                    #     os.path.join(log_dir, f"{filepath.split('/')[-1].split('.')[0]}.h5"),
                    #     "feats",
                    #     y_dec.astype(np.float32),
                    # )
                    # audio = torch.rand(y_dec.shape[0] * 256)
                    # write_hdf5(
                    #     os.path.join(log_dir, f"{filepath.split('/')[-1].split('.')[0]}.h5"),
                    #     "wave",
                    #     audio.cpu().numpy().astype(np.float32),
                    # )

                    # infer_mel_to_audio(log_dir)


        model.train()
        
        loader.batch_sampler.set_epoch(epoch)
        dur_losses = []
        prior_losses = []
        diff_losses = []
        
        previous_step_done = time.perf_counter()
        for i, batch in enumerate(loader):
                data_ready = time.perf_counter()
                dataloader_ms = (data_ready - previous_step_done) * 1000
                global_step += 1

                x = batch['x'].cuda(rank, non_blocking=True)
                x_lengths = batch['x_lengths'].cuda(rank, non_blocking=True)
                y = batch['y'].cuda(rank, non_blocking=True)
                y_lengths = batch['y_lengths'].cuda(rank, non_blocking=True)
                file = batch['filepath']
                spk_embs = batch['spk_embs'].cuda(rank, non_blocking=True)
                acc_embs = batch['acc_embs'].cuda(rank, non_blocking=True)
                do_profile = i < runtime_args["profile_batches"]
                if do_profile:
                    torch.cuda.synchronize(rank)
                transfer_done = time.perf_counter()
                result = model(x, x_lengths, y, y_lengths, spk_embs, acc_embs,
                               training_loss=True, out_size=out_size, profile=do_profile)
                if do_profile:
                    dur_loss, prior_loss, diff_loss, forward_timings = result
                else:
                    dur_loss, prior_loss, diff_loss = result

                optimizer_main.zero_grad()
                loss = sum([dur_loss, prior_loss, diff_loss])
                backward_start = time.perf_counter()
                loss.backward()
                if do_profile:
                    torch.cuda.synchronize(rank)
                    backward_ms = (time.perf_counter() - backward_start) * 1000

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.encoder.parameters(), 
                                                            max_norm=0.25)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(), 
                                                            max_norm=0.25) # 0.25 
                 
                optimizer_main.step()

                if do_profile and rank == 0:
                    torch.cuda.synchronize(rank)
                    step_total_ms = (time.perf_counter() - previous_step_done) * 1000
                    timing_text = ", ".join(f"{key}={value:.2f}ms" for key, value in forward_timings.items())
                    print(f"PROFILE batch={i}: dataloader={dataloader_ms:.2f}ms, h2d={(transfer_done-data_ready)*1000:.2f}ms, "
                          f"{timing_text}, backward={backward_ms:.2f}ms, step_total={step_total_ms:.2f}ms", flush=True)

                if rank == 0:
                    logger.add_scalar('training/duration_loss', dur_loss,
                                    global_step=iteration)
                    logger.add_scalar('training/prior_loss', prior_loss,
                                    global_step=iteration)
                    logger.add_scalar('training/diffusion_loss', diff_loss,
                                    global_step=iteration)
                    
                    logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                    global_step=iteration)
                    logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                    global_step=iteration)
                    
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}, lr: {learning_rate}'
                    
                    if iteration % runtime_args["log_interval"] == 0:
                        print(msg, flush=True)
                    dur_losses.append(dur_loss.item())
                    prior_losses.append(prior_loss.item())
                    diff_losses.append(diff_loss.item())
                 
                iteration += 1
                step_finished = time.perf_counter()
                recent_step_times.append(step_finished - previous_step_done)
                previous_step_done = time.perf_counter()
                if runtime_args["max_steps"] and global_step >= runtime_args["max_steps"]:
                    stop_training = True
                    break
                if rank == 0 and global_step % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_step = sum(recent_step_times) / len(recent_step_times)
                    remaining_steps = total_steps - global_step
                    eta = remaining_steps * avg_time_per_step

                    print(f"Step {global_step}/{total_steps} "
                        f"Elapsed: {elapsed/3600:.2f} hours "
                        f"Recent {len(recent_step_times)} steps: {avg_time_per_step:.3f} s/step "
                        f"ETA: {eta/86400:.2f} days")

        if stop_training:
            break

        if rank == 0:
            msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
            msg += '| prior loss = %.3f ' % np.mean(prior_losses)
            msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
             
            with open(f'{log_dir}/train.log', 'a') as f:
                f.write(msg)

            if epoch % params.save_every > 0:
                continue
            
            ckpt = {
                'epoch': epoch, 
                'model': model.module.state_dict(),
                # 'optim_spk': optimizer_spk.state_dict(),
                'optim_main': optimizer_main.state_dict(),
            }
            torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")

    if logger is not None:
        logger.close()
    dist.destroy_process_group()




if __name__ == "__main__":
    args = parse_args()
    apply_args(args)
    main(params, vars(args))
    
