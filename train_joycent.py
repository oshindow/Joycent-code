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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params as params
from model import GradTTSGST, GradTTSConformer, GradTTSConformerGSTWhisper3Qwen2facodec3accrmllm
from data_grl import TextMelSpeakerDataset, TextMelSpeakerBatchCollate, TextMelSpeakerAccentQwenDataset, TextMelSpeakerAccentQwenBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols
from text.zhdict import ZHDict
import os
from data_utils import DistributedBucketSampler
from optimizer import ScheduledOptim
torch.autograd.set_detect_anomaly(True)

train_filelist_path = 'resources/filelists/zh_all/train.dedup.txt.rmSSB0342.rmSSB1567'
# train_filelist_path = 'resources/whisAID/zh_all/train.csv'
valid_filelist_path = 'resources/filelists/zh_all/valid.txt'
cmudict_path = params.cmudict_path
zhdict_path = params.zhdict_path
add_blank = True
cln = True
n_spks = 222  # remove others
n_accents = 4 # 3 remove others
spk_emb_dim = params.spk_emb_dim

log_dir = '/data2/xintong/gradtts/logs/joycent_e1'
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
from utils import write_hdf5, read_hdf5 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# import sys
# sys.path.append('/home/xintong/accent_tts_server/ParallelWaveGAN/')
from ParallelWaveGAN.parallel_wavegan.datasets import (
    AudioDataset,
    AudioSCPDataset,
    MelDataset,
    MelF0ExcitationDataset,
    MelSCPDataset,
)
from torch.nn import functional as F
pretrained_model = ''
n_warm_up_step = 40000

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def infer_mel_to_audio(dumpdir):

    
    mel_query = "*.h5"
    mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA

    dataset = MelDataset(
        dumpdir,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn,
        return_utt_id=True,
    )
    
    # print("infer mel to audio:", len(dataset), dumpdir)
    # logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # start generation
    total_rtf = 0.0
    with torch.no_grad():
        for idx, items in enumerate(dataset):
            # if not use_f0_and_excitation:
            utt_id, c = items
            f0, excitation = None, None
            # print(utt_id, c)
            # else:
            #     utt_id, c, f0, excitation = items
            batch = dict(normalize_before=False)
            if c is not None:
                c = torch.tensor(c, dtype=torch.float).to('cuda:0')
                batch.update(c=c)
            if f0 is not None:
                f0 = torch.tensor(f0, dtype=torch.float).to('cuda:0')
                batch.update(f0=f0)
            if excitation is not None:
                excitation = torch.tensor(excitation, dtype=torch.float).to('cuda:0')
                batch.update(excitation=excitation)
            # start = time.time()
            y = vocoder.inference(**batch).view(-1)
            # print(config["sampling_rate"])
            # rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            # pbar.set_postfix({"RTF": rtf})
            # total_rtf += rtf

            # save as PCM 16 bit wav file
            print(os.path.join(dumpdir, f"{utt_id}_gen.wav"))
            sf.write(
                os.path.join(dumpdir, f"{utt_id}_gen.wav"),
                y.cpu().numpy(),
                config["sampling_rate"],
                "PCM_16",
            )

def main(params):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '60003'

    print('Total batch size:', params.batch_size)
    params.batch_size = params.batch_size // n_gpus
    print('Batch size per GPU :', params.batch_size)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus,))


def run(rank, n_gpus):
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
    
    logger = SummaryWriter(log_dir=log_dir)
    
    if rank == 0:
        print('Initializing data loaders...')
    train_dataset = TextMelSpeakerAccentQwenDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max, zhdict_path, train=True, whisper_flag=True)
    batch_collate = TextMelSpeakerAccentQwenBatchCollate()
    train_sampler = DistributedBucketSampler(
        logger,
        train_dataset,
        params.batch_size // n_gpus,
        [30, 50, 100, 200, 250, 300,400,500,600,700],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    loader = DataLoader(dataset=train_dataset,
                        collate_fn=batch_collate,
                        num_workers=8, shuffle=False, batch_sampler=train_sampler)
    
    if rank == 0:
        test_dataset = TextMelSpeakerAccentQwenDataset(valid_filelist_path, cmudict_path, add_blank,
                                            n_fft, n_feats, sample_rate, hop_length,
                                            win_length, f_min, f_max, zhdict_path, whisper_flag=True)
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
        checkpoint = torch.load(pretrained_model)
        
        # Restore the model and optimizer state
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])

        # Optionally, restore the epoch and loss if needed
        start_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        iteration = optimizer.state_dict()['state'][0]['step'].int().item()
        # iteration = 53495 
    except:
        start_epoch = 0
        iteration = 0
    # scheduler = ScheduledOptim(optimizer, n_feats, n_warm_up_step, iteration)
    model = DDP(model, device_ids=[rank])
    # print("finish initializing DDP model")
    if rank == 0:
        
        print('Logging test batch...')
        for i, item in enumerate(test_loader):
            mel, spk = item['y'], item['spk']
            i = int(spk.cpu())
            logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                            global_step=0, dataformats='HWC')
            save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    print('Start training...', len(loader), "batches")
    total_steps = n_epochs * len(loader)
    global_step = 0
    start_time = time.time()
    for epoch in range(start_epoch + 1, n_epochs + 1):
        if rank == 0:
            model.eval()
            print('Synthesis...')
            with torch.no_grad():
                for i, item in enumerate(test_loader):
                    # print(item)
                    x = item['x'].to(torch.long).cuda(rank)
                    x_lengths = torch.LongTensor([x.shape[-1]]).cuda(rank)
                    y, y_lengths = item['y_prompt'].cuda(rank), torch.LongTensor([item['y_prompt'].shape[-1]]).cuda(rank)
                    spk = item['spk'].to(torch.long).cuda(rank)
                    acc = item['acc'].to(torch.long).cuda(rank)
                    # llm_embs = item['llm_embs'].cuda(rank)
                    filepath = item['filepath']
                    spk_embs = item['spk_embs'].cuda(rank)
                    acc_embs = item['acc_embs'].cuda(rank)
                    i = int(spk.cpu())
                    
                    y_enc, y_dec, attn = model(x, x_lengths, y, y_lengths, spk_embs, acc_embs, n_timesteps=50, spk=spk, acc=acc)
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
        
        for i, batch in enumerate(loader):
                 
                model.zero_grad()
                global_step += 1
                 
                x, x_lengths = batch['x'].cuda(rank), batch['x_lengths'].cuda(rank)
                y, y_lengths = batch['y'].cuda(rank), batch['y_lengths'].cuda(rank)
                y_prompt = batch['y_prompt'].cuda(rank)
                spk = batch['spk'].cuda(rank)
                acc = batch['acc'].cuda(rank)
                file = batch['filepath']
                spk_embs = batch['spk_embs'].cuda(rank)
                acc_embs = batch['acc_embs'].cuda(rank)
                dur_loss, prior_loss, diff_loss = model.module.compute_loss(
                                        x, x_lengths, # go gradtts compute loss
                                        y_prompt, y, y_lengths, spk_embs, acc_embs,
                                        out_size=out_size)
                 
                optimizer_main.zero_grad()
                 
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.encoder.parameters(), 
                                                            max_norm=0.25)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(), 
                                                            max_norm=0.25) # 0.25 
                 
                optimizer_main.step()
                
                 
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
                    
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}, lr: {learning_rate}, {file}'
                    
                    print(msg)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                 
                iteration += 1
                if global_step % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_step = elapsed / global_step
                    remaining_steps = total_steps - global_step
                    eta = remaining_steps * avg_time_per_step

                    print(f"Step {global_step}/{total_steps} "
                        f"Elapsed: {elapsed/60:.2f} min "
                        f"ETA: {eta/60:.2f} min")
            

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




if __name__ == "__main__":
    main(params)
    