E1: logs/new_exp_sg_acc_blank
reimplement chinese version of grad-tts
~ 4 Days

E2: logs/new_exp_sg_acc_blank_grl_gst_ddp
scheduled learning rate, e-6 too low, converged slowly
NaN

E3: logs/new_exp_sg_acc_blank_grl_gst_ddp2_lrfix
fixed learning rate 1e-5 hope no NaN
NaN

E4: train_multi_speaker_multi_accent_zh_blank_grl_ddp2.py
complete framework

E5: GST loss and grl



multi-speaker, multi-accent, TTS 
singaporean accent + AISHELL-3 data augmentation

Data Augmentation Test:
E1: Grad-TTS + spk embedding, SG data only, w/o blank
    duration 问题
E2: Grad-TTS + spk/acc embedding, AISHELL-3 + SG data, w/o blank
    timbre 不像，吞字, prosody

# # % Grad-TTS (spk/acc-SG+MC) & logs/new_exp_sg_acc P2
# echo "% Grad-TTS (spk/acc-SG+MC) & logs/new_exp_sg_acc P2"
# python dump_feats_to_GPU4_zh_acc.py -f resources/filelists/synthesis_zh_acc_paper.txt -c logs/new_exp_sg_acc/grad_270.pt -o logs/new_exp_sg_acc/gen_grad_270/raw

# input_path=new_exp_sg_acc/gen_grad_270/raw 
# output_path=gen_grad_270_P2/
# rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/$input_path xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/$output_path -r

E3: Grad-TTS + spk/acc embedding, AISHELL-3 + SG data, w/ blank (resume)
    
E4: Grad-TTS + conformer encoder + spk/acc embedding, AISHELL-3 + SG data, w/ blank  (resume)

E6: Grad-TTS + conformer encoder + spk/Global accent token, AISHELL-3 + SG data, w/ blank (resume)
    train_multi_speaker_multi_accent_zh_blank_con_ddp2_gst.py
    logs/new_exp_sg_acc_blank_conformer_gst_E5/grad_207.pt

E7: Grad-TTS + conformer encoder + spk/Global accent token + loss, AISHELL-3 + SG data, w/ blank (resume)

E8: Grad-TTS + conformer encoder + spk/Global accent token (Layer normalizaiton) + gstloss, AISHELL-3 + SG data, w/ blank (resume)
- accent 最好，timber 差
train_multi_speaker_multi_accent_zh_blank_con_ddp2_gstloss_cln.py

E9: Grad-TTS + conformer encoder + spk/Global accent token (Layer normalizaiton) + gstloss + grlloss, AISHELL-3 + SG data, w/ blank (resume)
- 差

E10: Grad-TTS + conformer encoder + spk/Global accent token (Layer normalizaiton) + gstloss + grlloss (BEFORE PRED_ACC), AISHELL-3 + SG data, w/ blank (resume) bad
- timber 比E8 好一点，accent 不好

E11: Grad-TTS + conformer encoder + larger spk emb/Global accent token (Layer normalizaiton) + gstloss + grlloss + larger classifier (BEFORE PRED_ACC), AISHELL-3 + SG data, w/ blank 
- gst loss很差

E12: Grad-TTS + conformer encoder + larger spk emb/Global accent token (Layer normalizaiton) + gstloss + grlloss + normal classifier focal loss (BEFORE PRED_ACC), AISHELL-3 + SG data, w/ blank 
- accent 没有E8好，语速偏快，timber 没有E10好


E13: Grad-TTS + conformer encoder + larger spk emb/Global accent token (Layer normalizaiton) + gstloss + grlloss + normal classifier focal loss (BEFORE PRED_ACC), AISHELL-3 + SG data, w/ blank (GPU0,1)
adjust the bucket size

E14: aishell train base version 218 speakers, 3 accents bad
E15: sg finetuning 4 speakers, 1 accent bad

E16: E8 REMOVE speaker embedding

------------------------
E12: Grad-TTS larger channel + conformer encoder + spk/Global accent token + loss + grl, AISHELL-3 + SG data, w/ blank, data balance, rm silence token


Blank Symbol Test: use blank better, conformer 训练慢
    E1: AISHELL-3 Grad-tts Conformer with blank (GPU0)
        logs/new_exp_sg_acc_blank_conformer
    E2: AISHELL-3 Grad-tts Conformer without blank (GPU1)
        logs/new_exp_sg_acc_conformer   

Timbre:
    E0: AISHELL-3 Grad-tts with blank
        logs/new_exp_sg_acc_blank/grad_679.pt 
        ~ 5 Days
        speaker 还原度不够, 
        Datasets summary:
            aishell3 | around 500 audio clips per speaker, 218 speakers
            ljspeech | 13100 audio clips per speaker
            sg | 3000 audio clips per speaker
            LibriTTS | 200 audio clips per speaker, 247 speakers
    
    E3: Libritts Grad-tts (GPU2)
        数据量够，phoneme？silence？loss 差不多

E1: AISHELL-3 Grad-tts Conformer with blank try larger channel (GPU2)
    

retrain

 1943  nohup python3 train_multi_speaker_multi_accent_zh.py > new_exp_sg_acc_1.log &
 1947  nohup python3 train_multi_speaker_multi_accent_zh.py > new_exp_sg_acc_1.log &
 1950  nohup python3 train_multi_speaker_multi_accent_zh_blank.py > new_exp_sg_acc_blank_E3.log &
 1955  nohup python3 train_multi_speaker_multi_accent_zh_blank_conformer.py > new_exp_sg_acc_blank_conformer_E4.log &