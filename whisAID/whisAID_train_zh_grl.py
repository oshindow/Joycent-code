 
import torch
import torch.nn as nn
# import sys
# sys.path.insert(0, './whisper')
import whisper
 
from pathlib import Path
# from utils import error_stats 
# from data import TextMelSpeakerAccentDataset, TextMelSpeakerAccentBatchCollate
from preprocess_pinyin_accent import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import WhisperTokenizer
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from config import Config
 
 
import random
import numpy as np
import argparse
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

SEED = 2025
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)

class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="large", lang="zh") -> None:
        super().__init__()

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language=lang, task="transcribe")
        initial_prompt = "以下是普通话的句子。"
 
        self.options = whisper.DecodingOptions(language="zh", without_timestamps=True, prompt=initial_prompt)

        # model_path = "exp2/whisper_acc_pretrain/004/checkpoint-epoch=0009.ckpt"
        model_path = "large-v3-turbo"
        self.model = self.load_pretrained_whisper(model_path, n_accents=cfg.n_accents, n_speakers=cfg.n_speakers)

        self.train_path = cfg.train_path
        self.val_path = cfg.val_path
        self.test_path = cfg.test_path
        self.spk_info_path = cfg.spk_info_path
        self.cfg = cfg
        print("train_path:", self.train_path)
        self.train_dataset = WhisperPinyinDataset(self.train_path, self.tokenizer, self.spk_info_path, config=self.cfg, task='train',
                                             pseudo_labels=getattr(self, 'pseudo_labels', None))
        
        # self.train_dataset = TextMelSpeakerAccentDataset(train_filelist_path, cmudict_path, add_blank,
        #                                   n_fft, n_feats, sample_rate, hop_length,
        #                                   win_length, f_min, f_max, zhdict_path, train=True)
        
        # only decoder training
        
        # for p in self.model.encoder.parameters():
        #    p.requires_grad = False
        
        for p in self.model.decoder.parameters():
            p.requires_grad = False  # Freeze all encoder parameters
        
        # del self.model.decoder
         
        # self.loss_fn_accent = nn.CrossEntropyLoss()
        
        # self.train_dataset_length = 155065

        self.automatic_optimization = False
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        
        optimizer, optimizer_spk = self.optimizers()
        optimizer.zero_grad()   
        input_ids = batch["input_ids"]
        # spk 292, acc 12

        ### Accent labels
        accent_labels = torch.tensor(batch["accent_labels"], dtype=torch.long, device=self.device)
        spks = torch.tensor(batch["spk_ids"], dtype=torch.long, device=self.device)
        # print(accent_labels, spks)
        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

            # out = self.model.decode(mel=input_ids, options=self.options)
            # print(out)

        
        # accent 
        feats_acc, logits_acc = self.model.acc_head(audio_features.mean(dim=1))
        # feats_spk, logits_spk = self.model.spk_head(audio_features)
        
        # logits_acc_mean = logits_acc.mean(dim=1)
        # logits_spk_mean = logits_spk.mean(dim=1)
        
        # accent_loss = self.loss_fn_accent(logits_acc_mean, accent_labels)
        # accent_loss = F.cross_entropy(logits_acc, accent_labels)

        # feats_acc_mean = feats_acc.mean(dim=1).unsqueeze(1)

        # dis
        feats_acc_D = feats_acc.clone().detach()
        _, pred_spk_D = self.model.spk_grl(feats_acc_D.squeeze(1))
        loss_spk_grl_D = F.cross_entropy(pred_spk_D.squeeze(1), spks)

        optimizer_spk.zero_grad()
        self.manual_backward(loss_spk_grl_D)
        optimizer_spk.step()
        self.scheduler_spk.step()

        # generator
        _, pred_spk_adv = self.model.spk_grl(feats_acc.squeeze(1))
        loss_spk_grl_adv = F.cross_entropy(pred_spk_adv.squeeze(1), spks)
        accent_loss = F.cross_entropy(logits_acc, accent_labels)
        # loss = accent_loss
        loss = accent_loss + loss_spk_grl_adv * 0.05

             
        self.manual_backward(loss) 
        optimizer.step()
        self.scheduler.step()

        self.log("train/loss_spk_grl_adv", loss_spk_grl_adv, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss_acc", accent_loss, on_step=True, prog_bar=True, logger=True)
        # self.log("train/loss_spk", loss_spk, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss_spk_grl_D", loss_spk_grl_D, on_step=True, prog_bar=True, logger=True)
        
        # self.log()
        # return loss

    def validation_step(self, batch, batch_id):
        
        uids = batch["uids"]
        input_ids = batch["input_ids"]
        
       
         
        accent_labels = torch.tensor(batch["accent_labels"], dtype=torch.long, device=self.device)
        spks = torch.tensor(batch["spk_ids"], dtype=torch.long, device=self.device)
         
        
        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)
            feats_acc, logits_acc = self.model.acc_head(audio_features.mean(dim=1))
            # feats_spk, logits_spk = self.model.spk_head(audio_features)
            # logits_acc_mean = logits_acc.mean(dim=1)
            # logits_spk_mean = logits_spk.mean(dim=1)
        
            accent_id = logits_acc.argmax(dim=-1)
            # speaker_id = logits_spk.argmax(dim=-1)
            # print(feats_acc.shape) # 32, 256
            # _, pred_spk_adv = self.model.spk_grl(feats_acc)
            
            # pred_spk_adv = pred_spk_adv.argmax(dim=-1).squeeze(1)
            # print(pred_spk_adv, spks)
            # correct = (pred_spk_adv == spks).sum().item()
            # acc_spk = correct / spks.size(0)
        
        correct = (accent_id == accent_labels).sum().item()
        acc_acc = correct / accent_labels.size(0)
        
        self.log("val/acc_acc", acc_acc, on_step=True, prog_bar=True, logger=True)
        # self.log("val/acc_spk", acc_spk, on_step=True, prog_bar=True, logger=True)
        return {
            "acc": acc_acc,
        }
    

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        
        spk_params = list(model.spk_grl.parameters())
        spk_param_ids = set(id(p) for p in spk_params)
 
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if id(p) not in spk_param_ids
                    and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if id(p) not in spk_param_ids
                    and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        
        # print(optimizer_grouped_parameters)
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        self.optimizer_spk = torch.optim.Adam(model.spk_grl.parameters(), lr=self.cfg.learning_rate)
        
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        scheduler_spk = get_cosine_schedule_with_warmup(
            self.optimizer_spk, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )

        self.scheduler = scheduler
        self.scheduler_spk = scheduler_spk
        return [self.optimizer, self.optimizer_spk], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""

        if stage == 'fit' or stage is None:
            print('total train dataset length:', len(self.train_dataset.datalist))
            self.t_total = (
                (len(self.train_dataset.datalist) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
            
    def load_pretrained_whisper(self, model_name, n_accents, n_speakers):
        # Load the original Whisper model
        model = whisper.load_model(model_name, n_accents=n_accents, n_speakers=n_speakers)
        # model = whisper.Whisper(whisper_model.dims, n_accents=n_accents, n_speakers=n_speakers)  # Use the same dimensions as the original model

        # # Load only matching weights (ignore missing keys)
        # model.load_state_dict(whisper_model.state_dict(), strict=False)

        print("Loaded Whisper model")

        return model  # Return the modified model
        
    def train_dataloader(self):
        """訓練データローダーを作成する"""
        # self.train_dataset = WhisperPinyinDataset(self.train_path, self.tokenizer, self.spk_info_path, config=self.cfg, task='train',
        #                                      pseudo_labels=getattr(self, 'pseudo_labels', None))
         
    
        return torch.utils.data.DataLoader(self.train_dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        dataset = WhisperPinyinDataset(self.val_path, self.tokenizer, self.spk_info_path, config=self.cfg, task='dev')
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )


    # def on_train_epoch_end(self):
    #     self.model.eval()
    #     print("Decoding training data for pseudo-labeling...")

    #     dataloader = self.train_dataloader()
    #     print("on_train_epoch_end epoch", self.current_epoch)
    #     dataloader.sampler.set_epoch(self.current_epoch - 1)
    #     pseudo_labels = {}

    #     for batch in dataloader:
    #         uids = batch["uids"]
    #         input_ids = batch["input_ids"].to(self.device)

    #         with torch.no_grad():
    #             decoded = self.model.decode(input_ids, self.options)

    #         for uid, out in zip(uids, decoded):
    #             pseudo_labels[uid] = out.text.strip()

    #     # Save to disk or in-memory store (self.pseudo_labels)
    #     self.pseudo_labels = pseudo_labels
    #     with open(f"pseudo_labels_epoch_{self.current_epoch}.json", "w") as f:
    #         json.dump(self.pseudo_labels, f, ensure_ascii=False, indent=2)
    #     print(f"Updated pseudo-labels for {len(pseudo_labels)} training samples.")

if __name__ == '__main__':
    
    print(whisper.available_models())
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="whisAID_zh_grl",
        help="Supervision manifest that contains verbatim transcript",
    )
    parser.add_argument(
        "--train-id",
        type=str,
        default="005", # turbo
        help="Supervision manifest that contains verbatim transcript",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Supervision manifest that contains verbatim transcript",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="resources/whisAID/zh_all/train.csv",
        help="Batch size for training",
    )
    parser.add_argument(
        "--val-path",
        nargs="+",
        default=["resources/whisAID/zh_all/test_unseen.csv"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Root directory prepended to relative wav paths in whisAID csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/whisAID",
        help="Directory for TensorBoard logs and checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="small",
        help="Model name to use for training",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=128,
        help="Number of mel frequency bins",
    )
     
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Precision for training, e.g., '16', '32', 'bf16-mixed', 'fp16', 'bf16'",   
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=1e-6,
        help="Epsilon for the Adam optimizer",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Number of warmup steps for the learning rate scheduler",
    )

    args = parser.parse_args()
    train_name = args.train_name
    train_id = args.train_id
    model_name = args.model_name
    #model_name = "exp/checkpoint_stage1/checkpoint-epoch=0003.ckpt"
    lang = "zh"

    log_output_dir = Path(args.output_dir) / train_name
    check_output_dir = Path(args.output_dir) / train_name / train_id
    print("train-path:", args.train_path)
    print("check_output_dir:", check_output_dir)

    cfg = Config()
    assert "▁" not in cfg.otc_token
    cfg.otc_token = f"▁{cfg.otc_token}"

    cfg.num_train_epochs = args.epoch
    cfg.batch_size = args.batch_size
    cfg.train_path = args.train_path
    cfg.val_path = args.val_path
    cfg.data_root = args.data_root
    cfg.n_mels = args.n_mels
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.adam_epsilon = args.adam_epsilon
    cfg.warmup_steps = args.warmup_steps
    
    Path(log_output_dir).mkdir(parents=True, exist_ok=True)
    Path(check_output_dir).mkdir(parents=True, exist_ok=True)
 
    tflogger = TensorBoardLogger(
        save_dir=str(log_output_dir),
        name="logs",
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(check_output_dir),
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(cfg, model_name, lang)

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    seed_everything(2025, workers=True)
    
    trainer = Trainer(
        # precision="bf16-mixed", # =16,
        precision=args.precision,
        accelerator=DEVICE,
        # devices=2,
        # strategy="ddp",
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        val_check_interval=300, # When using an IterableDataset for `train_dataloader`, `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies checking validation every k training batches.
        # deterministic=True
    )

    trainer.fit(model)
