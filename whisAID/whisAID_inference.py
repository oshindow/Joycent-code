import torch
from preprocess_pinyin_accent import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
from transformers import WhisperTokenizer
import whisper
import sys
# sys.path.insert(0, '/home/xintong/whisper')
# from utils import error_stats, normlizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
# from config_en import Config
from config import Config
from sklearn.metrics import silhouette_score


# from config import Config
# en
# (002 medium) 
# model_path = "/data1/xintong/whisperAID/exp/whisAID_en/002/checkpoint-epoch=0003.ckpt"
# en grl
# (002 medium, 003 0.1adv unfrozen, 004 0.05adv unfrozen, 005 0.01adv unfrozen) 
# model_path = "/data1/xintong/whisperAID/exp/whisAID_zh_grl/065/checkpoint-epoch=0006.ckpt"
# zh
# (003 medium) 
# model_path = "/data1/xintong/whisperAID/exp/whisAID_zh/003/checkpoint-epoch=0005.ckpt"
# (002 large_v3_turbo) model_path = 
# (001 small) model_path = 
# zh grl
# (004 medium) 
model_path = "/data1/xintong/whisperAID/exp/whisAID_zh_grl/004/checkpoint-epoch=0006.ckpt"
# (005 large_v3_turbo) model_path = "/data1/xintong/whisperAID/exp/whisAID_zh_grl/005/checkpoint-epoch=0008.ckpt" 
# (006 small) model_path = "/data1/xintong/whisperAID/exp/whisAID_zh_grl/065/checkpoint-epoch=0006.ckpt"
# model_path = "/data1/xintong/whisperAID/exp/whisAID_en/001/checkpoint-epoch=0005.ckpt" # w/o grl
# print(Config['n_mels'])
config = Config()
config.n_mels = 80
# config.test_path = ["resources/whisAID/CommonAccent/test_seen.csv"]
# config.test_path = ['resources/whisAID/zh_all/test_seen.csv']
config.test_path = ['resources/whisAID/magichub_sg/test_seen.csv']
# config.test_path = ['joycent_e1_e346.sg']
# config.test_path = ['joycent_e2_e31.sg']
# config.test_path = ['joycent_e3_e279.sg']

model = whisper.load_model(model_path, n_accents=config.n_accents, n_speakers=config.n_speakers)
print(model_path)

spk_info_path = 'dump/aishell3/spk_info_only.txt'
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
test_dataset = WhisperPinyinDataset(config.test_path, tokenizer, spk_info_path, config, task='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=WhisperDataCollatorWhithPadding())

cer_results = []
rtfs = []
cnt = 0

print(len(test_loader))

all_correct = 0
all_samples = 0
all_preds = []
all_labels = []
all_feats = []
spk_label = []
device = 'cuda'
per_accent = {}
# accent_embs = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
for batch in test_loader:
        uids = batch["uids"]
        input_ids = batch["input_ids"].to(device)
        
        accent_labels = torch.tensor(batch["accent_labels"], dtype=torch.long, device=device)
        spks = torch.tensor(batch["spk_ids"], dtype=torch.long, device=device)
         
        
        with torch.no_grad():
            audio_features = model.encoder(input_ids)
            feats_acc, logits_acc = model.acc_head(audio_features.mean(dim=1))
            
            accent_id = logits_acc.argmax(dim=-1)
        
        correct = (accent_id == accent_labels).sum().item()
        acc_acc = correct / accent_labels.size(0)
        # print(spks)
        # print(batch["accent_labels"])
        for idx in range(len(uids)):
            # all_feats.append(feats_acc[idx].cpu().numpy())
            accent_label = batch["accent_labels"][idx]
            spk_label = batch["spk_ids"][idx]
            if accent_label not in per_accent:
                per_accent[accent_label] = {'feats': [feats_acc[idx].cpu().numpy()], 'spk_labels': [spk_label]}
            else:
                per_accent[accent_label]['feats'].append(feats_acc[idx].cpu().numpy())
                per_accent[accent_label]['spk_labels'].append(spk_label)
            
            # print(spk_label)         
            all_labels.append(accent_label)
            all_preds.append(accent_id[idx].detach().cpu().numpy())
        print(acc_acc)
            
# np.con
# feats_np = np.vstack(all_feats)
# labels = np.array(all_labels)
# # np.savez('tsne_accent_embeddings.npz', feats=feats_np, labels=labels)
# # labels = logits_acc.argmax(dim=-1).cpu().numpy()  # 假设要看 accent label 分布

# # 用 t-SNE 降维到 2D
# tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
# feats_2d = tsne.fit_transform(feats_np)

acc2id = {1: 'Changsha', 2: 'Guangdong', 
          3: 'Nanchang', 4: 'Shanghai', 
          5: 'Sichuan', 6: 'Tianjin', 7: 'Henan', 
          8: 'Wuhan', 9: 'Shanxi', 10: 'north', 
          11: 'south', 12: 'singapore'}

# acc2id = {'Changsha':1, 'Guangdong':2, 'Nanchang':3, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'Shanxi': 9, 'north': 10, 'south': 11, 'singapore': 12}
if 'en' in model_path:
    acc2id = {'us': 1, 'canadian': 2, 'australian': 3, 'southasian': 4, 'english': 5, 'southernafrican': 6, 'irish': 7, 'scottish': 8, 'filipino': 9, 'singaporean': 10, 'hongkong': 11, 'malaysian': 12, 'newzealand': 13}
else:
    acc2id = {'Changsha':1, 'Guangdong':2, 'Nanchang':3, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'Shanxi': 9, 'north': 10, 'south': 11, 'singapore': 12}
    # acc2id = {'Guangdong':2, 'Nanchang':3, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'Shanxi': 9, 'north': 10, 'south': 11, 'singapore': 12}
    
    # acc2id = {'Guangdong':2, 'Nanchang':3, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'north': 10, 'south': 11, 'singapore': 12}
    # acc2id = {'Guangdong':2, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Henan':7, 'Wuhan':8, 'north': 10, 'south': 11, 'singapore': 12}
    # acc2id = {'Guangdong':2, 'Shanghai':4, 'Sichuan':5, 'Wuhan':8, 'south': 11, 'singapore': 12}
    # acc2id = {'Guangdong':2, 'Shanghai':4, 'Sichuan':5, 'Tianjin':6, 'Wuhan':8, 'north': 10, 'south': 11, 'singapore': 12}
    acc2id = {'Guangdong':2, 'Henan':7, 'singapore': 12}
    
print(np.unique(all_preds))
# 可视化
# plt.figure(figsize=(8, 6))
# unique_labels = np.unique(labels)

# for lab in unique_labels:
#     print(lab, acc2id[lab])
#     idx = labels == lab
#     plt.scatter(feats_2d[idx, 0], feats_2d[idx, 1], 
#                 label=f"{acc2id[lab]}", alpha=0.7)
# scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
# plt.colorbar(scatter, label="Accent class")
# plt.title("t-SNE Visualization of Accent Embeddings")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.legend(
#     # title="Accent Class",
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.1),  # 图下方
#     ncol=5,                      # 一行显示 5 个
#     frameon=False                # 去掉边框，更简洁
# )
# plt.tight_layout()
# plt.show()
# plt.savefig('tsne_accent_embeddings.png', dpi=300, bbox_inches='tight')
# 生成分类报告
print("\nClassification Report:")
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=list(acc2id.keys()),
    digits=4
))

scores = []
for accent, values in per_accent.items():
    n_speakers = len(np.unique(values["spk_labels"]))
    if n_speakers < 2:
        continue
    score = silhouette_score(values["feats"], values["spk_labels"])
    scores.append(score)
    
    # if 
    print("Accent:", accent, "Silhouette:", score)

print("average:", sum(scores) / len(scores))
# # 绘制混淆矩阵
# def plot_confusion_matrix(labels, preds, class_names):
#     cm = confusion_matrix(labels, preds)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', 
#                 xticklabels=class_names, 
#                 yticklabels=class_names,
#                 cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     # plt.title('Confusion Matrix')
#     # plt.show()
#     plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

# plot_confusion_matrix(
#     all_labels,
#     all_preds,
#     ['Northern', 'Southern', 'Sichuan', 'Singaporean', 'Non-native', 'Others']
# )