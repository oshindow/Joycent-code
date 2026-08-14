# KeSpeech province-accent statistics

Each metric cell is `train/test seen/test unseen`. Duration is calculated from `lengths.json` as `mel_frames * 256 / 16000`.

| Accent ID | Province/municipality | Chinese | Utterances | Duration (hours) | Used total duration (hours) | Speakers |
|---:|---|---|---:|---:|---:|---:|
| 6 | Hebei | 河北省 | 186360/2000/2000 | 291.993/3.167/3.122 | 298.281 | 5081/2000/435 |
| 12 | Liaoning | 辽宁省 | 106237/2000/2000 | 165.164/3.100/3.105 | 171.370 | 3004/2000/255 |
| 15 | Shandong | 山东省 | 79141/2000/2000 | 123.517/3.147/3.072 | 129.736 | 2255/2000/188 |
| 7 | Heilongjiang | 黑龙江省 | 66654/2000/2000 | 103.370/3.120/3.038 | 109.529 | 2050/2000/160 |
| 0 | Anhui | 安徽省 | 63941/2000/2000 | 100.054/3.155/3.046 | 106.255 | 1786/1746/160 |
| 14 | Shaanxi | 陕西省 | 55675/2000/2000 | 89.256/3.265/3.260 | 95.781 | 1886/1846/112 |
| 10 | Jiangsu | 江苏省 | 55065/2000/2000 | 85.974/3.175/3.240 | 92.390 | 1684/1629/116 |
| 16 | Sichuan | 四川省 | 52581/2000/2000 | 81.524/3.126/3.101 | 87.751 | 1650/1614/94 |
| 8 | Henan | 河南省 | 47914/2000/2000 | 74.621/3.169/3.046 | 80.836 | 1376/1350/119 |
| 1 | Beijing | 北京市 | 33273/2000/2000 | 51.380/3.082/3.145 | 57.607 | 1010/1000/99 |
| 13 | Ningxia | 宁夏回族自治区 | 29496/1604/1604 | 46.275/2.511/2.479 | 51.265 | 878/865/65 |
| 3 | Gansu | 甘肃省 | 21683/1013/1013 | 34.370/1.642/1.726 | 37.738 | 636/625/47 |
| 11 | Jilin | 吉林省 | 12958/900/900 | 20.943/1.448/1.418 | 23.809 | 362/356/38 |
| 2 | Chongqing | 重庆市 | 13236/906/906 | 19.691/1.333/1.544 | 22.568 | 394/387/38 |
| 17 | Yunnan | 云南省 | 10759/669/669 | 17.487/1.077/1.087 | 19.652 | 365/357/29 |
| 5 | Guizhou | 贵州省 | 8877/319/319 | 14.240/0.519/0.441 | 15.200 | 283/279/14 |
| 9 | InnerMongolia | 内蒙古自治区 | 8473/418/418 | 13.018/0.629/0.610 | 14.257 | 264/258/19 |
| 4 | Guangxi | 广西壮族自治区 | 5611/258/258 | 9.039/0.420/0.423 | 9.882 | 173/168/12 |
| all | **Total** | — | 857934/26087/26087 | 1341.916/41.086/40.904 | 1423.906 | 25137/20480/2000 |

## Split policy

- `phase1` and `phase2` are collection phases, not train/dev/test splits.
- Base train/dev membership comes from the official ASR task.
- Seen-test utterances are sampled from official ASR train; every selected speaker retains at least one other utterance in final train.
- Unseen-test utterances come from the deduplicated union of official ASR test and SubdialectID test.
- Per accent, `test_seen` equals `test_unseen`; unseen is capped at 2,000 utterances.
- Unselected and non-task utterances remain in `unused.txt`; `all.txt` covers the complete metadata.
- The phone field is the placeholder `sil`.
- Split seed: `1234`.
