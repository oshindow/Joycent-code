# Combined Joycent + KeSpeech multi-accent statistics

Each metric cell is `train/test seen/test unseen`. Duration is calculated from `lengths.json` as `mel_frames * 256 / 16000`.

| Accent ID | Accent | Sources | Utterances | Duration (hours) | Total duration (hours) | Speakers |
|---:|---|---|---:|---:|---:|---:|
| 15 | Hebei | KeSpeech | 186360/2000/2000 | 291.993/3.167/3.122 | 298.281 | 5081/2000/435 |
| 20 | Liaoning | KeSpeech | 106237/2000/2000 | 165.164/3.100/3.105 | 171.370 | 3004/2000/255 |
| 23 | Shandong | KeSpeech | 79141/2000/2000 | 123.517/3.147/3.072 | 129.736 | 2255/2000/188 |
| 16 | Heilongjiang | KeSpeech | 66654/2000/2000 | 103.370/3.120/3.038 | 109.529 | 2050/2000/160 |
| 1 | Sichuan | Joycent+KeSpeech | 71647/4377/4377 | 97.588/5.171/4.865 | 107.625 | 1699/1663/96 |
| 9 | Anhui | KeSpeech | 63941/2000/2000 | 100.054/3.155/3.046 | 106.255 | 1786/1746/160 |
| 22 | Shaanxi | KeSpeech | 55675/2000/2000 | 89.256/3.265/3.260 | 95.781 | 1886/1846/112 |
| 4 | Henan | Joycent+KeSpeech | 62150/3769/3769 | 84.673/4.413/4.333 | 93.419 | 1396/1370/121 |
| 18 | Jiangsu | KeSpeech | 55065/2000/2000 | 85.974/3.175/3.240 | 92.390 | 1684/1629/116 |
| 10 | Beijing | KeSpeech | 33273/2000/2000 | 51.380/3.082/3.145 | 57.607 | 1010/1000/99 |
| 21 | Ningxia | KeSpeech | 29496/1604/1604 | 46.275/2.511/2.479 | 51.265 | 878/865/65 |
| 0 | north | Joycent | 43467/2000/2000 | 43.635/1.641/1.900 | 47.176 | 126/106/39 |
| 12 | Gansu | KeSpeech | 21683/1013/1013 | 34.370/1.642/1.726 | 37.738 | 636/625/47 |
| 19 | Jilin | KeSpeech | 12958/900/900 | 20.943/1.448/1.418 | 23.809 | 362/356/38 |
| 11 | Chongqing | KeSpeech | 13236/906/906 | 19.691/1.333/1.544 | 22.568 | 394/387/38 |
| 24 | Yunnan | KeSpeech | 10759/669/669 | 17.487/1.077/1.087 | 19.652 | 365/357/29 |
| 2 | Guangdong | Joycent | 19029/2371/2371 | 15.176/1.893/1.372 | 18.442 | 51/51/2 |
| 3 | south | Joycent | 17651/1000/1000 | 16.394/0.903/0.825 | 18.122 | 44/42/7 |
| 14 | Guizhou | KeSpeech | 8877/319/319 | 14.240/0.519/0.441 | 15.200 | 283/279/14 |
| 17 | InnerMongolia | KeSpeech | 8473/418/418 | 13.018/0.629/0.610 | 14.257 | 264/258/19 |
| 13 | Guangxi | KeSpeech | 5611/258/258 | 9.039/0.420/0.423 | 9.882 | 173/168/12 |
| 5 | Shanghai | Joycent | 7603/648/648 | 6.806/0.625/0.531 | 7.961 | 28/28/2 |
| 6 | Wuhan | Joycent | 4073/461/461 | 4.116/0.475/0.376 | 4.967 | 9/9/1 |
| 7 | Tianjin | Joycent | 6552/859/859 | 2.691/0.340/0.501 | 3.532 | 6/6/1 |
| 8 | singapore | Joycent | 3729/1109/1109 | 2.161/0.643/0.505 | 3.309 | 3/3/1 |
| all | **Total** | Joycent+KeSpeech | 993340/38681/38681 | 1459.011/50.896/49.964 | 1559.871 | 25473/20794/2057 |

## Merge policy

- Sichuan and Henan share accent IDs across Joycent and KeSpeech.
- All other KeSpeech province accents are appended after the original Joycent accent IDs.
- Joycent keeps speaker IDs 0–392; KeSpeech is offset by 393 and occupies 393–27629.
- `dev` and `unused` are not included in combined training/evaluation lists or statistics.
- Paths are absolute because the two source datasets have different roots.
