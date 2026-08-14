# Multi-accent data statistics

Every metric cell is formatted as `train/seen test/unseen test`.
Seen/unseen is determined only by whether the test speaker occurs in train.

| Accent ID | Accent | Utterances | Duration (hours) | Total duration (hours) | Speakers |
|---:|---|---:|---:|---:|---:|
| 0 | north | 43467/2000/2000 | 44.307/1.672/1.931 | 47.910 | 126/106/39 |
| 1 | Sichuan | 19066/2377/2377 | 16.356/2.081/1.801 | 20.238 | 49/49/2 |
| 2 | Guangdong | 19029/2371/2371 | 15.467/1.930/1.409 | 18.806 | 51/51/2 |
| 3 | south | 17651/1000/1000 | 16.667/0.919/0.840 | 18.426 | 44/42/7 |
| 4 | Henan | 14236/1769/1769 | 10.269/1.271/1.314 | 12.854 | 20/20/2 |
| 5 | Shanghai | 7603/648/648 | 6.923/0.635/0.541 | 8.098 | 28/28/2 |
| 6 | Wuhan | 4073/461/461 | 4.178/0.483/0.383 | 5.044 | 9/9/1 |
| 7 | Tianjin | 6552/859/859 | 2.792/0.354/0.514 | 3.659 | 6/6/1 |
| 8 | singapore | 3729/1109/1109 | 2.218/0.660/0.522 | 3.400 | 3/3/1 |
| all | **Total** | 135406/12594/12594 | 119.178/10.004/9.254 | 138.435 | 336/314/57 |

## Split policy

- Sichuan, Guangdong, Henan, and Shanghai reserve exactly two speaker-disjoint test speakers.
- Wuhan, Tianjin, and Singapore reserve exactly one speaker-disjoint test speaker.
- For each configured accent, seen-test utterances equal unseen-test utterances; all remaining rows are train.
- Except for north and south, every train speaker occurs at least once in seen test; north and south retain sampled speaker coverage.
- Non-AISHELL accents have no dev rows.
- North keeps exactly 2,000 seen-test plus 2,000 unseen-test rows.
- South keeps exactly 1,000 seen-test plus 1,000 unseen-test rows.
- Unselected AISHELL-3 official-test rows are retained in unused.txt, never moved into train.
- Test unseen speakers are counted relative to train speakers only.
- AISHELL-3 official dev is retained separately in dev.txt and is not included in the table.
