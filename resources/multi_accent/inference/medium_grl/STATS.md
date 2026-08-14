# Multi-accent inference-set statistics

Random seed: `37`. All speaker, accent, and text references are sampled from the test sets. Accent-reference audio is at least `2.0` seconds long; references may be reused.

## Input definitions

| Category | Speaker embedding input | Accent embedding input | Text input |
|---|---|---|---|
| seen in-accent | FACodec embedding extracted from one random `test_seen` reference wav of a speaker occurring in train; source and target accents are the same | Independently selected random test reference from the target accent for every utterance, using its corresponding GRL embedding | 10 globally unique phone sequences sampled without replacement from target-accent test transcripts containing at least 5 Chinese characters |
| unseen in-accent | FACodec embedding extracted from one random `test_unseen` reference wav of a speaker absent from train; source and target accents are the same | Independently selected random test reference from the target accent for every utterance, using its corresponding GRL embedding | 10 globally unique phone sequences sampled without replacement from target-accent test transcripts containing at least 5 Chinese characters |
| unseen cross-accent | FACodec embedding extracted from a random `test_unseen` reference wav of the source-accent speaker | Independently selected random test reference from the target accent for every utterance, using its corresponding GRL embedding | 10 globally unique phone sequences sampled without replacement from each target accent's test transcripts containing at least 5 Chinese characters per speaker-target pair |

## In-accent composition

| Accent ID | Accent | Available seen speakers | Selected seen speakers/utts | Available unseen speakers | Selected unseen speakers/utts |
|---:|---|---:|---:|---:|---:|
| 1 | Sichuan | 49 | 10/100 | 2 | 2/20 |
| 2 | Guangdong | 51 | 10/100 | 2 | 2/20 |
| 4 | Henan | 20 | 10/100 | 2 | 2/20 |
| 5 | Shanghai | 28 | 10/100 | 2 | 2/20 |
| 6 | Wuhan | 9 | 9/90 | 1 | 1/10 |
| 7 | Tianjin | 6 | 6/60 | 1 | 1/10 |
| 8 | singapore | 3 | 3/30 | 1 | 1/10 |

## Synthesis utterances per target accent

Cross-accent utterances are assigned to the accent being synthesized (the target accent), not the source speaker's accent.

| Target accent | Seen in-accent | Unseen in-accent | Unseen cross-accent | Total |
|---|---:|---:|---:|---:|
| Sichuan | 100 | 20 | 90 | 210 |
| Guangdong | 100 | 20 | 90 | 210 |
| Henan | 100 | 20 | 90 | 210 |
| Shanghai | 100 | 20 | 90 | 210 |
| Wuhan | 90 | 10 | 100 | 200 |
| Tianjin | 60 | 10 | 100 | 170 |
| singapore | 30 | 10 | 100 | 140 |

## Cross-accent composition

Each target accent is synthesized using unseen speakers from all six other active source accents, with 10 utterances per source-speaker/target pair.

| Target accent | Source accent composition (speakers) | Total source speakers | Utterances |
|---|---|---:|---:|
| Sichuan | Guangdong: 2, Henan: 2, Shanghai: 2, Wuhan: 1, Tianjin: 1, singapore: 1 | 9 | 90 |
| Guangdong | Sichuan: 2, Henan: 2, Shanghai: 2, Wuhan: 1, Tianjin: 1, singapore: 1 | 9 | 90 |
| Henan | Sichuan: 2, Guangdong: 2, Shanghai: 2, Wuhan: 1, Tianjin: 1, singapore: 1 | 9 | 90 |
| Shanghai | Sichuan: 2, Guangdong: 2, Henan: 2, Wuhan: 1, Tianjin: 1, singapore: 1 | 9 | 90 |
| Wuhan | Sichuan: 2, Guangdong: 2, Henan: 2, Shanghai: 2, Tianjin: 1, singapore: 1 | 10 | 100 |
| Tianjin | Sichuan: 2, Guangdong: 2, Henan: 2, Shanghai: 2, Wuhan: 1, singapore: 1 | 10 | 100 |
| singapore | Sichuan: 2, Guangdong: 2, Henan: 2, Shanghai: 2, Wuhan: 1, Tianjin: 1 | 10 | 100 |

## Totals

| Category | Speaker-target pairs | Utterances |
|---|---:|---:|
| seen in-accent | 58 | 580 |
| unseen in-accent | 11 | 110 |
| unseen cross-accent | 66 | 660 |
| **Total** | **135** | **1350** |

Unique sampled text utterances: **1350** (globally sampled without replacement; every transcript contains at least 5 Chinese characters).

The seen-speaker target of 10 cannot be met for Wuhan (9 available), Tianjin (6), or Singapore (3); all available seen speakers are retained for those accents.
