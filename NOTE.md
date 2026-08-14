############### zh medium grl gt
Classification Report:
              precision    recall  f1-score   support
 
       north     0.9578    0.9540    0.9559       500
       south     0.9238    0.9300    0.9269       300
   singapore     1.0000    0.9700    0.9848       200

    accuracy                         0.9500      1000
   macro avg     0.4117    0.4077    0.4097      1000
weighted avg     0.9561    0.9500    0.9530      1000

macro_f1: 0.4097
weighted_f1: 0.9530

Target reference speech predictions:
/home/xintong/Joycent-code/A0001_S001_0_G0002_segment_0134.wav -> 12 (singapore), confidence=0.9980

Similarity to target: reference speech
12 singapore        n=200   sim=0.8471
11 south            n=300   sim=-0.1488
10 north            n=500   sim=-0.2866

############### zh medium gt

Classification Report:
              precision    recall  f1-score   support
 
       north     0.9592    0.9400    0.9495       500
       south     0.9061    0.9333    0.9195       300
   singapore     1.0000    0.9450    0.9717       200

    accuracy                         0.9390      1000
   macro avg     0.4093    0.4026    0.4058      1000
weighted avg     0.9514    0.9390    0.9450      1000

macro_f1: 0.4058
weighted_f1: 0.9450

Target reference speech predictions:
/home/xintong/Joycent-code/A0001_S001_0_G0002_segment_0134.wav -> 12 (singapore), confidence=0.9945

Similarity to target: reference speech
12 singapore        n=200   sim=0.8313
11 south            n=300   sim=-0.1827
10 north            n=500   sim=-0.3994
Wrote per-sample similarity CSV: whisaid_singapore_similarity.csv


############### zh medium seen joycent

Classification Report:
              precision    recall  f1-score   support
 
   singapore     1.0000    0.2650    0.4190      1000

    accuracy                         0.2650      1000
   macro avg     0.0909    0.0241    0.0381      1000
weighted avg     1.0000    0.2650    0.4190      1000

macro_f1: 0.0381
weighted_f1: 0.4190

 
Similarity to target: reference speech
12 singapore        n=1000  sim=0.5821
 


############### zh medium grl seen joycent


Classification Report:
              precision    recall  f1-score   support
 
   singapore     1.0000    0.3850    0.5560      1000

    accuracy                         0.3850      1000
   macro avg     0.0833    0.0321    0.0463      1000
weighted avg     1.0000    0.3850    0.5560      1000

macro_f1: 0.0463
weighted_f1: 0.5560

  
Similarity to target: reference speech
12 singapore        n=1000  sim=0.6126
 

############### zh medium unseen joycent

Classification Report:
              precision    recall  f1-score   support
 
   singapore     1.0000    0.2880    0.4472      1000

    accuracy                         0.2880      1000
   macro avg     0.1000    0.0288    0.0447      1000
weighted avg     1.0000    0.2880    0.4472      1000

macro_f1: 0.0447
weighted_f1: 0.4472
 
Similarity to target: reference speech
12 singapore        n=1000  sim=0.6147
 
############### zh medium grl unseen joycent

Classification Report:
              precision    recall  f1-score   support

   singapore     1.0000    0.4450    0.6159      1000

    accuracy                         0.4450      1000
   macro avg     0.1000    0.0445    0.0616      1000
weighted avg     1.0000    0.4450    0.6159      1000

macro_f1: 0.0616
weighted_f1: 0.6159
 
Similarity to target: reference speech
12 singapore        n=1000  sim=0.6466

############### zh medium grl cosyvoice

Classification Report:
              precision    recall  f1-score   support
 
   singapore     1.0000    0.1136    0.2041       132

    accuracy                         0.1136       132
   macro avg     0.2500    0.0284    0.0510       132
weighted avg     1.0000    0.1136    0.2041       132

macro_f1: 0.0510
weighted_f1: 0.2041
 

Similarity to target: reference speech
12 singapore        n=132   sim=0.5209

############### zh medium cosyvoice

Classification Report:
              precision    recall  f1-score   support
 
   singapore     1.0000    0.0530    0.1007       132

    accuracy                         0.0530       132
   macro avg     0.3333    0.0177    0.0336       132
weighted avg     1.0000    0.0530    0.1007       132

macro_f1: 0.0336
weighted_f1: 0.1007

 

Similarity to target: reference speech
12 singapore        n=132   sim=0.5201

############### zh medium macst

    accuracy                         0.0000      13.0
   macro avg     0.0000    0.0000    0.0000      13.0
weighted avg     0.0000    0.0000    0.0000      13.0

macro_f1: 0.0000
weighted_f1: 0.0000

Target reference speech predictions:
/home/xintong/Joycent-code/A0001_S001_0_G0002_segment_0134.wav -> 12 (singapore), confidence=0.9945

Similarity to target: reference speech
12 singapore        n=13    sim=0.0459
Wrote per-sample similarity CSV: whisaid_singapore_similarity.csv

############### zh medium grl macst


    accuracy                         0.1538        13
   macro avg     0.2000    0.0308    0.0533        13
weighted avg     1.0000    0.1538    0.2667        13

macro_f1: 0.0533
weighted_f1: 0.2667

Target reference speech predictions:
/home/xintong/Joycent-code/A0001_S001_0_G0002_segment_0134.wav -> 12 (singapore), confidence=0.9980

Similarity to target: reference speech
12 singapore        n=13    sim=0.0940

####### speaker sim
seen joycent
overall: n=1000 speaker_sim=0.5750
unseen joycent
0.5883


0.5871
