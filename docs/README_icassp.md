1. python3.8+

conda install python=3.9
tiktoken
numpy==1.26.4
tqdm
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
tensorboard
six
Cython
cd model/monotonic_align; mkdir -p model/monotonic_align; python setup.py build_ext --inplace; cd ../..
einops
numba
librosa
unidecode
matplotlib
h5py
flask
inflect
jieba
pyymal

inference_text2audio_whisper.py add whisper
inference_text2audio_whisper2.py add spk classifier
inference_text2audio_whisper3.py add grl
inference_text2audio_whisper3_qwen.py add qwen llm emb
inference_text2audio_whisper3_qwen2.py acc_cln_layer = 3, spk_cln_layer = 5
inference_text2audio_whisper3_qwen2_facodec.py acc_cln_layer = 0, spk_cln_layer = 5, facodec spk emb
inference_text2audio_whisper3_qwen2_facodec2.py acc_cln_layer = 0, spk_cln_layer = 5, facodec spk emb, llm emb
inference_text2audio_whisper3_qwen2_facodec3.py acc_cln_layer = 0, spk_cln_layer = 5, facodec spk emb, llm emb, decoder acc emb (loss 最好, 效果最好)
inference_text2audio_whisper3_qwen2_facodec4.py acc_cln_layer = 0, spk_cln_layer = 5, facodec spk emb, decoder acc emb (llm 不能去)
inference_text2audio_whisper3_qwen2_facodec5.py random acc_cln_layer, spk_cln_layer = 5, facodec spk emb, decoder acc emb, 在这个模型上inference 的时候选择？没啥区别，控制更不明显
inference_text2audio_whisper3_qwen2_facodec6.py all acc_cln_layer, spk_cln_layer = 5, facodec spk emb, decoder acc emb, (all cln layer accent 最弱, 0 layer 最强, 怎么控制强度?)，在这个模型上inference 的时候选择？ inference time 不行，控制的不明显
inference_text2audio_whisper3_qwen2_facodec7.py dropout acc_cln_layer, spk_cln_layer = 5, facodec spk emb, decoder acc emb 一般

+llm emb
