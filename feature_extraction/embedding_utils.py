import os


def iter_wav_paths(filelist_path):
    with open(filelist_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield line.split("|")[0]


def shard_items(items, num_shards=1, shard_id=0):
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    return [item for index, item in enumerate(items) if index % num_shards == shard_id]


def resolve_data_path(path, data_root=""):
    path = os.path.expanduser(path)
    if os.path.isabs(path) or not data_root:
        return path
    return os.path.join(os.path.expanduser(data_root), path)


def make_embedding_path(wav_path, output_dir_name, keep_wav_suffix=False):
    if "wav_16k" in wav_path:
        path = wav_path.replace("wav_16k", output_dir_name)
    elif "WAV" in wav_path:
        path = wav_path.replace("WAV", output_dir_name)
    elif "prompt_acc" in wav_path:
        path = wav_path.replace("prompt_acc", output_dir_name)
    else:
        raise ValueError(f"Cannot infer embedding path from wav path: {wav_path}")

    if keep_wav_suffix:
        return path + ".npy"
    return os.path.splitext(path)[0] + ".npy"


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
