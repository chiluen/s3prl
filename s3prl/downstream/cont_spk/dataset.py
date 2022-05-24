import os
import re
import sys
import time
import random
import pickle

import tqdm
import torch
import torchaudio
import numpy as np 
from torch import nn
from pathlib import Path
from sox import Transformer
from torchaudio import load
from librosa.util import find_files
from joblib.parallel import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from torchaudio.sox_effects import apply_effects_file

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

# Voxceleb 2 Speaker verification
class Contrastive_train(Dataset):
    def __init__(self, vad_config, key_list, file_path, meta_data, max_timestep=None, n_jobs=12):
        self.roots = file_path
        self.root_key = key_list
        self.max_timestep = max_timestep
        self.vad_c = vad_config 
        self.dataset = []
        self.all_speakers = []

        for index in range(len(self.root_key)):
            cache_path = Path(os.path.dirname(__file__)) / '.wav_lengths' / f'{self.root_key[index]}_length.pt'
            cache_path.parent.mkdir(exist_ok=True)
            root = Path(self.roots[index])

            if not cache_path.is_file():
                def trimmed_length(path):
                    wav_sample, _ = apply_effects_file(path, EFFECTS)
                    wav_sample = wav_sample.squeeze(0)
                    length = wav_sample.shape[0]
                    return length

                wav_paths = find_files(root)
                wav_lengths = Parallel(n_jobs=n_jobs)(delayed(trimmed_length)(path) for path in tqdm.tqdm(wav_paths, desc="Preprocessing"))
                wav_tags = [Path(path).parts[-3:] for path in wav_paths]
                torch.save([wav_tags, wav_lengths], str(cache_path))
            else:
                wav_tags, wav_lengths = torch.load(str(cache_path))
                wav_paths = [root.joinpath(*tag) for tag in wav_tags]

            speaker_dirs = ([f.stem for f in root.iterdir() if f.is_dir()])
            self.all_speakers.extend(speaker_dirs)
            for path, length in zip(wav_paths, wav_lengths):
                if length > self.vad_c['min_sec']:
                    self.dataset.append(path)

        self.all_speakers.sort()
        self.speaker_num = len(self.all_speakers)

        self.augment = Compose([
                        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                       ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        wav, _ = apply_effects_file(str(path), EFFECTS)
        wav = wav.squeeze(0)
        length = wav.shape[0]
        
        if self.max_timestep != None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start : start + self.max_timestep]
                length = self.max_timestep
        
        #對wav做augmentation
        wav_original = wav.numpy()
        wav_manipulate = self.augment(wav_original, sample_rate=16000)
        
        tags = Path(path).parts[-3:]
        utterance_id = "-".join(tags).replace(".wav", "")
        label = self.all_speakers.index(tags[0])

        #還要回傳一個length
        return wav_original, wav_manipulate, utterance_id, label


    def collate_fn(self, data_sample):
        wavs_original, wavs_manipulate, utterance_ids, labels = zip(*data_sample)
        all_wavs = wavs_original + wavs_manipulate
        utterance_ids = utterance_ids + utterance_ids
        labels = labels + labels
        return all_wavs, utterance_ids, labels