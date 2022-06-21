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


EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

# Voxceleb 2 Speaker verification
class SpeakerVerifi_train(Dataset):
    def __init__(self, vad_config, key_list, file_path, meta_data, max_timestep=None, n_jobs=12, **kwargs):
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

        self.add_silence = kwargs['add_silence']
        self.silence_length = kwargs['silence_length']

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        wav, _ = apply_effects_file(str(path), EFFECTS)
        wav = wav.squeeze(0)
        length = wav.shape[0]

        wav = self.add_silence_func(wav, self.add_silence, self.silence_length) # add by chiluen
        
        if self.max_timestep != None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start : start + self.max_timestep]

        tags = Path(path).parts[-3:]
        utterance_id = "-".join(tags).replace(".wav", "")
        label = self.all_speakers.index(tags[0])
        return wav.numpy(), utterance_id, label
        
    def collate_fn(self, samples):
        #長度為10的tuple, 裝的東西是上面的東西
        return zip(*samples)

    def add_silence_func(self, wav, add_silence_place, silence_length):
        """
        都會傳進去
        """
        if add_silence_place == 'No':
            return wav
        temp_wav = torch.chunk(wav, 10)
        wav_silence = torch.zeros(len(wav) // silence_length)
        
        if add_silence_place == 'front':
            temp_wav = list(temp_wav)
            temp_wav.insert(0, wav_silence)
            return torch.cat(temp_wav)
        elif add_silence_place == 'middle':
            temp_wav = list(temp_wav)
            temp_wav.insert(5, wav_silence)
            return torch.cat(temp_wav)
        elif add_silence_place == 'end':
            temp_wav = list(temp_wav)
            temp_wav.insert(10, wav_silence)
            return torch.cat(temp_wav)
        else:
            return wav


class SpeakerVerifi_test(Dataset):
    def __init__(self, vad_config, file_path, meta_data, **kwargs):
        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.vad_c = vad_config 
        self.dataset = self.necessary_dict['pair_table'] 

        self.add_silence = kwargs['add_silence']
        self.silence_length = kwargs['silence_length']
        
    def processing(self):
        pair_table = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1])
            pair_2= os.path.join(self.root, list_pair[2])
            one_pair = [list_pair[0],pair_1,pair_2 ]
            pair_table.append(one_pair)
        return {
            "spk_paths": None,
            "total_spk_num": None,
            "pair_table": pair_table
        }

    def __len__(self):
        return len(self.necessary_dict['pair_table'])

    def __getitem__(self, idx):
        y_label, x1_path, x2_path = self.dataset[idx]

        def path2name(path):
            return Path("-".join((Path(path).parts)[-3:])).stem

        x1_name = path2name(x1_path)
        x2_name = path2name(x2_path)

        wav1, _ = apply_effects_file(x1_path, EFFECTS)
        wav2, _ = apply_effects_file(x2_path, EFFECTS)

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        wav1 = self.add_silence_func(wav1, self.add_silence, self.silence_length) # add by chiluen
        wav2 = self.add_silence_func(wav2, self.add_silence, self.silence_length) # add by chiluen

        return wav1.numpy(), wav2.numpy(), x1_name, x2_name, int(y_label[0])

    def collate_fn(self, data_sample):
        wavs1, wavs2, x1_names, x2_names, ylabels = zip(*data_sample)
        all_wavs = wavs1 + wavs2
        all_names = x1_names + x2_names
        return all_wavs, all_names, ylabels



    def add_silence_func(self, wav, add_silence_place, silence_length):
        """
        都會傳進去
        """
        if add_silence_place == 'No':
            return wav
        temp_wav = torch.chunk(wav, 10)
        wav_silence = torch.zeros(len(wav) // silence_length)
        
        if add_silence_place == 'front':
            temp_wav = list(temp_wav)
            temp_wav.insert(0, wav_silence)
            return torch.cat(temp_wav)
        elif add_silence_place == 'middle':
            temp_wav = list(temp_wav)
            temp_wav.insert(5, wav_silence)
            return torch.cat(temp_wav)
        elif add_silence_place == 'end':
            temp_wav = list(temp_wav)
            temp_wav.insert(10, wav_silence)
            return torch.cat(temp_wav)
        else:
            return wav

