from omegaconf import OmegaConf
import torch
import random
from torch.utils import data
from pathlib import Path
import os
import numpy as np
import math
import json
from datasets import register_dataset
from preprocessors import OptimizedSTFTPreprocessor

@register_dataset(name="classification_dataset")
class ClassificationDataset(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
        #THE PLAN
        #also make masked_tf_datased_from_cached
        self.cfg = cfg
        self.task_cfg = task_cfg
        manifest_path = cfg.data
        self.dataroot = Path(manifest_path).parent.absolute()
        self.lengths = []

        with open(manifest_path, 'r') as file:
            self.datajson = json.load(file)
        filtered_datajson = []
        for sample in self.datajson:
            file_name = sample['eeg']
            file_path = os.path.join(self.dataroot, file_name)
            data = np.load(file_path)

            length = data.shape[0]
            self.lengths.append(length)
            if math.isnan(sample['emotion_label']):
                continue
            if length >= 500 and length < 6000:
                filtered_datajson.append(sample)
        print(f'After filtering: {len(filtered_datajson)} / {len(self.datajson)}')
        self.datajson = filtered_datajson

        self.cached_features = None
        print(len(self))

        if preprocessor_cfg.name=="stft":
            # extracter = STFTPreprocessor(preprocessor_cfg)
            self.extracter = OptimizedSTFTPreprocessor(preprocessor_cfg)
        else:
            raise NotImplementedError('NO!')

        self.speaker_mapping = {
            'YMS': 0,
            'ZKH': 1,
            'YLS': 2,
            'YDR': 3,
            'ZPH': 4,
            'ZGW': 5,
            'YRH': 6,
            'YSL': 7,
            'ZAB': 8,
            'ZJS': 9,
            'YDG': 10,
            'ZJN': 11,
            'YSD': 12,
            'YAK': 13,
            'YAG': 14,
            'YRK': 15,
            'YFR': 16,
            'YMD': 17,
            'YTL': 18,
            'ZKB': 19,
            'ZMG': 20,
            'YFS': 21,
            'YIS': 22,
            'ZJM': 23,
            'ZDN': 24,
            'YAC': 25,
            'YHS': 26,
            'ZDM': 27,
            'ZKW': 28,
            'YRP': 29
        }

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["target"].shape[-1]

    def __len__(self):
        return len(self.datajson)

    def __getitem__(self, idx):
        file_name = self.datajson[idx]['eeg']
        file_path = os.path.join(self.dataroot, file_name)
        wav = np.load(file_path)

        output_data = self.extracter(wav.T).T
        output_data = output_data.reshape(output_data.shape[0], -1)

        assert not np.any(np.isnan(output_data)), np.isnan(output_data).sum()

        # label = int(self.datajson[idx]['emotion_label'] + 1)
        # assert label in [0, 1, 2], label

        label = self.speaker_mapping[self.datajson[idx]['subject_name']]
        assert isinstance(label, int), label

        return {"length": output_data.shape[1],
                "spec": output_data,
                "label": int(label)}