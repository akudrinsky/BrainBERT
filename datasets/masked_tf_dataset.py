from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import random
from torch.utils import data
from pathlib import Path
import os
import numpy as np
from scipy.io import wavfile
import json
from datasets import register_dataset
from preprocessors import OptimizedSTFTPreprocessor
from util.mask_utils import mask_inputs

@register_dataset(name="masked_tf_dataset")
class MaskedTFDataset(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
        self.cfg = cfg
        self.task_cfg = task_cfg
        manifest_path = cfg.data
        self.dataroot = Path(manifest_path).parent.absolute()
        
        with open(manifest_path, 'r') as file:
            self.datajson = json.load(file)
        # filtered_datajson = []
        # for sample in tqdm(self.datajson, desc='Filter short samples'):
        #     file_name = sample['eeg']
        #     file_path = os.path.join(self.dataroot, file_name)
        #     data = np.load(file_path)

        #     length = data.shape[0]
        #     if length >= 500:
        #         filtered_datajson.append(sample)
        # print(f'After filtering: {len(filtered_datajson)} / {len(self.datajson)}')
        # self.datajson = filtered_datajson
        
        self.cached_features = None

        if preprocessor_cfg.name=="stft":
            extracter = OptimizedSTFTPreprocessor(preprocessor_cfg)
            self.extracter = extracter
        else:
            raise NotImplementedError('NO!')

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["masked_input"].shape[-1]

    def __len__(self):
        return len(self.datajson)

    def __getitem__(self, idx):
        file_name = self.datajson[idx]['eeg']
        file_path = os.path.join(self.dataroot, file_name)
        wav_all_channels = np.load(file_path).astype('float32')
        # CHECK DIMENSIONS

        data_all_channels = torch.tensor(self.extracter(wav_all_channels.T).T)
        # print(data_all_channels.shape)
        
        masked_data_all_channels, mask_label_all_channels = [], []
        for channel in range(wav_all_channels.shape[1]):
            masked_data, mask_label = mask_inputs(data_all_channels[:, :, channel], self.task_cfg)

            masked_data_all_channels.append(masked_data)
            mask_label_all_channels.append(mask_label)

        data_all_channels = data_all_channels.reshape(data_all_channels.shape[0], -1)
        
        masked_data_all_channels = np.stack(masked_data_all_channels).transpose(1, 0, 2)
        masked_data_all_channels = masked_data_all_channels.reshape(masked_data_all_channels.shape[0], -1)
        
        mask_label_all_channels = np.stack(mask_label_all_channels).transpose(1, 0, 2)
        mask_label_all_channels = mask_label_all_channels.reshape(mask_label_all_channels.shape[0], -1)

        return {"masked_input": torch.tensor(masked_data_all_channels),
                "length": data_all_channels.shape[0],
                "mask_label": torch.tensor(mask_label_all_channels),
                "wav": torch.tensor(wav_all_channels),
                "target": data_all_channels,
               }
