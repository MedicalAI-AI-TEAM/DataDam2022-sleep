import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import json


class DataDamloader(Dataset):
    def __init__(
        self,
        table=pd.DataFrame([]),
        label="label",
    ):
        self.table = table.reset_index(drop=True)
        self.len = len(table)
        self.label = label

        self.lead_types = [
            "I",
            "II",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]

        self.num_leads = 8
        self.num_aux = 0
        self.sampling_rate = 500
        self.second = 10
        self.use_lead_length = int(self.sampling_rate * self.second)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.table.iloc[index]
        label = row[self.label]
        label = torch.tensor(label, dtype=torch.float32)
        ecg_path = os.path.join("../label", row["mai_file_name"])
        obj_id = ecg_path.split(".")[0]

        ecg = self.preprocessing(ecg_path)

        ecg = torch.tensor(ecg, dtype=torch.float32)
        return obj_id, ecg, label

    def preprocessing(self, ecg_path):
        ecg_dict = self._get_ecg_json(ecg_path)

        # parsing ecg json
        ecg = []

        ecg = np.array(ecg_dict["waveform"]["data"]["I"])
        ecg = ecg.reshape(300, -1)
        ecg = ecg[:, :20000]
        ecg = (ecg - np.nanmean(ecg)) / (np.nanstd(ecg) + 1e-5)
        return ecg

    def _get_ecg_json(self, ecg_json_path):
        with open(ecg_json_path, "r", encoding="UTF-8") as ecg_json_file:
            ecg_str = ecg_json_file.read().replace("\x00", "")
            ecg_json = json.loads(ecg_str)

        return ecg_json
