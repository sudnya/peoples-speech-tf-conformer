'''
Copyright [2022] [Sudnya Diamos]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import jsonlines
import tensorflow as tf
import numpy as np
from pathlib import Path

#from tensorflow_asr.datasets import asr_dataset
from tensorflow_asr.datasets.asr_dataset import ASRDataset

import logging
import json

from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.datasets.base_dataset import AUTOTUNE, BUFFER_SIZE, TFRECORD_SHARDS, BaseDataset
from tensorflow_asr.featurizers.speech_featurizers import (
    SpeechFeaturizer,
    load_and_convert_to_wav,
    read_raw_audio,
    tf_read_raw_audio,
)
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.utils import data_util, feature_util, file_util, math_util

logger = tf.get_logger()


class PeoplesSpeechDataset(ASRDataset):
    """Dataset for ASR using  PeoplesSpeech dataset. Contains jsonlines.

    New Sample Entry:
    {
        "audio_document_id": "CC-2016-0404-BudgetFinance_2of2.mp3",
        "identifier": "CC20160404BudgetFinance2of2",
        "text_document_id": "CC-2016-0404-BudgetFinance_2of2.asr.srt",
        "training_data": {
            "duration_ms": [14080,14700,14650],
            "label": ["blah blah hahaha", "and what else", "who was he going to"],
            "name": [
                "CC20160404BudgetFinance2of2/CC-2016-0404-BudgetFinance_2of2_DOT_mp3_00000.flac",
                "CC20160404BudgetFinance2of2/CC-2016-0404-BudgetFinance_2of2_DOT_mp3_00001.flac",
                "CC20160404BudgetFinance2of2/CC-2016-0404-BudgetFinance_2of2_DOT_mp3_00002.flac"
            ]
        }
    }
    Sample entry: 
    {
        "audio_path": "...path-to-/clean-4-one-two-three.flac", 
        "train": true, 
        "test": false, 
        "uid": "1a5fb807f32c6dbb1d3302793a6c55fe", 
        "labeled": true, 
        "label": "one two three", 
        "image_path": "s3://peoples-speech/datasets/prestamo/uploads/clean-4-one-two-three.png", 
        "label_path": "/Users/sudnya/Desktop/one-two-three-data/da0df04b6ad9112b72839d65a9e2966b.json" 
    }
    """
    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0:
            return
        self.entries = []

        dataset_folder_path = self.data_paths[0].rsplit('/', 1)[0]
        print(dataset_folder_path)

        self.entries = load_peoples_speech(self.data_paths[0])
    
        # The files is "\t" seperated
        self.entries = [line.split("\t", 2) for line in self.entries]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1]).numpy()])
        
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)  # Mix transcripts.tsv
        
        self.total_steps = len(self.entries)
    
    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes):
            return load_and_convert_to_wav(path.decode("utf-8")).numpy()

        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)
        return record[0], audio, record[2]

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)
    
    def extract_label(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
            return '\"' + data['label'] + '\"'

def load_peoples_speech(file_path):
    base_path = file_path.rsplit('/', 1)[0]
    records = []
    with open(file_path) as dataset_file:
        with jsonlines.Reader(dataset_file) as reader:
            for entry in reader:
                dir_path =  base_path + '/' + entry['identifier']
                if Path(dir_path).exists() and Path(dir_path).is_dir():
                    records.append(entry)
    entries = []

    for record in records:
        filenames = record['training_data']['name']
        durations = record['training_data']['duration_ms']
        labels = record['training_data']['label']
        assert len(filenames) == len(durations) == len(labels)
        for i in range(len(filenames)):
            file_path = base_path + '/' + filenames[i]
            if Path(file_path).exists() and Path(file_path).is_file():
                row = file_path + '\t' + str(durations[i]) + '\t\"' + labels[i] + '\"'
                entries.append(row)
    return entries
