[People’s Speech](https://mlcommons.org/en/peoples-speech/) is among the world’s largest English speech recognition datasets licensed for academic and commercial use. It includes 30,000+ hours of transcribed speech in English with a diverse set of speakers and acoustic environments. This open dataset by [MLCommons.org](https://mlcommons.org/en/) is large enough to train speech-to-text systems and is available with a permissive license.

In this tutorial, we will learn to use this dataset to fine tune a pre-trained model. We’ll train this model locally and leverage pre-trained models released by [TensorflowASR] (https://github.com/TensorSpeech/TensorFlowASR).

[#TODO - maybe add a table of contents and wrap it in some summary]

# 1. Install peoples-speech-tf-conformer
1. Clone the [peoples-speech-tf-conformer repo] (https://github.com/sudnya/peoples-speech-tf-conformer) using the following command
 `git clone https://github.com/sudnya/peoples-speech-tf-conformer.git`
2. Create a new virtual environment using the following command
`python3 -m venv ~/environments-virtual/your-preferred-environment-name`
3. Activate the virtual environment created in the above step using the following command
`source ~/environments-virtual/your-preferred-environment-name/bin/activate`
4. Install the [peoples-speech-tf-conformer] as follows
`pip install git+git://github.com/sudnya/peoples-speech-tf-conformer`

# 2. Download the pretrained model
The [TensorflowASR repository](https://github.com/TensorSpeech/TensorFlowASR) provides a pre-trained model of the subword Conformer ready to be downloaded from [here.](https://drive.google.com/drive/folders/1VAihgSB5vGXwIVTl3hkUk95joxY1YbfW)

Download all the files in the above link to your local machine.

# 3. Download the dataset
## TODO: currently only uses a tiny file

# 4. Update the config file
An example configuration file can be found [here] (https://github.com/sudnya/peoples-speech-tf-conformer/blob/master/peoples-speech-dataset-config.yml)

1. Update the path to the vocabulary file under the `decoder_config` section to the file downloaded in the pretrained model step above.
`vocabulary: ~/path-to-the-pretrained-subword-conformer/conformer.subwords`
2. Update the path to the `corpus_files` in the `decoder_config` to the dataset file for the dataset mentioned in the [download the dataset section](#3. Download the dataset). Note that the dataset file is expected to be jsonlines. Here is an example line that represents the expected format

```{"audio_path": "/path-to-the-audio-file.mp4", "train": true, "test": false, "uid": "1a5fb807f32c6dbb1d3302793a6c55fe", "labeled": true, "label": "one two three", "image_path": "s3://can-be-empty.png", "label_path": "/path-to-the-/da0df04b6ad9112b72839d65a9e2966b.json" }
```

The label path file expects the following format
`{"label": "one two three"}`
3. Update the path to the `data_paths` in the `test_dataset_config` to point to the dataset file.

# 5. Run pretrained model to evaluate on dataset
Run the pretrained model downloaded [in an earlier step] (# 2. Download the pretrained model) in predict mode on the dataset downloaded in the [download dataset step] (# 3. Download the dataset) with the the configuration file [described above] (# 4. Update the config file).

 `~> python peoples_speech_tf_conformer/run-peoples-speech.py --saved ~/path-to-the-pretrained-subword-conformer/latest.h5 --subwords --config ~/path-to-config/peoples-speech-dataset-config.yml`
 
# 6. Understanding the results
#### TODO