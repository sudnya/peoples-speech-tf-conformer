
import jsonlines
from pathlib import Path

#data_identifier = "CC20160404BudgetFinance2of2"

'''
entries read
['/Users/sudnya/Desktop/one-two-three-data/clean-4-one-two-three.flac\t5\t"one two three"']



entries with features
[['/Users/sudnya/Desktop/one-two-three-data/clean-4-one-two-three.flac', '5', '809 71 319 494 876 809']]



at the end of function
[['/Users/sudnya/Desktop/one-two-three-data/clean-4-one-two-three.flac'
  '5' '809 71 319 494 876 809']]
'''

# TODO: user supplies path to manifest file and path to downloaded dataset (extracted)
# use the right most dir name in dataset path as identifier to search for
# in the manifest file 

def convert_to_required_format(base_dir, filenames, durations, labels):
    entries = []
    assert len(filenames) == len(durations) == len(labels)
    for i in range(len(filenames)):
        row = base_dir + filenames[i] + '\t' + str(durations[i]) + '\t' + labels[i]
        entries.append(row)
    return entries


def load_peoples_speech(file_path):
    base_path = file_path.rsplit('/', 1)[0]
    records = []
    with open(file_path) as dataset_file:
        with jsonlines.Reader(dataset_file) as reader:
            for entry in reader:
                dir_path =  base_path + '/' + entry['identifier']
                if Path(dir_path).exists() and Path(dir_path).is_dir():
                    print('Found it')
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
                print('exists ' + file_path) 
                row = file_path + '\t' + str(durations[i]) + '\t\"' + labels[i] + '\"'
                entries.append(row)
    return entries


peoples_speech_manifest = "/Users/sudnya/Downloads/peoples-speech-dataset/part-00000-4e132642-c01c-4db6-9db0-a1e19193f6f8-c000.json"
dataset_folder_path = peoples_speech_manifest.rsplit('/', 1)[0]
print(dataset_folder_path)

entries = load_peoples_speech(peoples_speech_manifest)
#entries = convert_to_required_format(dataset_folder_path, rec['training_data']['name'], rec['training_data']['duration_ms'], rec['training_data']['label'])
print(entries)



