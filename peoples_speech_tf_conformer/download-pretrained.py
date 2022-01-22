import gdown

file_id = 'https://drive.google.com/file/d/1p46Ny6IVVdEmXUYhtj_X_EGgJIlAfh3C/view?usp=sharing'
local_destination = '/tmp/pretrained-subword-conformer.zip'

url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
output = '20150428_collected_images.tgz'
gdown.download(file_id, local_destination, quiet=False)


'''

import requests
import zipfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def fixBadZipfile(zipFile):  
    extracted_dir = '/tmp/pretrained-subword-conformer'
    f = open(zipFile, 'r+b')  
    data = f.read()  
    pos = data.find(b'\x50\x4b\x05\x06') # End of central directory signature  
    if (pos > 0):  
        print("Trancating file at location " + str(pos + 22)+ ".")  
        f.seek(pos + 22)   # size of 'ZIP end of central directory record' 
        f.truncate()  
        f.close()  
    else:  
        # raise error, file is truncated  
        pass
    return 


if __name__ == "__main__":
    file_id = 'https://drive.google.com/file/d/1p46Ny6IVVdEmXUYhtj_X_EGgJIlAfh3C/view?usp=sharing'
    local_destination = '/tmp/pretrained-subword-conformer.zip'
    
    download_file_from_google_drive(file_id, local_destination)
    fixBadZipfile(local_destination)
    with zipfile.ZipFile(local_destination, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

    print("Extracted at {} ".format(extracted_dir))
'''