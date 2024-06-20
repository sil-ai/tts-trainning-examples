# ORU-Titan-example
Example notebook to run a trainning job in the ORU Titan server, using the GPUs and exporting the model to an S3 bucket.

1. Login in the url https://ood.orca.oru.edu/pun/sys/dashboard with your account sil_username@titan.orca.oru.edu and password (for example aquintero@titan.orca.oru.edu)
2. Go to interactive apps > Jupyter Notebbok ![Notebook](login.png)
3. Launch a session with Partition gpu and some Number of hours
4. git clone this repo
5. Activate git
6. module load git
7. Create a virtual env and register a kernel
8. python -m venv ttsvenv
9. source ttsvenv/bin/activate
10. pip install -r requirements 
11. python -m ipykernel install --user --name=ttsvenv --display-name="ttsvenv"
12. Open the TTS_train notebook with the venv kernel. The end of the notebbok saves a checkpoint of the model.


# Running TTS Notebook

Considerations:

Take into account that this notebook will be able to be used in different environments like
ORU, Google Collab or AWS, for that reason we will only be using relative paths, not absolute,
to make it much more versatile.

The path were the data will be stored is:
Your-Path/data/content 

Here will be:
- tts_manifest.json(audio file's meta data)
- audio files (cache)


# Important cells

## Set CUDA_VISIBLE_DEVICES
This cell is necessary to run the model later, ORU offers several GPU, but the model
has to be run on only one, so we need to specify the variable:

```
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
```

## Function change_to_base_directory

This is a util function that allow you to return to your base directory, in case you need it, a variable base_path is set at the beggining of the notebook, so we know where we 
started and where is everything located 

```
import os
base_path = os.getcwd()
def change_to_base_directory(base_path):

    # Get current directory
    current_directory = os.getcwd()

    # Check if it is the desired directory(base)
    if current_directory == base_path:
        print("You already are on:", base_path)
    else:
        print("Changing to:", base_path)
        os.chdir(base_path)  # Change to base
        print("New path:", os.getcwd())

```


## Dataset Loading

This cell will load the dataset, we define a cache directory where everything will be 
downloaded, if you have already downloaded the dataset and run this cell, it will not 
download it again.

```
from datasets import load_dataset, Audio

import os

cache_dir = 'data/content/cache'
dataset = load_dataset("mozilla-foundation/common_voice_16_1", lang, split="train", token=True, trust_remote_code=True, cache_dir = 'data/content/cache', streaming=False, download_mode='force_redownload') # streaming = True ?
len(dataset)
```

However sometimes it is necessary to load it again, maybe because
you were running a cell that was modifying the dataset and it stopped halfway through,
so the data lost its integrity, in that case you can specify the following option:

```
download_mode='force_redownload'
```

## Pushing Dataset to HF Hub

This cell allows you to upload the dataset you have on local to the HF Hub, it is recommended that you keep this cell commented. This step only needs to be done when there's an update to the dataset.
```
dataset.push_to_hub(f'sil-ai/{lang_iso}-tts-training-data', private=True, token=True)
```