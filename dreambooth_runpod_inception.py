#!/usr/bin/env python
# coding: utf-8

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument('--project_name', default='my_project', help='name of the project')
parser.add_argument('--token_name', default='firstNameLastName', help='name of the person you are training')
parser.add_argument('--training_type', default='supervised', help='type of training to use')
parser.add_argument('--number_of_steps', default=2000, type=int, help='number of steps to run the project')

# Parse the command line arguments
args = parser.parse_args()

# You can access the arguments using the attribute notation
project_name = args.project_name
training_type = args.training_type
token_name = args.token_name
number_of_steps = args.number_of_steps

# Now you can use the arguments in your script
print("Running project " +  project_name + " for token name " + token_name + " with " +  training_type + " training for " + str(number_of_steps) + " steps")


# BUILD ENV
get_ipython().system('pip install omegaconf')
get_ipython().system('pip install einops')
get_ipython().system('pip install pytorch-lightning==1.6.5')
get_ipython().system('pip install test-tube')
get_ipython().system('pip install transformers')
get_ipython().system('pip install kornia')
get_ipython().system('pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers')
get_ipython().system('pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip')
get_ipython().system('pip install setuptools==59.5.0')
get_ipython().system('pip install pillow==9.0.1')
get_ipython().system('pip install torchmetrics==0.6.0')
get_ipython().system('pip install -e .')
get_ipython().system('pip install protobuf==3.20.1')
get_ipython().system('pip install gdown')
get_ipython().system('pip install -qq diffusers["training"]==0.3.0 transformers ftfy')
get_ipython().system('pip install -qq "ipywidgets>=7,<8"')
get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install ipywidgets==7.7.1')
get_ipython().system('pip install captionizer==1.0.1')


# In[ ]:


# Hugging Face Login
from huggingface_hub import login

login("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",True)

# In[ ]:


# Download the 1.4 sd model
from IPython.display import clear_output

from huggingface_hub import hf_hub_download
downloaded_model_path = hf_hub_download(
 repo_id="CompVis/stable-diffusion-v-1-4-original",
 filename="sd-v1-4.ckpt",
 use_auth_token=True
)

# Move the sd-v1-4.ckpt to the root of this directory as "model.ckpt"
actual_locations_of_model_blob = get_ipython().getoutput('readlink -f {downloaded_model_path}')
get_ipython().system('mv {actual_locations_of_model_blob[-1]} model.ckpt')
clear_output()
print("✅ model.ckpt successfully downloaded")


#
# # Download pre-generated regularization images
# We've created the following image sets
#
# `man_euler` - provided by Niko Pueringer (Corridor Digital) - euler @ 40 steps, CFG 7.5
# `man_unsplash` - pictures from various photographers
# `person_ddim` - this is the only one which we will use
# `woman_ddim` - provided by David Bielejeski - ddim @ 50 steps, CFG 10.0
# `person_ddim` is recommended

# Raunaq: We will use only person_ddim for persons, and dog
#Download Regularization Images

# Keeping the dataset as person by default
dataset = "person_ddim"
if (training_type == "dog"):
    dataset = "dog"
elif (training_type == "person"):
    dataset = "person_ddim"

if (training_type == "dog"):
    get_ipython().system('git clone https://github.com/raunaqbn/Stable-Diffusion-Regularization-Images-dog.git')
elif (training_type == "person"):
    get_ipython().system('git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-{dataset}.git')

get_ipython().system('mkdir -p regularization_images/{dataset}')
get_ipython().system('mv -v Stable-Diffusion-Regularization-Images-{dataset}/{dataset}/*.* regularization_images/{dataset}')


# # Upload your training images
# Upload 10-20 images of someone to
#
# ```
# /workspace/Dreambooth-Stable-Diffusion/training_images
# ```
#
# WARNING: Be sure to upload an *even* amount of images, otherwise the training inexplicably stops at 1500 steps.
#
# *   2-3 full body
# *   3-5 upper body
# *   5-12 close-up on face
#
# The images should be:
#
# - as close as possible to the kind of images you're trying to make

# In[ ]:


#@markdown Add here the URLs to the images of the subject you are adding
urls = [
 "https://imgur.com/vnKY6e3.png",
 "https://imgur.com/FgAhNST.png",
 "https://imgur.com/YnVP4Ws.png",
 "https://imgur.com/iLqto4x.png",
 "https://imgur.com/vyFSAqG.png",
 "https://imgur.com/DwY5Acp.png",
 "https://imgur.com/5coSpkz.png",
 "https://imgur.com/AavrvQG.png",
 "https://imgur.com/AdqNJxC.png",
 "https://imgur.com/56UbkoZ.png",
 "https://imgur.com/4codIbW.png",
 "https://imgur.com/HsB1xlk.png",
 "https://imgur.com/bcd9iTX.png",
 "https://imgur.com/DIIFt85.png",
 "https://imgur.com/KlyoQ8F.png",
 "https://imgur.com/ohtD2oy.png",
 "https://imgur.com/yPbejcI.png",
 "https://imgur.com/uQKXA1v.png",
 "https://imgur.com/Gr28Bjf.png",
 "https://imgur.com/yeT0wAm.png",
 # You can add additional images here -- about 20-30 images in different
]

# In[ ]:


#@title Download and check the images you have just added
import os
import requests
from io import BytesIO
from PIL import Image


def image_grid(imgs, rows, cols):
 assert len(imgs) == rows*cols

 w, h = imgs[0].size
 grid = Image.new('RGB', size=(cols*w, rows*h))
 grid_w, grid_h = grid.size

 for i, img in enumerate(imgs):
  grid.paste(img, box=(i%cols*w, i//cols*h))
 return grid

def download_image(url):
 try:
  response = requests.get(url)
 except:
  return None
 return Image.open(BytesIO(response.content)).convert("RGB")

images = list(filter(None,[download_image(url) for url in urls]))
save_path = "./training_images"
if not os.path.exists(save_path):
 os.mkdir(save_path)
[image.save(f"{save_path}/{i}.png", format="png") for i, image in enumerate(images)]
image_grid(images, 1, len(images))


# ## Training
#
# If training a person or subject, keep an eye on your project's `logs/{folder}/images/train/samples_scaled_gs-00xxxx` generations.
#
# If training a style, keep an eye on your project's `logs/{folder}/images/train/samples_gs-00xxxx` generations.

# In[ ]:


# Training



# MAX STEPS
# How many steps do you want to train for.
max_training_steps = number_of_steps
class_word = "person"
# Match class_word to the category of the regularization images you chose above.
if (training_type == "dog"):
    class_word = "dog" # typical uses are "man", "person", "woman"
elif (training_type == "person"):
    class_word = "person"

# This is the unique token you are incorporating into the stable diffusion model.
token = token_name


reg_data_root = "./regularization_images/" + dataset

get_ipython().system('rm -rf training_images/.ipynb_checkpoints')
get_ipython().system('python "main.py"  --base configs/stable-diffusion/v1-finetune_unfrozen.yaml  -t  --actual_resume "model.ckpt"  --reg_data_root "{reg_data_root}"  -n "{project_name}"  --gpus 0,  --data_root "./training_images"  --max_training_steps {max_training_steps}  --class_word "{class_word}"  --token "{token}"  --no-test')


# ## Copy and name the checkpoint file

# In[ ]:


# Copy the checkpoint into our `trained_models` folder

directory_paths = get_ipython().getoutput('ls -d logs/*')
last_checkpoint_file = directory_paths[-1] + "/checkpoints/last.ckpt"
training_images = get_ipython().getoutput('find training_images/*')
date_string = get_ipython().getoutput('date +"%Y-%m-%dT%H-%M-%S"')
file_name = date_string[-1] + "_" + project_name + "_" + str(len(training_images)) + "_training_images_" +  str(max_training_steps) + "_max_training_steps_" + token + "_token_" + class_word + "_class_word.ckpt"

file_name = file_name.replace(" ", "_")

get_ipython().system('mkdir -p trained_models')
get_ipython().system('mv "{last_checkpoint_file}" "trained_models/{file_name}"')

print("Download your trained model file from trained_models/" + file_name + " and use in your favorite Stable Diffusion repo!")


# # Optional - Upload to google drive
# * run the following commands in a new `terminal` in the `Dreambooth-Stable-Diffusion` directory
# * `chmod +x ./gdrive`
# * `./gdrive about`
# * `paste your token here after navigating to the link`
# * `./gdrive upload trained_models/{file_name.ckpt}`

# # Big Important Note!
#
# The way to use your token is `<token> <class>` ie `joepenna person` and not just `joepenna`

# ## Generate Images With Your Trained Model!

# In[ ]:


#get_ipython().system('python scripts/stable_txt2img.py  --ddim_eta 0.0  --n_samples 1  --n_iter 4  --scale 7.0  --ddim_steps 50  --ckpt "./trained_models/{file_name}"  --prompt "joepenna person as a masterpiece portrait painting by John Singer Sargent in the style of Rembrandt"')
