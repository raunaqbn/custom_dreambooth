#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#@title Load repo (if needed)
get_ipython().system('git clone https://github.com/JoePenna/Dreambooth-Stable-Diffusion')
get_ipython().run_line_magic('cd', 'Dreambooth-Stable-Diffusion')


# In[ ]:


#@title BUILD ENV
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
get_ipython().system('pip install pydrive')
get_ipython().system('pip install -qq diffusers["training"]==0.3.0 transformers ftfy')
get_ipython().system('pip install -qq "ipywidgets>=7,<8"')
get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install ipywidgets==7.7.1')
get_ipython().system('pip install captionizer==1.0.1')

import os
os._exit(00)


# In[ ]:


#@title # Required - Navigate back to the directory.
get_ipython().run_line_magic('cd', 'Dreambooth-Stable-Diffusion')


# In[3]:


#@markdown Hugging Face Login
from huggingface_hub import login

login("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",True)


# In[ ]:


#@markdown Download the 1.4 sd model
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


# In[2]:


#@title # Download Regularization Images
#@markdown We’ve created the following image sets
#@markdown - `man_euler` - provided by Niko Pueringer (Corridor Digital) - euler @ 40 steps, CFG 7.5
#@markdown - `man_unsplash` - pictures from various photographers
#@markdown - `person_ddim`
#@markdown - `woman_ddim` - provided by David Bielejeski - ddim @ 50 steps, CFG 10.0 <br />
#@markdown - `blonde_woman` - provided by David Bielejeski - ddim @ 50 steps, CFG 10.0 <br />

dataset="person_ddim" #@param ["man_euler", "man_unsplash", "person_ddim", "woman_ddim", "blonde_woman"]
get_ipython().system('git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-{dataset}.git')

get_ipython().system('mkdir -p regularization_images/{dataset}')
get_ipython().system('mv -v Stable-Diffusion-Regularization-Images-{dataset}/{dataset}/*.* regularization_images/{dataset}')


# In[ ]:


#@title # Training Images
#@markdown ## Upload your training images
#@markdown WARNING: Be sure to upload an even amount of images, otherwise the training inexplicably stops at 1500 steps. <br />
#@markdown - 2-3 full body
#@markdown - 3-5 upper body
#@markdown - 5-12 close-up on face  <br /> <br />
#@markdown The images should be as close as possible to the kind of images you’re trying to make (most of the time, that means no selfies).
from google.colab import files
from IPython.display import clear_output

# Create the directory
get_ipython().system('rm -rf training_images')
get_ipython().system('mkdir -p training_images')

# Upload the files
uploaded = files.upload()
for filename in uploaded.keys():
 updated_file_name = filename.replace(" ", "_")
 get_ipython().system('mv "{filename}" "training_images/{updated_file_name}"')
 clear_output()

# Tell the user what is going on
training_images_file_paths = get_ipython().getoutput('find training_images/*')
if len(training_images_file_paths) == 0:
 print("❌ no training images found. Please upload images to training_images")
else:
 print("✅ " + str(len(training_images_file_paths)) + " training images found")


# In[ ]:


#@title # Training

#@markdown This isn't used for training, just to help you remember what your trained into the model.
project_name = "project_name" #@param {type:"string"}

# MAX STEPS
#@markdown How many steps do you want to train for?
max_training_steps = 2000 #@param {type:"integer"}

#@markdown Match class_word to the category of the regularization images you chose above.
class_word = "person" #@param ["man", "person", "woman"] {allow-input: true}

#@markdown This is the unique token you are incorporating into the stable diffusion model.
token = "firstNameLastName" #@param {type:"string"}
reg_data_root = "/content/Dreambooth-Stable-Diffusion/regularization_images/" + dataset

get_ipython().system('rm -rf training_images/.ipynb_checkpoints')
get_ipython().system('python "main.py"  --base configs/stable-diffusion/v1-finetune_unfrozen.yaml  -t  --actual_resume "model.ckpt"  --reg_data_root "{reg_data_root}"  -n "{project_name}"  --gpus 0,  --data_root "/content/Dreambooth-Stable-Diffusion/training_images"  --max_training_steps {max_training_steps}  --class_word "{class_word}"  --token "{token}"  --no-test')


# In[ ]:


#@title # Copy and name the checkpoint file

directory_paths = get_ipython().getoutput('ls -d logs/*')
last_checkpoint_file = directory_paths[-1] + "/checkpoints/last.ckpt"
training_images = get_ipython().getoutput('find training_images/*')
date_string = get_ipython().getoutput('date +"%Y-%m-%dT%H-%M-%S"')
file_name = date_string[-1] + "_" + project_name + "_" + str(len(training_images)) + "_training_images_" +  str(max_training_steps) + "_max_training_steps_" + token + "_token_" + class_word + "_class_word.ckpt"

file_name = file_name.replace(" ", "_")

get_ipython().system('mkdir -p trained_models')
get_ipython().system('mv "{last_checkpoint_file}" "trained_models/{file_name}"')

print("Download your trained model file from trained_models/" + file_name + " and use in your favorite Stable Diffusion repo!")


# In[ ]:


#@title Save model in google drive
from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('cp trained_models/{file_name} /content/drive/MyDrive/{file_name}')

