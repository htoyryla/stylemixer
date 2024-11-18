# stylemixer
An image mixer building upon the good old neural style transfer 


# installation

conda create -n style python==3.10
conda activate style
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install gradio
pip install kornia
pip install einops

# run

conda activate style
python stylerapp2e.py --cuda

Then use your browser to go to the given url.

#

![Näyttökuva 2024-11-17 kello 14 05 29](https://github.com/user-attachments/assets/891f383b-dcbe-4adb-b9bd-8229fc332e5b)
