# stylemixer
An image mixer building upon the good old neural style transfer 


# installation

```
conda create -n style python==3.10
conda activate style
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install gradio
pip install kornia
pip install einops
```

# run

```
conda activate style
python stylerapp2e.py --cuda
```


Then use your browser to go to the given url.

# basic operation

![Näyttökuva 2024-11-17 kello 14 05 29](https://github.com/user-attachments/assets/891f383b-dcbe-4adb-b9bd-8229fc332e5b)

Give a content image and a style image. Adjust their weights. A good guess is to use a very low weight for the content image and use the style weight find a good balance.

Start reso is important as it determines the scale of style features in the final image. Good values are between 80 and 240. 

The algorithm is based on using a VGG19 model as a feature detector. You can select which layers are being used to evaluate and transfer style. Lower layers respond to color and texture, middle layers to various shapes an objects, and the highest layers tend to produce deep dream like effects.
