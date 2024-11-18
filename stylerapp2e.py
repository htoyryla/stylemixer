import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from torch import autocast
import random
from kornia.geometry import resize
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter
import torchvision.models as models
from PIL import Image, ImageEnhance

from kornia.color import hsv_to_rgb, rgb_to_hsv
from kornia.filters import bilateral_blur
from kornia.enhance import equalize, equalize_clahe, adjust_contrast, adjust_brightness
import argparse

from einops import rearrange

import gradio

# use command line parameters
parser = argparse.ArgumentParser()

# define params and their types with defaults if needed
parser.add_argument('--auth',  type=str, default="", help='use login authentication')
parser.add_argument('--subpath', type=str, default="", help='url subpath')
#parser.add_argument('--niter', type=int, nargs='*', default=[400, 400, 400, 400, 300, 300, 300], help='number of iterations')
parser.add_argument('--imageSize', type=int, nargs='*', default=[140, 256, 480, 640, 800, 960, 1200], help='image size')
parser.add_argument('--cuda', action="store_true", help='use cuda if available')
parser.add_argument('--style_layers', type=int, nargs='*', default=[2, 3, 5, 7, 9, 11], help='style layers indices')
parser.add_argument('--content_layer', type=int, default=10, help='content layer index')

# get params into opt object, so we can access them like opt.image
opt = parser.parse_args()




torch.cuda.empty_cache()

def eq2(x, v):
    x -= x.min()
    x /= x.max()
    x = equalize_clahe(x, clip_limit = v, grid_size = (8,8))
    x = x * 2 - 1
    return x  

cuda = opt.cuda

if cuda:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
  device = torch.device("cpu")

cuda = opt.cuda

if cuda:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
  device = torch.device("cpu")
  
xforms = []
xforms.append(Resize(opt.imageSize[-1]))    # resize image

xforms.append(ToTensor())            # convert to pytorch tensor
xforms.append(Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))            # normalize to range -1...1
  

preprocess = Compose(xforms)

vgg = models.vgg19(pretrained=True).features.to(device).eval()

def get_optimizer(im, lr):
    optimizer = torch.optim.LBFGS([im.requires_grad_()])
    return optimizer
	
# list of suitable layers for content/style evaluation

players = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]

content_idx = opt.content_layer
#style_idxs = opt.style_layers #[1, 3, 5, 9, 13, 15] 

content_layer = vgg[players[content_idx]]

print(content_layer) 

# add a hook to read style evaluation from selected layers

# use a gram matrix to evaluate style (texture) instead of content

def gram(input):
    a, b, c, d = input.size() 
    f = input.clone().reshape(a * b, c * d)  # resise F_XL into \hat F_XL
    gr = torch.mm(f, f.t())  # compute the gram product
    return gr.div(a * b * c * d)
    

# now add the actual style hooks    

#style_grams = [None]*len(style_idxs)



#------------------


def run(cimg, simg, cw_, nw_, contrast, brightness, color, sharpness, sw_, scontrast, sbrightness, chmax, chinv, chfactor, minres, maxres, niter, style_idxs):
  global opt    

  print(cw_, nw_, chmax, chinv, chfactor, slayers)

  resolutions = [80, 120, 160, 240, 320, 440, 560, 680, 800, 960, 1200]

  resolutions =  [x for x in resolutions if x >= minres]
  resolutions =  [x for x in resolutions if x <= maxres]

  niters = [niter]*len(resolutions)

  if niter >= 200:
        split = int(len(niters)/2)
        print(split)
        niters = niters[:split] + [x - 100 for x in niters[split:]]    

  print(resolutions, niters)

  # read and  preprocess target and style images

  cimg = cimg.convert("RGB")
  
  if contrast != 1:
      cimg = ImageEnhance.Contrast(cimg).enhance(contrast)
  if brightness != 1:      
      cimg = ImageEnhance.Brightness(cimg).enhance(brightness)
  if color != 1:
      cimg = ImageEnhance.Color(cimg).enhance(color)
  if sharpness != 1:
      cimg = ImageEnhance.Sharpness(cimg).enhance(sharpness)
  
  cimg = preprocess(cimg)

  imgC = cimg.to(device).clamp_(-2,2)
  
  #if opt.ceq > 0:
  #    imgC = eq2(imgC, opt.ceq)

  # We are going to optimize imgG, so we need gradients for it

  #lr = opt.lr #0.05 # you might want to change this and see what happens

  imgG = imgC.clone().detach()
  imgG.requires_grad = True

  content_start = imgC.unsqueeze(0)

  iters = zip(resolutions, niters)
    
  hooks = []

  first = True     
  k = 0
  for size, niter in iters: 
      
    ims = None  
    
    print("    resizing to ", size)
    content_start = resize(content_start, size)

    print("    ", k, size, niter, content_start.shape)
    
    
    if len(hooks) > 0:
        for hk in hooks:
            hk.remove()

    # add a hook to read output of content layer

    content_acts = [None]
    def content_hook(i):
        def hook(model, input, output):
            content_acts[i] = output
        return hook
        
    hk = content_layer.register_forward_hook(content_hook(0))
    hooks.append(hk)

    o = vgg(content_start)  # tässä virhe? nyt content target on edellisen kierroksen tulos eikä alkuperäinen content image
    content_targets = content_acts[0].detach() #.shape  
    
    # feed style image to VGG and store outputs from style hooks
    
    xsforms = []
    stylesize = size #int(opt.style_scale * size)
    xsforms.append(Resize(stylesize))    # resize image

    if brightness != 1 or contrast != 1:
        xsforms.append(ColorJitter(brightness=(sbrightness, sbrightness), contrast=(scontrast, scontrast)))

    xsforms.append(ToTensor())            # convert to pytorch tensor
    xsforms.append(Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))            # normalize to range -1...1

    # compose into a transform pipeline

    style_preprocess = Compose(xsforms)

    imgS = style_preprocess(simg).to(device).clamp_(-2,2)

    filterOn = True
    
    #if opt.seq > 0:
    #    imgS = eq2(imgS, opt.seq)

    def style_hook(i):
        def hook(model, input, output):
            #print(i, output.shape)
            nonlocal chmax, chfactor, chinv, filterOn

            n = output.shape[1]
            if filterOn and chmax > 1:
                top_n = int(n/chmax)
                #print(i, output.shape, n, top_n)
                l1_norms= torch.sum(torch.abs(output), dim=(2, 3))[0]
                #print(l1_norms.shape)
                sorted_indices = torch.argsort(l1_norms, descending=True)
                top_channels = sorted_indices[:top_n]
                #print(sorted_indices.shape, top_channels.shape)
                if chinv == "attenuate top":
                    scaling_factors = torch.ones(n, device="cuda") * chfactor  # Attenuate by a factor of 0.5 by default
                    scaling_factors[top_channels] = 1 * (1 / chfactor)
                else:    
                    scaling_factors = torch.ones(n, device="cuda") * (1 / chfactor)  # Attenuate by a factor of 0.5 by default
                    scaling_factors[top_channels] = 1 * chfactor
                scm = scaling_factors.mean()
                scaling_factors = scaling_factors/scm
                scaling_factors = scaling_factors.view(1, n, 1, 1)
                #print(scaling_factors.shape, output.shape, scaling_factors.mean(()))
                style_grams[i] = gram(output * scaling_factors)
            else:
                style_grams[i] = gram(output)
                
        return hook

    n = 0
    for s in style_idxs:
        style_layer = vgg[players[s]]    
        hk = style_layer.register_forward_hook(style_hook(n))
        hooks.append(hk)
        n += 1   

    style_targets = [None]*len(style_idxs)
    style_grams = [None]*len(style_idxs)
    o = vgg(imgS.unsqueeze(0))
    for n in range(len(style_idxs)):      
        style_targets[n] = style_grams[n].detach() #.shape        

    del imgS, o

    # get ready to iterate
    if nw_ > 0 and first:
        noise = torch.zeros_like(content_start).normal_(0,1).to(device)
        start = noise * nw_ + content_start * (1 - nw_)
    else:
        start = content_start    
    
    first = False
    
    imgG = start.detach()    
    optimizer = get_optimizer(imgG, 0.05)
    
    run = [0]
    
    def closure():
         nonlocal k, size, cw_, sw_, ims, run, content_start, filterOn
         #print(k, run[0], size)
        
         optimizer.zero_grad()
 
         imgG_ = imgG.clone().clamp_(-2,2)
		 #imgOut = torch.zeros()

         filterOn = False

         o = vgg(imgG_)
 
         content_actuals = content_acts[0]
         
         style_actuals = [None]*len(style_idxs)
 
         # evaluate content loss
 
         lossc = float(cw_) * F.mse_loss(content_targets, content_actuals)

         # evaluate total style loss

         style_losses = []
         losss = 0
         for n in range(len(style_idxs)):      
             style_actuals[n] = style_grams[n]
             sl = float(sw_) * F.mse_loss(style_targets[n], style_actuals[n])
             style_losses.append(sl.item())
             losss += sl

         # additional loss to keep pixel values in range
 
         out_of_range_loss = F.mse_loss(imgG, imgG.clamp(-2, 2))

         loss = lossc + losss + out_of_range_loss

         # run backwards to find gradient (how to change imgG to make loss smaller) 
         loss.backward()
 
         run[0] += 1

         if run[0] % 20 == 0:
             imgOut = imgG.clone().detach().clamp_(-2,2)[0]
             imgOut = imgOut - imgOut.min()
             imgOut = imgOut / imgOut.max()
             x_out = 255. * rearrange(imgOut.cpu().numpy(), 'c h w -> h w c')
             ims = Image.fromarray(x_out.astype(np.uint8))             
             #print(run[0])
             
             if run[0] >= niter:
                 print("    end", size)
                 content_start = imgG.detach().clamp_(-2,2)
                 k = k + 1
                 
         return loss    
    
    # then go

    i = 0
    while run[0] <=  niter:
             
       # now actually run a single optimizer step

       optimizer.step(closure)
       #print(run[0])
       if run[0] % 20 == 0:
           #print(ims)
           status = "Size "+str(size)+", iter "+str(run[0])
           print(status)
           yield ims, status
       i = i + 1

  status = "Ready"    
  print(status)
  yield ims, status

css = """
#outimg {overflow: visible !important; height: 100%;  }
#outimg.img {max-height: 100%; ,max-width: 100%;object-fit: contain;}
#output {max-width: 800px !important;}
#input {max-width: 400px ! important;}
#inputs {max-width: 400px ! important;}
.inputimg {overflow: visible !important; height: 400px;  }
.inputimg.img {max-height: 100%; ,max-width: 100%;object-fit: contain;}
.gradio-container {background-color: #ddd} 
gradio-app {background-color: #ccc !important}
"""

import gradio as gr
from functools import partial
from itertools import chain

with gr.Blocks(title="#stylemixer", css=css) as demo:

    logged_user = gr.State(value="???")

    btns = []
    txts = []
    ims = []
    strengths = []
    
    with gr.Row():
        html = '<div style="font-size: 36px; color:#666666;">#stylemixer</div>'+  \
        '<div style="font-size: 14px; color:#666666;">by @htoyryla 2023-2024</div>'
        logo = gr.HTML(html)
    with gr.Row():
        with gr.Column(scale=1, elem_id="input"):
                cimg = gr.Image(label="Content image", interactive=True, visible=True, type="pil", elem_classes="inputimg")
                cw = gr.Slider(label="Weight", minimum=0, maximum=0.4, step=0.002, value=0.01, interactive=True)
                nw = gr.Slider(label="Noise", minimum=0, maximum=1, step=0.02, value=0., interactive=True)
                contrast = gr.Slider(label="Contrast", minimum=0.4, maximum=2., step=0.1, value=1, interactive=True, visible=True)
                brightness = gr.Slider(label="Brightness", minimum=0.4, maximum=2., step=0.1, value=1, interactive=True, visible=True)
                color = gr.Slider(label="Color", minimum=0, maximum=2., step=0.1, value=1, interactive=True, visible=True)
                sharpness = gr.Slider(label="Sharpness", minimum=0, maximum=2., step=0.1, value=1, interactive=True, visible=True)
        with gr.Column(scale=1, elem_id="inputs"):        
                simg = gr.Image(label="Style image", interactive=True, visible=True, type="pil", elem_classes="inputimg")
                sw = gr.Slider(label="Weight", minimum=1000, maximum=100000, step=100, value=10000, interactive=True)
                scontrast = gr.Slider(label="Contrast", minimum=0.4, maximum=2., step=0.1, value=1, interactive=True, visible=False)
                sbrightness = gr.Slider(label="Brightness", minimum=0.4, maximum=2., step=0.1, value=1, interactive=True, visible=False)
                chmax = gr.Dropdown(choices=[("None", 1), ("1/2",2), ("1/4",4),("1/8",8)], value=1,label="Filter features")
                chinv = gr.Radio(choices=["boost top","attenuate top"], value="boost top", label="Filter type")
                chfactor =  gr.Slider(label="Filter level", minimum=1., maximum=10., step=0.1, value=1., interactive=True)
        with gr.Column(elem_id="output"):
          with gr.Row():
            minres = gr.Dropdown(choices=[80, 120, 160, 240,320,480], value=160,label="Start reso")
            maxres = gr.Dropdown(choices=[680, 800,960,1200], value=800,label="Final reso")
            niter = gr.Dropdown(choices=[100, 200,300,400], value=200,label="Quality")
          with gr.Row():  
            slayers = gr.Dropdown([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], value=opt.style_layers, multiselect=True, label="Style layers", info="low values: color, middle: texture, high: special effects")
            with gr.Column():
                process_status = gr.Textbox(label="Status", value="Ready")
                submit = gr.Button("Generate", elem_id ="generate-button")
          with gr.Row():    
            output = gr.Image(label="Output", elem_id="outimg") #, width=800, height=800) #, width="640px")
    

    inps = [cimg, simg, cw, nw, contrast, brightness, color, sharpness, sw, scontrast, sbrightness, chmax, chinv, chfactor, minres, maxres, niter, slayers]
    submit.click(fn=run, inputs=inps, outputs=[output, process_status], concurrency_limit=1)
        
    def auth(u, p):
        global logged_user
        if opt.auth == "":
            return True
        
        filename = opt.auth  #"umb.s"
        with open(filename,"rt") as f:
             while True: 
                 text = f.readline()
                 if not text:
                     break
                 u_, p_ = text.strip().split(":")
                 #print(text, u_, p_)    
                 if u == u_ and p == p_:
                     print("logged in ",u_)
                     logged_user.value=u_
                     return True
         
        return False

demo.queue()
demo.launch(server_name="0.0.0.0") #, auth=auth, root_path=opt.subpath)
