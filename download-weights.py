import torch
import diffusers
diffusers.AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16', cache_dir='model')
