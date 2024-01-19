from diffusers import AutoPipelineForText2Image
import torch
from tqdm import tqdm
from functools import cache


pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant="fp16").to('cuda')
attributes = {
    'smiling': [
        ('a closeup portrait of a person', 'a closeup portrait of a smiling person'),
        ('a dog in a park', 'a dog in a park, smiling'),
        ('a photo of construction worker', 'a photo of construction worker, smiling'),
        ('a kid on the playground', 'a kid on the playground, smiling'),
    ],
    'old': [
        ('a closeup portrait of a young person', 'a closeup portrait of an old person'),
        ('a young dog in a park', 'an old dog in a park'),
        ('a photo of a young construction worker', 'a photo of an old construction worker'),
        ('a kid on the playground', 'a person on the playground'),
    ],
    'cyberpunk': [
        ('a closeup portrait of a person', 'a closeup portrait of a person, cyberpunk'),
        ('a dog in a park', 'a dog in a park, cyberpunk'),
        ('a photo of construction worker', 'a photo of construction worker, cyberpunk'),
        ('a kid on the playground', 'a kid on the playground, cyberpunk'),
    ]
}


@cache
def get_representation(text, count):
    '''Get the representation for a given prompt.'''
    reprs = []
    def get_repr(module, input, output):
        # take mean over spatial dimensions
        reprs.append(output.detach())
    for i in range(count):
        with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(get_repr):
            pipe(prompt = text, num_inference_steps = 1, guidance_scale=0.0)
    return torch.stack(reprs).mean(axis=0).cpu()


def save_directions():
    directions = torch.zeros((len(attributes), 1280, 16, 16), device='cuda')
    for i, (name, pairs) in enumerate(tqdm(attributes.items(), desc='generating attribute directions')):
        diffs = []
        for negative, positive in tqdm(pairs, leave=False):
            negative_repr = get_representation(negative, 20)
            positive_repr = get_representation(positive, 20)
            diffs.append(positive_repr - negative_repr)
        directions[i] = torch.mean(torch.stack(diffs), axis=0)
    torch.save(directions, 'directions.pt')


if __name__ == '__main__':
    save_directions()
