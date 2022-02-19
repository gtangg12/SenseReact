import torch
from CLIP import clip
from PIL import Image
from tqdm import tqdm
import numpy as np

model, preprocess = clip.load("ViT-L/14", device='cuda')

img_path = lambda i: f'/nobackup/users/wzhao6/treehacks2022/imgs_sample/frame_{i}.jpg'
device = 'cuda'
n_imgs = 4263
caption_every = 10
batch_size = 16

#image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#text = clip.tokenize(["a diagram", "a dog", "a cat", "a person is talking", "two people are fighting", "they're having sex", "a person is eating", "two people are having sex"]).to(device)

with torch.no_grad():
    #logits_per_image, logits_per_text = model(image, text)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    img_tokens = []
    for i in range(0, n_imgs, 10):
        img_tokens.append(preprocess(Image.open(img_path(i))))
        print('image preprocess', i)

    img_embeds = []
    for batch_idx in tqdm(range(n_imgs//(16*10))):
        batch = torch.stack(img_tokens[(batch_idx*16):(batch_idx*16+16)])
        batch = batch.to(device)
        img_embeds.append(model.encode_image(batch).cpu().numpy())
        print('Successfully processed batch', batch_idx)
        
    img_embeds = np.concatenate(img_embeds)

    np.save('embeddings_sample.npy', img_embeds)

print('Success')
