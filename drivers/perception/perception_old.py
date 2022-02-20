"""
    Driver for translating video into captions

    Server side driver workhorse
"""

import sys
import re
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch import functional as F
from transformers import GPT2Tokenizer
from PIL import Image
sys.path.append('../../')
from kernel_util import *
from server import SERVER_DOCK
from drivers.perception.clip import clip
from drivers.perception.MappingNet.model import ClipCaptionPrefix
from drivers.perception.MappingNet.search import generate_beam


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


READ_EVERY = 5
CAPTION_AFTER_FRAMES = 10
MAX_CHUNK_FRAMES = 300
DRIFT_SIMILARITY_THRESHOLD = 0.5
NEW_CHUNK_SIMILARITY_THRESHOLD = 0.5
UNPROMPT_PREFIX_LENGTH = 40


captions = ['The man is having a picnic in the park with his dog.']
clip_model, clip_preprocess = None, None
clip_model_unprompt, clip_preprocess_unprompt = None, None
tokenizer_unprompt = None
mapping_net_unprompt = None


def load_clip():
    global clip_model, clip_preprocess
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)


def load_unprompt():
    global tokenizer_unprompt, mapping_net_unprompt, clip_model_unprompt, clip_preprocess_unprompt
    tokenizer_unprompt = GPT2Tokenizer.from_pretrained("gpt2")
    clip_model_unprompt, clip_preprocess_unprompt = clip.load("RN50x4", device=device, jit=False)

    model_path = './MappingNet/pretrained_models/model_wieghts.pt'
    mapping_net_unprompt = ClipCaptionPrefix(UNPROMPT_PREFIX_LENGTH, clip_length=40, prefix_size=640,
                                    num_layers=8, mapping_type='transformer')
    mapping_net_unprompt.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    mapping_net_unprompt.eval().to(device)


def embed_texts(texts):
    return clip_model.encode_text(texts)


def embed_frame(frame):
    tokens = clip_preprocess(frame)
    embeddings = clip_model.encode_image(tokens)
    return embeddings


def embed_compare_func(clip_embedding):
    cosine_sim = nn.CosineSimilarity(dim=0)

    def embed_compare(embedding):
        return cosine_sim(clip_embedding, embedding)
    return embed_compare


def next_caption(clip_embedding):
    """ """
    assert len(captions), "Cannot call next caption until captions seeded"

    N_PREV_CAPTIONS = 10
    events = ' '.join(captions[-N_PREV_CAPTIONS:])
    prompt = f'{events} List 20 events that will likely happen next.'
    '''
    response = models['text_completion'].create(engine='text-davinci-001',
                                                prompt=prompt,
                                                temperature=0.3,
                                                max_tokens=512)
    text = get_response_text(response)
    '''
    text = """
    1. The man will finish eating and put away the picnic supplies.
    2. The dog will lay down and rest.
    3. The man will take a nap.
    4. The dog will wake the man up.
    5. The man will get up and take a walk with the dog.
    6. The man will see other people in the park and stop to chat.
    7. The dog will bark at other dogs.
    8. The man will buy the dog a treat.
    9. The man will play fetch with the dog.
    10. The dog will get tired and want to go home.
    11. The man will pack up the picnic supplies and put them away.
    12. The man will walk the dog back to his house.
    13. The man will give the dog a bath.
    14. The man will feed the dog.
    15. The man will put the dog in his bed.
    16. The man will watch TV.
    17. The dog will lay down at the man's feet.
    18. The man will pet the dog.
    19. The dog will fall asleep.
    20. The man will turn off the light and go to bed. """
    candidates = [x.strip('\n ') for x in re.split(r'[0-9]+. ', text)]
    candidates[0] = captions[-1]
    embeddings = embed_texts(candidates)

    scores = list(map(embed_compare_func(clip_embedding), embeddings))
    scores = torch.stack(scores)
    topidx = torch.argmax(scores).item()
    return candidates[topidx], scores[topidx]


def base_caption(frame):
    image = Image.fromarray(frame)
    image = clip_preprocess_unprompt(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model_unprompt.encode_image(image).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = mapping_net_unprompt.clip_project(prefix).reshape(1, UNPROMPT_PREFIX_LENGTH, -1)
    generated_text_prefix = generate_beam(mapping_net_unprompt, tokenizer_unprompt, embed=prefix_embed)[0]

    return generated_text_prefix


def clean_caption():
    pass


def main():
    load_clip()

    inp_dir = f'{SERVER_DOCK}/perception_inp'
    out_dir = f'{SERVER_DOCK}/perception_out'

    last_segment_length = 0
    last_segment_caption_embedding = None
    sampling_counter = 0
    begin_new_chunk = False
    new_chunk_length = 0
    new_chunk_tokens = []

    while True:
        filename = sorted(os.listdir(inp_dir))[0]
        frame = torch.load(filename)

        """
        Assume we're not currently beginning a new chunk.
        We read a frame every READ_EVERY frames, and compute its embedding.
        If its similarity to the last segment's caption is too low, or if
        the last segment has lasted at least MAX_CHUNK_FRAMES, we start a new chunk.
        """

        sampling_counter += 1
        if sampling_counter % READ_EVERY == 0:
            sampling_counter = 0
            embedding = embed_frame(frame)

            """
            Detect whether we should start a new chunk. There're 3 cases:
                1. This is the first frame
                2. The last segment has been long enough
                3. The last segment's caption is too dissimilar to the current frame
            """
            if last_segment_caption_embedding is None:
                begin_new_chunk = True
            elif last_segment_length > MAX_CHUNK_FRAMES:
                begin_new_chunk = True
            else:
                similarity = F.cosing_similarity(last_segment_caption_embedding, embedding)
                if similarity < NEW_CHUNK_SIMILARITY_THRESHOLD:
                    begin_new_chunk = False
                    last_segment_length += READ_EVERY
                else:
                    begin_new_chunk = True

        if begin_new_chunk:
            """
            We're currently beginning a new chunk.
            We read every frame for the next CAPTION_AFTER_FRAMES frames,
            then using the average embedding of the frames, we compute
            a caption for the chunk.
            """
            new_chunk_length += 1
            new_chunk_tokens.append(embed_frame(frame))
            if new_chunk_length >= CAPTION_AFTER_FRAMES:
                batch = torch.stack(new_chunk_tokens).to(device)
                embeds = clip_model.encode_image(batch)
                embeds = torch.mean(embeds, dim=0)

                if last_segment_caption_embedding is not None:
                    # Get the most similar caption using Bayesian factorization
                    suggested_caption, similarity = next_caption(embeds)

                    if similarity > DRIFT_SIMILARITY_THRESHOLD:
                        caption = suggested_caption
                    else:
                        caption = base_caption(frame)

                else:
                    caption = base_caption(frame)

                begin_new_chunk = False
                new_chunk_length = 0
                new_chunk_tokens = []
                sampling_counter = 0
                last_segment_length = CAPTION_AFTER_FRAMES
                last_segment_caption_embedding = clip_model.encode_text(caption)
                captions.append(caption)

                filename = f'{time.time()}.txt'
                with open(f'{out_dir}/{filename}', 'w') as fout:
                    fout.write(caption)


if __name__ == '__main__':
    main()
