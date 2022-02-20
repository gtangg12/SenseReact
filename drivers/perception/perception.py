"""
    Driver for translating video into captions
"""

import sys
import re
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer
from PIL import Image
from queue import Queue
sys.path.append('../../')
from kernel_util import *
from drivers.perception.clip import clip
from drivers.perception.MappingNet.model import ClipCaptionPrefix
from drivers.perception.MappingNet.search import generate_beam


class PerceptionDriver:
    buffer = Queue()

    @classmethod
    def push(cls, caption):
        cls.buffer.put(caption)
        print(caption)

    @classmethod
    def next(cls):
        return 'tmp'
        #return cls.buffer.get(block=True)

    @classmethod
    def reset(cls):
        cls.buffer = Queue()


READ_EVERY = 5
CAPTION_AFTER_FRAMES = 10
MAX_CHUNK_FRAMES = 1000
DRIFT_SIMILARITY_THRESHOLD = 0.18
NEW_CHUNK_SIMILARITY_THRESHOLD = 0.23
UNPROMPT_PREFIX_LENGTH = 40

clip_model_device='cuda:0'
clip_model_unprompt_device='cuda:1'
mapping_net_unprompt_device='cuda:2'


captions = []#['The man is having a picnic in the park with his dog.']
clip_model, clip_preprocess = None, None


clip_model_unprompt, clip_preprocess_unprompt = None, None
tokenizer_unprompt = None
mapping_net_unprompt = None

def load_clip():
    global clip_model, clip_preprocess
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=clip_model_device)

def load_unprompt():
    global tokenizer_unprompt, mapping_net_unprompt, clip_model_unprompt, clip_preprocess_unprompt
    tokenizer_unprompt = GPT2Tokenizer.from_pretrained("gpt2")
    clip_model_unprompt, clip_preprocess_unprompt = clip.load("RN50x4", device=clip_model_unprompt_device, jit=False)

    model_path = './MappingNet/pretrained_models/model_wieghts.pt'
    mapping_net_unprompt = ClipCaptionPrefix(UNPROMPT_PREFIX_LENGTH, clip_length=40, prefix_size=640,
                                    num_layers=8, mapping_type='transformer')
    mapping_net_unprompt.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    mapping_net_unprompt.eval().to(mapping_net_unprompt_device)


def embed_texts(texts):
    texts = clip.tokenize(texts).to(clip_model_device)
    return clip_model.encode_text(texts).float()


def embed_frame(frame):
    image = Image.fromarray(frame)
    tokens = clip_preprocess(image).unsqueeze(0).to(clip_model_device)
    embeddings = clip_model.encode_image(tokens).squeeze(0)
    return embeddings.float()


def embed_compare_func(clip_embedding):
    #cosine_sim = nn.CosineSimilarity(dim=0)

    def embed_compare(embedding):
        return F.cosine_similarity(clip_embedding, embedding, dim=0)
        # return cosine_sim(clip_embedding, embedding)
    return embed_compare


def next_caption(clip_embedding):
    """ """
    assert len(captions), "Cannot call next caption until captions seeded"

    N_PREV_CAPTIONS = 10
    events = ' '.join(captions[-N_PREV_CAPTIONS:])
    prompt = f'{events} List 20 events that might happen next:'
    print(prompt)
    
    candidates = []
    for _ in range(2):
        response = models['text_completion'].create(engine='text-davinci-001',
                                                    prompt=prompt,
                                                    temperature=0.7,
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
        '''
        
        candidates.extend([x.strip('\n ') for x in re.split(r'[0-9]+. ', text)])
    candidates.append(captions[-1])
    print(candidates)
    embeddings = embed_texts(candidates)

    scores = list(map(embed_compare_func(clip_embedding), embeddings))
    print(scores)
    scores = torch.stack(scores)
    topidx = torch.argmax(scores).item()
    # print('fadjoicajsmjodcjamsockjnadscokj')
    # print(candidates[topidx])
    # print(candidates)
    return candidates[topidx], scores[topidx]


def get_google_caption(frame):
    return "A man is walking towards a wall."
    """
    image = Image.fromarray(frame)
    image = clip_preprocess_unprompt(image).unsqueeze(0).to(clip_model_unprompt_device)
    with torch.no_grad():
        prefix = clip_model_unprompt.encode_image(image).to(mapping_net_unprompt_device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = mapping_net_unprompt.clip_project(prefix).reshape(1, UNPROMPT_PREFIX_LENGTH, -1)
    generated_text_prefix = generate_beam(mapping_net_unprompt, tokenizer_unprompt, embed=prefix_embed)[0]

    return generated_text_prefix
    """



def clean_caption():
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Recording settings.')

    parser.add_argument('-o', '--output_path', default='output.mp4', type=str,
        help='')
    parser.add_argument('-n', '--num_frames', default=2000, type=int,
        help='')
    parser.add_argument('--fps', default=24, type=int,
        help='')

    return parser.parse_args()

def main():
    args = parse_args()
    load_clip()

    video_cap = cv2.VideoCapture('/nobackup/users/wzhao6/treehacks2022/sample_video.mp4')
    cap_prop = lambda x : int(video_cap.get(x))

    width, height = \
        cap_prop(cv2.CAP_PROP_FRAME_WIDTH), cap_prop(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Camera dimensions: {}x{}".format(height, width))

    frames = []

    last_segment_length = 0
    last_segment_caption_embedding = None
    sampling_counter = 0
    begin_new_chunk = False
    new_chunk_length = 0
    new_chunk_tokens = []

    frame_counter = 0

    load_clip()
    load_unprompt()    
    start = time.time()
    print('Begin Reading From Camera')
    while True:
        frame_counter += 1
        success, frame = video_cap.read()
        if not success or len(frames) > args.num_frames - 1:
            break


        """
        Assume we're not currently beginning a new chunk.
        We read a frame every READ_EVERY frames, and compute its embedding.
        If its similarity to the last segment's caption is too low, or if
        the last segment has lasted at least MAX_CHUNK_FRAMES, we start a new chunk.
        """

        if not begin_new_chunk:
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
                    #print(last_segment_caption_embedding.shape, embedding.shape)
                    similarity = F.cosine_similarity(last_segment_caption_embedding, embedding, dim=0)
                    print(similarity)
                    if similarity > NEW_CHUNK_SIMILARITY_THRESHOLD:
                        begin_new_chunk = False
                        last_segment_length += READ_EVERY

                    else:
                        print(frame_counter)
                        #print('sfasoijoasok')
                        #print(last_segment_caption_embedding, embedding)
                        #print((last_segment_caption_embedding * embedding).sum())
                        #print(last_segment_caption_embedding.norm())
                        #print(embedding.norm())
                        begin_new_chunk = True

        if begin_new_chunk:
            """
            We're currently beginning a new chunk.
            We read every frame for the next CAPTION_AFTER_FRAMES frames,
            then using the average embedding of the frames, we compute
            a caption for the chunk.
            """
            new_chunk_length += 1
            image = Image.fromarray(frame)
            new_chunk_tokens.append(clip_preprocess(image))
            if new_chunk_length >= CAPTION_AFTER_FRAMES:
                batch = torch.stack(new_chunk_tokens).to(clip_model_device)
                embeds = clip_model.encode_image(batch).float()
                embeds = torch.mean(embeds, dim=0)

                if last_segment_caption_embedding is not None:
                    # Get the most similar caption using Bayesian factorization

                    """
                    # short-circuit: if the previous caption turned out to be not so bad, we keep it
                    if F.cosine_similarity(last_segment_caption_embedding, embeds, dim=0) > DRIFT_SIMILARITY_THRESHOLD:
                        # except in the case that the last segment is too long, in which case we have to have a new caption
                        if not last_segment_length > MAX_CHUNK_FRAMES:
                            begin_new_chunk = False
                            new_chunk_length = 0
                            new_chunk_tokens = []
                            continue
                    """

                    suggested_caption, similarity = next_caption(embeds)
                    print("suggested caption is {} with similarity {}".format(suggested_caption, similarity))

                    if similarity > DRIFT_SIMILARITY_THRESHOLD:
                        caption = suggested_caption
                        print("Using suggested caption: {}".format(caption))
                    else:
                        caption = get_google_caption(frame)
                        print("Using google caption: {}".format(caption))

                else:
                    caption = get_google_caption(frame)
                    print("Using google caption: {}".format(caption))

                begin_new_chunk = False
                new_chunk_length = 0
                new_chunk_tokens = []
                sampling_counter = 0
                last_segment_length = CAPTION_AFTER_FRAMES
                last_segment_caption_embedding = clip_model.encode_text(clip.tokenize([caption]).to(clip_model_device)).squeeze(0).float()
                PerceptionDriver.push(caption)
                print('new caption in frame', frame_counter)
                captions.append(caption)


        #if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #    break

    print ("Recording time taken : {0} seconds".format(time.time() - start))


if __name__ == '__main__':
    main()
