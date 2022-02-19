"""
    Driver for translating raw video into captions
"""
import sys
import re
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from queue import Queue
from drivers.perception.clip import clip
sys.path.append('../../')
from kernel_util import *


class PerceptionDriver:
    buffer = Queue()

    def __init__(self):
        pass

    @classmethod
    def push(cls, caption):
        buffer.put(caption)

    @classmethod
    def next(cls):
        return buffer.get(block=True)

    @classmethod
    def reset(cls):
        buffer = Queue()


READ_EVERY = 5
CAPTION_AFTER_FRAMES = 10
MAX_CHUNK_FRAMES = 300
DRIFT_SIMILARITY_THRESHOLD = 0.5
NEW_CHUNK_SIMILARITY_THRESHOLD = 0.5


captions = ['The man is having a picnic in the park with his dog.']
clip_model, clip_preprocess = None, None


def load_clip():
    clip_model, clip_preprocess = clip.load("ViT-L/14", device='cuda')


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


def get_google_caption(embeds):
    pass


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

    video_cap = cv2.VideoCapture(0)
    cap_prop = lambda x : int(video_cap.get(x))

    width, height = \
        cap_prop(cv2.CAP_PROP_FRAME_WIDTH), cap_prop(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Camera dimensions: {}x{}".format(height, width))

    start = time.time()
    frames = []

    last_segment_length = 0
    last_segment_caption_embedding = None
    sampling_counter = 0
    begin_new_chunk = False
    new_chunk_length = 0
    new_chunk_tokens = []

    while True:
        success, frame = video_cap.read()
        if not success or len(frames) > args.num_frames - 1:
            break


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
                similarity = clip.cosine_similarity(last_segment_caption_embedding, embedding)
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
                batch = torch.stack(new_chunk_tokens).to('cuda')
                embeds = clip_model.encode_image(batch)
                embeds = torch.mean(embeds, dim=0)

                if last_segment_caption_embedding is not None:
                    # Get the most similar caption using Bayesian factorization
                    suggested_caption, similarity = next_caption(embeds)

                    if similarity > DRIFT_SIMILARITY_THRESHOLD:
                        caption = suggested_caption
                    else:
                        caption = get_google_caption(embeds)

                else:
                    caption = get_google_caption(embeds)

                begin_new_chunk = False
                new_chunk_length = 0
                new_chunk_tokens = []
                sampling_counter = 0
                last_segment_length = CAPTION_AFTER_FRAMES
                last_segment_caption_embedding = clip_model.encode_text(caption)
                PerceptionDriver.push(caption)
                captions.append(caption)


        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    print ("Recording time taken : {0} seconds".format(time.time() - start))


if __name__ == '__main__':
    main()
