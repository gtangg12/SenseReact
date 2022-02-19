"""
Translate input video into captions
"""
import sys
sys.path.append('../')

import re
from queue import Queue
from kernel_util import models


class PerceptionDriver:
    buffer = Queue()

    def next():
        return buffer.get(block=True)

    def reset():
        buffer = Queue()


captions = ['The man is having a picnic in the park with his dog.']


def embed_text(text):
    pass

def next_caption(clip_embedding):
    """ """
    get_text : lambda response : response['choices'][0]['text']

    NUM_PREV = 10
    events = ' '.join(captions[-NUM_PREV:])
    prompt = f'{events} List 20 events that will likely happen next.'

    response = models['text_completion'].create(engine='text-davinci-001',
                                                prompt=prompt,
                                                temperature=0.3,
                                                max_tokens=512)
    text = get_text(response)
    candidates = re.split(r'\n[0-9]+. ', text)
    candidates[0] = candidates[-1]
    embeddings = [embed_text(x) for x in candidates]



def clean_caption():
    pass


NUM_PREV = 10
events = ' '.join(captions[-NUM_PREV:])
prompt = f'{events}\n\nList 20 events that will likely happen next.'


'''
response = models['text_completion'].create(engine='text-davinci-001',
                                            prompt=prompt,
                                            temperature=0.3,
                                            max_tokens=512)


text = response['choices'][0]['text']
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
possible_next_captions = re.split(r'\n[0-9]+. ', text)
possible_next_captions[0] = possible_next_captions[-1]
print(possible_next_captions)
