import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

"""
Usage:
    response = models[name].create(engine='text-davinci-001',
                                   prompt=prompt,
                                   ...
                                   [other params]
                                   ...)
"""

models = {
    'text_completion': openai.Completion,
}
