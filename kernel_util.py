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
    #'chat': openai.Chat,
}


def get_response_text(response):
    return response['choices'][0]['text']


from drivers.camera.camera import CameraDriver
from drivers.console.console import ConsoleDriver
from drivers.perception.perception_client import PerceptionDriver

drivers = {
    'camera': CameraDriver,
    'console': ConsoleDriver,
    'perception': PerceptionDriver,
}
