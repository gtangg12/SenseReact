import torch
import zipfile
import torchaudio
from glob import glob


class Multi_STT:
    def __init__(self, lang='en'): # supports en (english), es (spanish), de (german)
        self.device = torch.device('cpu')
        self.model, self.decoder, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                              model='silero_stt',
                                                              language=lang,
                                                              device=self.device)
        self.read_batch, self.split_into_batches, self.read_audio, self.prepare_model_input = self.utils

    def transcribe(self, audio_file):
        audio = glob(audio_file)
        batches = self.split_into_batches(audio, batch_size=10)
        input = self.prepare_model_input(self.read_batch(batches[0]), device=self.device)
        output = self.model(input)

        return self.decoder(output[0].cpu())

if __name__ == '__main__':
    model_en = Multi_STT('en')
    print(model_en.transcribe('./samples/sample_en.wav'))