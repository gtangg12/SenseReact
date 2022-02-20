import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

class TTS:
    def __init__(self):
        self.fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en", name="fastspeech2")
        self.mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en", name="mb_melgan")
        self.processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
    
    def get_sequence(self, input_text):
        input_ids = self.processor.text_to_sequence(input_text)
        return input_ids

    def synthesize(self, input_text):
        input_ids = self.get_sequence(input_text)

        _, mel_outputs, _, _, _ = self.fastspeech2.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )

        audio = self.mb_melgan(mel_outputs)[0, :, 0]

        return audio.numpy()

if __name__ == "__main__":
    from tts_utils import save_audio

    model = TTS()
    audio = model.synthesize("I want to go to the park tonight.")
    save_audio('./samples/audio.wav', audio)