import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

class MultiTTS:
    datasets = {'en': 'ljspeech', 'ch': 'baker', 'ko': 'kss'}
    inference = {'en': False, 'ch': True, 'ko': False}

    def __init__(self, lang='en'): # support en (english), ch (chinese), and ko (korean)
        self.lang = lang
        self.dataset = MultiTTS.datasets[lang]

        self.fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-{}-{}".format(self.dataset, self.lang), name="fastspeech2")
        self.mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-{}-{}".format(self.dataset, self.lang), name="mb_melgan")
        self.processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-{}-{}".format(self.dataset, self.lang))
    
    def get_sequence(self, input_text):
        if MultiTTS.inference[self.lang]:
            input_ids = self.processor.text_to_sequence(input_text, inference=True)
        else:
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

    en_model = MultiTTS('en')
    ch_model = MultiTTS('ch')
    ko_model = MultiTTS('ko')

    en_audio = en_model.synthesize("I want to go to the park tonight.")
    ch_audio = ch_model.synthesize("我今晚想去公园。")
    ko_audio = ko_model.synthesize("오늘 밤에 공원에 가고 싶어요.")

    save_audio('./samples/en_audio.wav', en_audio)
    save_audio('./samples/ch_audio.wav', ch_audio)
    save_audio('./samples/ko_audio.wav', ko_audio)