import csv
import datetime
import enum
import os
import re
import time
from collections import deque
from dataclasses import dataclass

import click
import gdown
import librosa
import numpy as np
import onnxruntime
# import pyaudio
import scipy
from scipy.io.wavfile import write, read
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
# from ibm_watson import SpeechToTextV1
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


MODEL_PATH_ROOT_ = "models"

def get_model_file_from_gdrive(name, url):
    filepath = os.path.join(MODEL_PATH_ROOT_, name)
    if not os.path.exists(filepath):
        os.makedirs(MODEL_PATH_ROOT_, exist_ok=True)
        gdown.download(url, filepath, quiet=False)

    return filepath

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                    original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return  float2pcm(sig, dtype='int16').tobytes()


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

from moviepy.editor import *

from espnet2.bin.asr_align import CTCSegmentation

# import spacy
# import ginza

def is_in_colaboratory():
    import os
    if 'COLAB_GPU' in os.environ:
        return True
    else:
        return False

class HumanVoiceDetector:
    def __init__(self, recognition_mode):
        self.class_names = ['Speech', 'Child speech, kid speaking', 'Conversation', 'Narration, monologue', 'Babbling', 'Speech synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell', 'Children shouting', 'Screaming', 'Whispering', 'Laughter', 'Baby laughter', 'Giggle', 'Snicker', 'Belly laugh', 'Chuckle, chortle', 'Crying, sobbing', 'Baby cry, infant cry', 'Whimper', 'Wail, moan', 'Sigh', 'Singing', 'Choir', 'Yodeling', 'Chant', 'Mantra', 'Child singing', 'Synthetic singing', 'Rapping', 'Humming', 'Groan', 'Grunt', 'Whistling', 'Breathing', 'Wheeze', 'Snoring', 'Gasp', 'Pant', 'Snort', 'Cough', 'Throat clearing', 'Sneeze', 'Sniff', 'Run', 'Shuffle', 'Walk, footsteps', 'Chewing, mastication', 'Biting', 'Gargling', 'Stomach rumble', 'Burping, eructation', 'Hiccup', 'Fart', 'Hands', 'Finger snapping', 'Clapping', 'Heart sounds, heartbeat', 'Heart murmur', 'Cheering', 'Applause', 'Chatter', 'Crowd', 'Hubbub, speech noise, speech babble', 'Children playing', 'Animal', 'Domestic animals, pets', 'Dog', 'Bark', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper (dog)', 'Cat', 'Purr', 'Meow', 'Hiss', 'Caterwaul', 'Livestock, farm animals, working animals', 'Horse', 'Clip-clop', 'Neigh, whinny', 'Cattle, bovinae', 'Moo', 'Cowbell', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl', 'Chicken, rooster', 'Cluck', 'Crowing, cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Goose', 'Honk', 'Wild animals', 'Roaring cats (lions, tigers)', 'Roar', 'Bird', 'Bird vocalization, bird call, bird song', 'Chirp, tweet', 'Squawk', 'Pigeon, dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot', 'Bird flight, flapping wings', 'Canidae, dogs, wolves', 'Rodents, rats, mice', 'Mouse', 'Patter', 'Insect', 'Cricket', 'Mosquito', 'Fly, housefly', 'Buzz', 'Bee, wasp, etc.', 'Frog', 'Croak', 'Snake', 'Rattle', 'Whale vocalization', 'Music', 'Musical instrument', 'Plucked string instrument', 'Guitar', 'Electric guitar', 'Bass guitar', 'Acoustic guitar', 'Steel guitar, slide guitar', 'Tapping (guitar technique)', 'Strum', 'Banjo', 'Sitar', 'Mandolin', 'Zither', 'Ukulele', 'Keyboard (musical)', 'Piano', 'Electric piano', 'Organ', 'Electronic organ', 'Hammond organ', 'Synthesizer', 'Sampler', 'Harpsichord', 'Percussion', 'Drum kit', 'Drum machine', 'Drum', 'Snare drum', 'Rimshot', 'Drum roll', 'Bass drum', 'Timpani', 'Tabla', 'Cymbal', 'Hi-hat', 'Wood block', 'Tambourine', 'Rattle (instrument)', 'Maraca', 'Gong', 'Tubular bells', 'Mallet percussion', 'Marimba, xylophone', 'Glockenspiel', 'Vibraphone', 'Steelpan', 'Orchestra', 'Brass instrument', 'French horn', 'Trumpet', 'Trombone', 'Bowed string instrument', 'String section', 'Violin, fiddle', 'Pizzicato', 'Cello', 'Double bass', 'Wind instrument, woodwind instrument', 'Flute', 'Saxophone', 'Clarinet', 'Harp', 'Bell', 'Church bell', 'Jingle bell', 'Bicycle bell', 'Tuning fork', 'Chime', 'Wind chime', 'Change ringing (campanology)', 'Harmonica', 'Accordion', 'Bagpipes', 'Didgeridoo', 'Shofar', 'Theremin', 'Singing bowl', 'Scratching (performance technique)', 'Pop music', 'Hip hop music', 'Beatboxing', 'Rock music', 'Heavy metal', 'Punk rock', 'Grunge', 'Progressive rock', 'Rock and roll', 'Psychedelic rock', 'Rhythm and blues', 'Soul music', 'Reggae', 'Country', 'Swing music', 'Bluegrass', 'Funk', 'Folk music', 'Middle Eastern music', 'Jazz', 'Disco', 'Classical music', 'Opera', 'Electronic music', 'House music', 'Techno', 'Dubstep', 'Drum and bass', 'Electronica', 'Electronic dance music', 'Ambient music', 'Trance music', 'Music of Latin America', 'Salsa music', 'Flamenco', 'Blues', 'Music for children', 'New-age music', 'Vocal music', 'A capella', 'Music of Africa', 'Afrobeat', 'Christian music', 'Gospel music', 'Music of Asia', 'Carnatic music', 'Music of Bollywood', 'Ska', 'Traditional music', 'Independent music', 'Song', 'Background music', 'Theme music', 'Jingle (music)', 'Soundtrack music', 'Lullaby', 'Video game music', 'Christmas music', 'Dance music', 'Wedding music', 'Happy music', 'Sad music', 'Tender music', 'Exciting music', 'Angry music', 'Scary music', 'Wind', 'Rustling leaves', 'Wind noise (microphone)', 'Thunderstorm', 'Thunder', 'Water', 'Rain', 'Raindrop', 'Rain on surface', 'Stream', 'Waterfall', 'Ocean', 'Waves, surf', 'Steam', 'Gurgling', 'Fire', 'Crackle', 'Vehicle', 'Boat, Water vehicle', 'Sailboat, sailing ship', 'Rowboat, canoe, kayak', 'Motorboat, speedboat', 'Ship', 'Motor vehicle (road)', 'Car', 'Vehicle horn, car horn, honking', 'Toot', 'Car alarm', 'Power windows, electric windows', 'Skidding', 'Tire squeal', 'Car passing by', 'Race car, auto racing', 'Truck', 'Air brake', 'Air horn, truck horn', 'Reversing beeps', 'Ice cream truck, ice cream van', 'Bus', 'Emergency vehicle', 'Police car (siren)', 'Ambulance (siren)', 'Fire engine, fire truck (siren)', 'Motorcycle', 'Traffic noise, roadway noise', 'Rail transport', 'Train', 'Train whistle', 'Train horn', 'Railroad car, train wagon', 'Train wheels squealing', 'Subway, metro, underground', 'Aircraft', 'Aircraft engine', 'Jet engine', 'Propeller, airscrew', 'Helicopter', 'Fixed-wing aircraft, airplane', 'Bicycle', 'Skateboard', 'Engine', 'Light engine (high frequency)', "Dental drill, dentist's drill", 'Lawn mower', 'Chainsaw', 'Medium engine (mid frequency)', 'Heavy engine (low frequency)', 'Engine knocking', 'Engine starting', 'Idling', 'Accelerating, revving, vroom', 'Door', 'Doorbell', 'Ding-dong', 'Sliding door', 'Slam', 'Knock', 'Tap', 'Squeak', 'Cupboard open or close', 'Drawer open or close', 'Dishes, pots, and pans', 'Cutlery, silverware', 'Chopping (food)', 'Frying (food)', 'Microwave oven', 'Blender', 'Water tap, faucet', 'Sink (filling or washing)', 'Bathtub (filling or washing)', 'Hair dryer', 'Toilet flush', 'Toothbrush', 'Electric toothbrush', 'Vacuum cleaner', 'Zipper (clothing)', 'Keys jangling', 'Coin (dropping)', 'Scissors', 'Electric shaver, electric razor', 'Shuffling cards', 'Typing', 'Typewriter', 'Computer keyboard', 'Writing', 'Alarm', 'Telephone', 'Telephone bell ringing', 'Ringtone', 'Telephone dialing, DTMF', 'Dial tone', 'Busy signal', 'Alarm clock', 'Siren', 'Civil defense siren', 'Buzzer', 'Smoke detector, smoke alarm', 'Fire alarm', 'Foghorn', 'Whistle', 'Steam whistle', 'Mechanisms', 'Ratchet, pawl', 'Clock', 'Tick', 'Tick-tock', 'Gears', 'Pulleys', 'Sewing machine', 'Mechanical fan', 'Air conditioning', 'Cash register', 'Printer', 'Camera', 'Single-lens reflex camera', 'Tools', 'Hammer', 'Jackhammer', 'Sawing', 'Filing (rasp)', 'Sanding', 'Power tool', 'Drill', 'Explosion', 'Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Artillery fire', 'Cap gun', 'Fireworks', 'Firecracker', 'Burst, pop', 'Eruption', 'Boom', 'Wood', 'Chop', 'Splinter', 'Crack', 'Glass', 'Chink, clink', 'Shatter', 'Liquid', 'Splash, splatter', 'Slosh', 'Squish', 'Drip', 'Pour', 'Trickle, dribble', 'Gush', 'Fill (with liquid)', 'Spray', 'Pump (liquid)', 'Stir', 'Boiling', 'Sonar', 'Arrow', 'Whoosh, swoosh, swish', 'Thump, thud', 'Thunk', 'Electronic tuner', 'Effects unit', 'Chorus effect', 'Basketball bounce', 'Bang', 'Slap, smack', 'Whack, thwack', 'Smash, crash', 'Breaking', 'Bouncing', 'Whip', 'Flap', 'Scratch', 'Scrape', 'Rub', 'Roll', 'Crushing', 'Crumpling, crinkling', 'Tearing', 'Beep, bleep', 'Ping', 'Ding', 'Clang', 'Squeal', 'Creak', 'Rustle', 'Whir', 'Clatter', 'Sizzle', 'Clicking', 'Clickety-clack', 'Rumble', 'Plop', 'Jingle, tinkle', 'Hum', 'Zing', 'Boing', 'Crunch', 'Silence', 'Sine wave', 'Harmonic', 'Chirp tone', 'Sound effect', 'Pulse', 'Inside, small room', 'Inside, large room or hall', 'Inside, public space', 'Outside, urban or manmade', 'Outside, rural or natural', 'Reverberation', 'Echo', 'Noise', 'Environmental noise', 'Static', 'Mains hum', 'Distortion', 'Sidetone', 'Cacophony', 'White noise', 'Pink noise', 'Throbbing', 'Vibration', 'Television', 'Radio', 'Field recording']

        if recognition_mode == "espnet":
            d = ModelDownloader()
            aaa = d.download_and_unpack("kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave")
            # aaa = d.download_and_unpack("espnet/Shinji_Watanabe_laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave")
            print(aaa)
            self.speech2text = Speech2Text(
                    **aaa,
                    device="cuda"
            )
        # self.aaa = aaa

        if is_in_colaboratory():
            self.config = yaml.load(open(os.path.expanduser("/content/zundavox.yaml")), Loader=Loader)
        else:
            self.config = yaml.load(open(os.path.expanduser("~/.zundavox.yaml")), Loader=Loader)

        # authenticator = IAMAuthenticator(config['watson']['api_key'])
        # speech_to_text = SpeechToTextV1(
        #     authenticator=authenticator,
        # )
        
        # self.nlp = spacy.load('ja_ginza_electra')

        # speech_to_text.set_service_url(config['watson']['url'])
        # self.speech_to_text = speech_to_text
        # self.sound_recognition = "espnet"
        self.sound_recognition = recognition_mode

    def write_speech_information_to_file(self, resampled_sound, sample_rate, f):
            # import IPython; IPython.embed()
        # f.write(str(n_wait * FRAME_LEN) + "\n")
        if self.sound_recognition == "espnet":
            length_s = len(resampled_sound) / sample_rate

            sentence = str(self.speech2text(resampled_sound)[0][0])
            # segments, uttrs = self.create_segments(sentence, resampled_sound)
            f.write(sentence + ",{\"length\": %s}\n" % length_s)

            # if len(uttrs) < 10:
            #     f.write(sentence + "\n")
            # else:
            # print(segments)

        elif self.sound_recognition == "ibm":
            write('keyword.wav', original_sample_rate, sound)
            speech_recognition_results = self.speech_to_text.recognize(
                        audio=open('keyword.wav', 'rb'), content_type='audio/wav', model="ja-JP_BroadbandModel").get_result()
            sentence = ""
            for result in speech_recognition_results["results"]:
                best_one = result["alternatives"][0]["transcript"]
                sentence += re.sub(" ", "", best_one)
            f.write(sentence + "\n")
        
        elif self.sound_recognition == "azure":
            import azure.cognitiveservices.speech as speechsdk
            
            length_s = len(resampled_sound) / sample_rate
            write('keyword.wav', sample_rate, (resampled_sound * np.iinfo(np.int16).max).astype(np.int16))
            speech_config = speechsdk.SpeechConfig(subscription=self.config['azure']['subscription'], speech_recognition_language="ja-JP", region="japaneast")
            audio_input = speechsdk.AudioConfig(filename="keyword.wav")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
            result = speech_recognizer.recognize_once()

            f.write(result.text + ",{\"length\": %s}\n" % length_s)


    def create_segments(self, sentence, sound):
        uttrs = []
        readings = []
        types = []
        doc = self.nlp(sentence)
        for sent in doc.sents:
            for token in sent:
                if token.pos_ != 'PUNCT' and token.pos_ != 'SYM' and token.pos_ != "X":
                    if token.orth_.strip() !=  "":
                        uttrs.append(token.orth_)
                        readings.append(ginza.reading_form(token))
                        types.append(token.pos_)


        segments = None
        try:
            segments = self.aligner(sound, uttrs)
        except IndexError:
            import traceback
            traceback.print_exc()

        return segments, uttrs

    def infer_sound_type(self, this_frame_data, original_sample_rate):
        input_name = self.session.get_inputs()[0].name  
        input_wave = this_frame_data.astype(np.float32)
        sample_rate, input_wave = ensure_sample_rate(original_sample_rate, input_wave)

        outputs = self.session.run([], {input_name: input_wave})[0][0]
        class_name = self.class_names[np.argmax(outputs)]

        return class_name, outputs

    def wait_for_human_voice(self, detect_class, video_file, output_text_path):
        # p = pyaudio.PyAudio()
        # stream = p.open(format=pyaudio.paInt16,
        #                 channels=1,
        #                 rate=24000,
        #                 input=True,
        #                 frames_per_buffer=frame_len)

        # sound = AudioFileClip("RPReplay_Final1639753358.mp4")
        # sound.write_audiofile("tmp_wav.wav", 24000, 2, 2000, "pcm_s16le")

        original_sample_rate = 24000

        import subprocess
        import shlex
        subprocess.check_output(shlex.split(f"ffmpeg -y -i {video_file} tmp_wav.wav"))
        original_sample_rate, wav = read("tmp_wav.wav", "rb")
        # original_sample_rate, wav = read("tmp_wav_DeepFilterNet.wav", "rb")
        original_wav = wav[:, 1] / np.iinfo(np.int16).max

        # self.aligner = CTCSegmentation( **self.aaa , fs= 16000)

        # self.aligner.set_config( gratis_blank=True, kaldi_style_text=False )

        PER_1_BUFFER_LEN = 0.96 * 0.1
        frame_len = int(original_sample_rate * PER_1_BUFFER_LEN)

        # import IPython; IPython.embed()

        buffers = deque()
        left_list = []

        providers = ['CPUExecutionProvider']

        self.session = onnxruntime.InferenceSession(
            get_model_file_from_gdrive("yamnet.onnx", "https://drive.google.com/uc?id=1u7V15wRp3_gcUdXPzm9WtJy51ENpCqEC"), 
            providers=providers)

        found_speech = False
        n_different = 0
        n_elapsed = 0

        # whole_segments = self.aligner(self.speech2text(wav)[0][0], wav)
        # print(whole_segments)

        with open(output_text_path, "w+") as f:
            f.write(f"[video2 {video_file}]" + "\n")
            n_wait = 0
            for i in range(0, len(original_wav), frame_len):
                frame_data = original_wav[i:i + frame_len]
            # while True:
            #     data = stream.read(frame_len, exception_on_overflow=False)
            #     frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

                # import IPython; IPython.embed()

                buffers.append(frame_data)
                if len(buffers) > 6:
                    if found_speech:
                        left_list.append(buffers.popleft())
                    else:
                        buffers.popleft()

                    this_frame_data = np.concatenate(buffers)
    
                    class_name, outputs = self.infer_sound_type(this_frame_data, original_sample_rate)

                    print(class_name, outputs[np.argmax(outputs)])
                    if outputs[np.argmax(outputs)] < 0.2:
                        class_name = ""

                    if found_speech != True and class_name == detect_class:
                        found_speech = True
                        n_elapsed_previous = n_elapsed
                    else:
                        n_wait += 1

                    if found_speech:
                        if class_name != detect_class:
                            n_different += 1
                            
                        if n_different > 0:
                            sound = np.concatenate(left_list + list(buffers))

                            sample_rate, resampled_sound = ensure_sample_rate(original_sample_rate, sound)

                            f.write(f"[wait {str(n_elapsed_previous * PER_1_BUFFER_LEN)}]" + "\n")
                            self.write_speech_information_to_file(resampled_sound, sample_rate, f)

                            # import IPython; IPython.embed()
                            buffers.clear()
                            left_list.clear()
                            found_speech = False
                            n_different = 0
                            n_wait = 0
                            n_elapsed += 1
                            continue

                n_elapsed += 1
                            

        # stream.stop_stream()
        # stream.close()
        # p.terminate()

@click.command()
@click.option("--video-path")
@click.option("--recognition-mode", default="azure")
@click.option("--output-script-path", default="azure")
def main(video_path, output_script_path, recognition_mode):
    voice_detector = HumanVoiceDetector(recognition_mode)
    voice_detector.wait_for_human_voice("Speech", video_path, output_script_path)


if __name__ == "__main__":
    main()