import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile

import fitz
import numpy as np
import scipy
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
from moviepy.video.io import ImageSequenceClip
from PIL import Image
from scipy.io import wavfile
from scipy.io.wavfile import write

from zundavox.agent_display import TsukuyomichanVisualizer


class SlideVideoBuilder:
    def __init__(self, d_path, background_video_path=None, slide_pdf_path=None):
        self.frame = 0
        self.frame_rate = 30
        self.pil_images = []
        self.wav = np.array([], dtype=np.int16)
        self.fs = 24000
        self.subtitles = []
        self.d_path = d_path
        self.image_files = []
        self.video_files = []
        self.previous_output_time = 0
        self.clip_list = None

        # 
        self.len_background_video_frames = 0
        if background_video_path is not None:
            self.background_clip = VideoFileClip(background_video_path) 
            self.len_background_video_frames = len(list(self.background_clip.iter_frames()))
            print(self.len_background_video_frames)
            # for frame in clip.iter_frames():
            #     self.background_video_frames.append(Image.fromarray(frame))

        # Load slides into a memory as a low image files.
        self.slides = []
        if slide_pdf_path is not None:
            doc = fitz.open(slide_pdf_path)
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), dpi=(200, 200))
                data = pix.getImageData("png")
                img = Image.open(io.BytesIO(data))
                self.slides.append(img)
        self.current_slide = None

    def start_slide(self, no):
        self.current_slide = self.slides[no]

    def stop_slide(self):
        self.current_slide = None

    def stop_video_clip(self):
        self.clip_list = None
        self.video_start_frame = 0

    def set_video_clip(self, video_clip):
        self.clip_list = video_clip.iter_frames()

        self.video_start_frame = self.frame

    def relative_time(self):
        return (self.frame - self.video_start_frame) / self.frame_rate

    def time(self):
        return self.frame / self.frame_rate

    def output(self, wav, fs):
        self.wav = np.concatenate((self.wav, wav))

    def sound_and_video_is_aligned(self):
        return int(self.fs * (self.time() - len(self.wav)/self.fs)) == 0

    def pad_with_empty(self):
        subtract_wavlen_videolen = int(self.fs * (self.time() - len(self.wav)/self.fs))
        if subtract_wavlen_videolen > 0:
            self.wav = np.concatenate((self.wav, np.array([0] * subtract_wavlen_videolen)))
            elapsed_time_s = (self.time() - self.previous_output_time)
            if elapsed_time_s > 5:
                self._save_current_pil_images()
                self.previous_output_time = self.time()
            # elapsed_time_s = (self.time() - self.previous_output_time)
            # if elapsed_time_s % self.frame_rate < 3 and elapsed_time_s > 30:
            #     self._save_current_pil_images()
            #     self.previous_output_time = self.time()
            return True
        elif subtract_wavlen_videolen == 0:
            elapsed_time_s = (self.time() - self.previous_output_time)
            if elapsed_time_s > 5:
                self._save_current_pil_images()
                self.previous_output_time = self.time()
            # elapsed_time_s = (self.time() - self.previous_output_time)
            # if elapsed_time_s % self.frame_rate < 3 and elapsed_time_s > 30:
            #     self._save_current_pil_images()
            #     self.previous_output_time = self.time()
            return  True
        else:
            return False

    def add_whole_video_frame(self, frame):
        """
        """
        # background = self.background_video_frames[self.frame % len(self.background_video_frames)].copy()
        background = Image.fromarray(self.background_clip.get_frame(self.frame % self.len_background_video_frames / self.frame_rate))
        background_resized = background.resize((1920, 1080))
        background_resized.paste(frame, (1920//2 - frame.width//2, 1080//2 - frame.height//2))

        self.pil_images.append(background_resized)
        self.frame += 1

    def add_video_frame(self, character_image_pil):
        """Add video frame given character image.

        """
        if self.len_background_video_frames > 0:
            if character_image_pil is None:
                if self.clip_list is not None:
                    try:
                        next_frame = next(self.clip_list)
                    except StopIteration:
                        self.clip_list = None
                        return
                    else:
                        next_pil = Image.fromarray(next_frame).resize((1920, 1080))
                        self.pil_images.append(next_pil)
            else:
                person = character_image_pil.resize((int(character_image_pil.width), int(character_image_pil.height)))
                # tsumugi
                # person_height = person.height // 2
                # zunda
                person_height = (person.height) * 2 // 3

                background = Image.fromarray(self.background_clip.get_frame(self.frame % self.len_background_video_frames / self.frame_rate))
                # background = self.background_video_frames[self.frame % len(self.background_video_frames)].copy()
                background_resized = background.resize((1920, 1080))
                if self.current_slide is not None:
                    background_resized.paste(self.current_slide.resize((1500, 800)), (50, 50))
                    # resize_ratio = 1500 / pil_image.height
                    person = character_image_pil.resize((int(character_image_pil.width), int(character_image_pil.height)))
                    background_resized.paste(person, (1920 - person.width + 100 + 100, 1080 - person_height), person)
                elif self.clip_list is not None:
                    try:
                        next_frame = next(self.clip_list)
                    except StopIteration:
                        self.clip_list = None
                        return
                    else:
                        next_pil = Image.fromarray(next_frame)
                        background_resized.paste(next_pil.resize((1500, 800)), (50, 50))
                        # resize_ratio = 1500 / pil_image.height
                        background_resized.paste(person, (1920 - person.width + 100 + 100, 1080 - person_height), person)
                else:
                    resize_ratio = 1500 / character_image_pil.height
                    background_resized.paste(
                        character_image_pil, 
                        (1920 // 2 - character_image_pil.width // 2, 1080 - person_height), 
                        character_image_pil)
                self.pil_images.append(background_resized)
        else:
            self.pil_images.append(character_image_pil)
        self.frame += 1

    def add_subtitle(self, start_time, end_time, text):
        self.subtitles.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": text
        })

    def _save_current_pil_images(self):
        # for i, pil_image in enumerate(self.pil_images):
        #     filepath = os.path.join(self.d_path, f"frame_{len(self.image_files)}.png")
        #     if scale_rate != 1.0:
        #         pil_image = pil_image.resize((int(pil_image.width * scale_rate), int(pil_image.height * scale_rate)))
    
        #     # pil_image.save(filepath)
        #     self.image_files.append(filepath)

        if len(self.pil_images) > 0:
            filepath = os.path.join(self.d_path, f"video_{len(self.video_files)}.mp4")
            clip = ImageSequenceClip.ImageSequenceClip([np.asarray(pil_image) for pil_image in self.pil_images], fps=self.frame_rate)
            clip.write_videofile(filepath)
            # clip.write_videofile(filepath, codec="mpeg4")
            self.video_files.append(filepath)
            self.pil_images.clear()

    def output_video(self, output_path, subtitle_path):
        # import IPython; IPython.embed()
        self.wav = self.wav.astype(np.int16)
        write(os.path.join(self.d_path, "tmp.wav"), self.fs, self.wav)
        self._save_current_pil_images()
        # audio = AudioArrayClip([self.wav], fps=self.fs)
        audio = AudioFileClip(os.path.join(self.d_path, "tmp.wav"))
        # new_audioclip = CompositeAudioClip([audio])
        # clip = ImageSequenceClip.ImageSequenceClip(self.image_files, fps=self.frame_rate)
        # clip.audio = audio
        # clip.write_videofile(output_path)

        try:
            clips = []
            clip_tmp_path = os.path.join(self.d_path, "tmp_concat.mp4")
            for video_file in self.video_files:
                clips.append(VideoFileClip(video_file))
            clip = concatenate_videoclips(clips)
            clip.write_videofile(clip_tmp_path)

            # Add audio.
            clip2 = VideoFileClip(clip_tmp_path)
            clip2.audio = audio
            clip2.write_videofile(output_path)
        except:
            import IPython; IPython.embed()

        # import IPython; IPython.embed()

        def second_to_string(second):
            hour = second // 3600
            remain = second % 3600
            minute = remain // 60
            remain = remain % 60
            second = remain // 1
            millisecond = remain % 1
            return "%02d:%02d:%02d,%03d" % (hour, minute, second, millisecond * 1000)

        if subtitle_path is not None:
            text = ""
            for i, subtitle in enumerate(self.subtitles):
                text += f"{i}\n"
                text += second_to_string(subtitle["start_time"]) + " --> " + second_to_string(subtitle["end_time"]) + "\n"
                text += subtitle["text"] + "\n\n"

            open(subtitle_path, "w").write(text)

            short_text = ""
            for i, subtitle in enumerate(self.subtitles):
                short_text += f"{i}\n"
                short_text += second_to_string(subtitle["start_time"]) + " --> " + second_to_string(subtitle["end_time"]) + "\n"
                for i in range(0, len(subtitle["text"]), 35):
                    short_text += subtitle["text"][i:i+35] + "\n"
                short_text += "\n"

            open("short_subtitle.srt", "w").write(short_text)

class PresentationVideoGenerator:
    """Generates presentation video."""
    def __init__(self, background_video, slide_pdf, character, config):
        self.temporary_directory = tempfile.mkdtemp()
        self.video_generator = SlideVideoBuilder(self.temporary_directory, background_video, slide_pdf)
        self.visualizer = TsukuyomichanVisualizer(
            None, 
            config.CHARACTER_GENERATION,
            feeling_estimator=None,
            clock=self.video_generator, wav_output=self.video_generator, # background_color=(0,255,0), 
            transparent_background=True, #  if background_video else False
            character=character,
        )
        self.show_character_visual = True
        self.config = config

    def generate_video(self, output_path, manuscript_file, scale_rate, subtitle_path=None, background_video=None, slide_pdf=None, character="zunda1"):
        manuscript = "".join(open(manuscript_file).readlines())

        manuscript = [s for s in manuscript.split("\n") if s.strip() != ""]
        
        for i in range(0, len(manuscript)):
            if re.match(r"^[0-9\.]+$", manuscript[i]):
                manuscript[i] = float(manuscript[i])

        with tempfile.TemporaryDirectory() as d_path:
            self.generate(manuscript, output_path, subtitle_path)

    def generate(self, manuscript, output_path, subtitle_path):
        visualizer = self.visualizer
        video_generator = self.video_generator
        for text in manuscript:
            if isinstance(text, str):
                if text == "[slide]":
                    video_generator.stop_slide()
                    video_generator.stop_video_clip()
                    continue
        
                if text == "[no_visual]":
                    self.show_character_visual = False
                    continue

                slide_match = re.match(r"^\[slide ([0-9]+)\]$", text)
                if slide_match:
                    video_generator.start_slide(int(slide_match.group(1)) - 1)
                    video_generator.stop_video_clip()
                    continue
                    
                sound_path = re.match(r"^\[sound (.+)\]$", text)
                if sound_path:
                    sample_rate, wav_data = wavfile.read(sound_path.group(1), 'rb')
                    concat_wav_data =  ((wav_data[:, 0] + wav_data[:, 1]) / 2).astype(np.int16)
                    sample_rate, resampled_concat_wav_data = ensure_sample_rate(sample_rate, wav_data[:, 0])
                    video_generator.output(resampled_concat_wav_data, None)
                    continue

                video_path = re.match(r"^\[video (.+)\]$", text)
                if video_path:
                    video_clip = VideoFileClip(video_path.group(1))
                    for frame in video_clip.iter_frames():
                        video_generator.add_whole_video_frame(Image.fromarray(frame)) 
                    if video_clip.audio is not None:
                        subprocess.check_output(shlex.split(f"ffmpeg -y -i {video_path.group(1)} /tmp/jiejie_wav.wav"))
                        sample_rate, wav_data = wavfile.read("/tmp/jiejie_wav.wav", 'rb')
                        sample_rate, resampled_concat_wav_data = ensure_sample_rate(sample_rate, wav_data[:, 0])
                        video_generator.output(resampled_concat_wav_data, None)
                    continue


                video_path = re.match(r"^\[video2 (.+)\]$", text)
                if video_path:
                    video_generator.stop_slide()
                    video_generator.set_video_clip(VideoFileClip(video_path.group(1)))
                    # import IPython; IPython.embed()
                    continue

                wait_until = re.match(r"^\[wait (.+)\]$", text)
                if wait_until:
                    next_clock = float(wait_until.group(1))
                    print(f"waiting {next_clock}")
                    i = 0
                    while video_generator.relative_time() < next_clock:
                        video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))
                        i+= 1
                    print(i)
                    # This is necessary 
                    # video_generator.pad_with_empty()
                    # Not sure if it's necessary. need refactor.
                    while not video_generator.pad_with_empty():
                        video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))
                    continue

                # Starting subtitle.
                subtitle_start_time = video_generator.time()

                text_config = text.split(",")
                text = text_config[0]
                option = {}
                # import IPython; IPython.embed()
                if len(text_config) > 1:
                    option = json.loads(text_config[1])

                if not video_generator.sound_and_video_is_aligned():
                    if int(video_generator.fs * (video_generator.time() - len(video_generator.wav)/video_generator.fs)) > 4:
                        import IPython; IPython.embed()

                video_generator.add_video_frame(visualizer.visualize(text, generate_visual=self.show_character_visual, speech_option=option))

                while visualizer.saying_something():
                    video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))

                subtitle_end_time = video_generator.time()
                video_generator.add_subtitle(subtitle_start_time, subtitle_end_time, text)
        
                # wait 0.5 after each text.
                # next_clock = video_generator.time() + 1.0
                # i = 0
                # while video_generator.time() < next_clock:
                #     video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))
                #     i+= 1
                # This is necessary to pad last phrase.
                while not video_generator.pad_with_empty():
                    video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))
            elif isinstance(text, float):
                next_clock = video_generator.time() + text
                i = 0
                while video_generator.time() < next_clock:
                    video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))
                    i+= 1
                print(i)
                # This is necessary 
                # video_generator.pad_with_empty()
                # Not sure if it's necessary. need refactor.
                while not video_generator.pad_with_empty():
                    video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))

        # This is necessary to pad last phrase.
        while not video_generator.pad_with_empty():
            video_generator.add_video_frame(visualizer.visualize(None, generate_visual=self.show_character_visual))


        video_generator.output_video(output_path, subtitle_path)
        shutil.rmtree(self.temporary_directory)


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=24000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform
