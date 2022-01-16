import enum
import time

import cv2
import emoji
import ginza
import numpy
import numpy as np
import PIL
import simpleaudio as sa
import spacy
import torch

from espnet2.bin.asr_align import CTCSegmentation
from espnet_model_zoo.downloader import ModelDownloader

import tha2.poser.modes.mode_20
from tha2.util import (convert_output_image_from_torch_to_numpy,
                       extract_pytorch_image_from_PIL_image)
from utils.file import get_model_file, get_model_file_from_gdrive

# play_obj = sa.play_buffer(wav, 1, 2, fs)

class BlinkController:
    def __init__(self, clock):
        self.clock = clock
        self.state_start_time = clock.time()
        self.state = "closing"

    def blink_rate(self):
        state_periods = {
            "closing": 0.1,
            "closed": 0.05,
            "opening": 0.1, 
            "wait": 2
        }
        next_states = {
            "closing": "closed",
            "closed": "opening",
            "opening": "wait",
            "wait": "closing"
        }

        duration_from_last = self.clock.time() - self.state_start_time

        # print(self.state)

        if duration_from_last > state_periods[self.state]:
            self.state_start_time = self.clock.time()
            self.state = next_states[self.state]

        duration_from_last = self.clock.time() - self.state_start_time

        if self.state == "closing":
            return duration_from_last / state_periods["closing"]
        elif self.state == "closed":
            return 1.0
        elif self.state == "opening":
            return 1.0 - duration_from_last/state_periods["opening"]
        elif self.state == "wait": 
            return 0.0

class BodyController:
    def __init__(self, clock):
        self.clock = clock
        self.state_start_time = clock.time()
        self.state = "closing"

    def control(self, pose, pose_parameters):
        state_periods = {
            "closing": 0.5,
            "closed": 0.5,
            "opening": 0.5, 
            "wait": 3
        }
        next_states = {
            "closing": "closed",
            "closed": "opening",
            "opening": "wait",
            "wait": "closing"
        }

        duration_from_last = self.clock.time() - self.state_start_time

        # print(self.state)

        if duration_from_last > state_periods[self.state]:
            self.state_start_time = self.clock.time()
            duration_from_last = 0
            self.state = next_states[self.state]

        AMPLITUDE = 0.25

        rate = None
        if self.state == "closing":
            rate = AMPLITUDE * duration_from_last / state_periods["closing"]
        elif self.state == "closed":
            rate = AMPLITUDE
        elif self.state == "opening":
            rate =  AMPLITUDE - AMPLITUDE * duration_from_last/state_periods["opening"]
        elif self.state == "wait": 
            rate = 0.0

        assert rate != None

        if rate > AMPLITUDE or rate < -AMPLITUDE:
            import IPython; IPython.embed()

        pose[0, pose_parameters.get_parameter_index("neck_z")] = rate
        # if rate > 0.0:
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_left")] = 1.0
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_right")] = 1.0
        #     pose[0, pose_parameters.get_parameter_index("eye_wink_left")] = 0.0
        #     pose[0, pose_parameters.get_parameter_index("eye_wink_right")] = 0.0
        # else:
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_left")] = 0
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_right")] = 0

mouth_shapes = ["mouth_aaa", "mouth_iii", "mouth_uuu", "mouth_eee", "mouth_ooo", "mouth_delta"]

mouth_map = [
    ["アカガサザタダナハバパマヤラワャ", "mouth_aaa"],
    ["イキギシジチジニヒビピミリ", "mouth_iii"],
    ["ウクグスズツズヌフブプムユルュ", "mouth_uuu"],
    ["エケゲセゼテデネヘベぺメレ", "mouth_eee"],
    ["オコゴソゾトドノホボポモヨロヲョ", "mouth_ooo"],
    ["ン", "mouth_nnn"]
]

class MouthShapeController:
    def __init__(self, clock, time_mouth_map):
        self.clock = clock
        self.start_time = clock.time()
        self.time_mouth_map = time_mouth_map
        self.discrete_parameter = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def control(self, pose, pose_parameters):
        """

        Returns:
            - True if 
        """
        if self.time_mouth_map is None:
            for mouth_shape in mouth_shapes:
                pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0
            return False

        duration_from_sentence_start = (self.clock.time() - self.start_time)

        current_time_mouth = None
        for time_mouth in self.time_mouth_map:
            if time_mouth[0] < duration_from_sentence_start and duration_from_sentence_start <= time_mouth[1]:
                current_time_mouth = time_mouth
                break

        if self.time_mouth_map[-1][1] < duration_from_sentence_start:
            return True

        if current_time_mouth is None:
            for mouth_shape in mouth_shapes:
                pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0
            return False

        name = current_time_mouth[2]

        last_utterance = 0

        # print((time_mouth[1] - time_mouth[0]))

        # duration_from_start = duration_from_sentence_start - current_time_mouth[0]
        # pil_image.save(f"{name}.png")
        progress_rate_in_utterance = \
            (duration_from_sentence_start - current_time_mouth[0]) \
                / ((current_time_mouth[1] - current_time_mouth[0]) * 0.5)
        progress_rate_in_utterance = progress_rate_in_utterance if progress_rate_in_utterance < 1.0 else 1.0
        
        # indice = np.searchsorted([progress_rate_in_utterance], self.discrete_parameter, side="left")[0]

        # degree_rate = self.discrete_parameter[indice]
        if name is not None: #  and (self.clock.time() - last_utterance) > 0.05:
            # print(progress_rate_in_utterance)
            print(f"Controlling with {current_time_mouth}: {progress_rate_in_utterance}")
    
            if name != "mouth_nnn":
                pose[0, pose_parameters.get_parameter_index(name)] = progress_rate_in_utterance
            for mouth_shape in mouth_shapes:
                if mouth_shape != name:
                    pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0

        last_utterance = self.clock.time()
        return False

import queue
import re

import onnxruntime
import simpleaudio as sa
from basicsr.archs.rrdbnet_arch import RRDBNet
# class UpscaleMethod(enum.Enum):
#     RealESRGAN = "realesrgan"
#     RealESRGANOnnx = "realesrgan_onnx"
from basicsr.archs.srresnet_arch import MSRResNet
from PIL import Image
# from pyanime4k import ac
from realesrgan import RealESRGANer


@torch.no_grad()
def upscale_image(model, np_image_rgb, device):
    np_image_rgb = cv2.cvtColor(np_image_rgb, cv2.COLOR_RGB2BGR)
    image_rgb_tensor = torch.tensor(np_image_rgb.astype(np.float32)).to(device)
    image_rgb_tensor /= 255
    image_rgb_tensor = image_rgb_tensor.permute(2, 0, 1)
    image_rgb_tensor = image_rgb_tensor.unsqueeze(0)
    # import IPython; IPython.embed()
    output_img = model(image_rgb_tensor)
    output_img = output_img.data.squeeze().float().clamp_(0, 1)
    output_img = output_img.permute((1, 2, 0))
    output = (output_img * 255.0).round().cpu().numpy().astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output

class LightweightRealESRGANUpscaler:
    def __init__(self):
            # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=3, num_grow_ch=32, scale=4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=16, num_block=3, num_grow_ch=32, scale=4)
        model = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=6, upscale=4)
        # loadnet = torch.load("RealESRGAN_x4plus_anime_6B.pth")
        # loadnet = torch.load("net_g_55000.pth")
        # loadnet = torch.load("net_g_21000.pth")
        # loadnet = torch.load("net_g_45000.pth")
        # loadnet = torch.load("smallest_gan.pth")
        loadnet = torch.load("srgan_v4_60000.pth")
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(device)
        self.device = device

    def upscale(self, pil_image):
        # output, _ = self.upsampler.enhance(np.array(pil_image), outscale=4)
        output = upscale_image(self.model, np.array(pil_image), self.device)
        pil_image = PIL.Image.fromarray(output, mode='RGB')
        return pil_image

from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RealESRGANUpscaler:
    def __init__(self, mode):
        if mode == "rrdbnet":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = get_model_file(
                "RealESRGAN_x4plus_anime_6B.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
        elif mode == "mrresnet":
            model = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=6, upscale=4)
            model_path="srgan_v4_60000.pth"
        elif mode == "srvggnet":
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            model_path = get_model_file(
                "RealESRGANv2-animevideo-xsx4.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth"
            )
        else:
            raise RuntimeError(f"No Such upscaling mode {mode}.")

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False)

    def upscale(self, pil_image):
        output, _ = self.upsampler.enhance(np.array(pil_image), outscale=4)
        pil_image = PIL.Image.fromarray(output, mode='RGBA')

        return pil_image

class FastSRGANUpscaler:
    def __init__(self):
        self.super_resolution_session = onnxruntime.InferenceSession("generator2.onnx", providers = ['CUDAExecutionProvider'])

    def upscale(self, pil_image):
        s = time.time()
        output_image_2 = self.super_resolution_session.run([], {"input_1": [np.array(pil_image)/255]})[0][0]
        output_image_2 = (((output_image_2 + 1) / 2.) * 255).astype(np.uint8)
        output_image_2 = cv2.resize(output_image_2, (output_image_2.shape[0] //2, output_image_2.shape[1]//2))
        pil_image = PIL.Image.fromarray(output_image_2, mode='RGB')
        # pil_image = pil_image.resize((pil_image.width // 2, pil_image.height //2 ))
        print(time.time() - s)
        return pil_image

class RealtimeSuperResolutionUpscaler:
    def __init__(self):
        self.super_resolution_session = onnxruntime.InferenceSession("realtime_super_resolution.onnx", providers = ['CUDAExecutionProvider'])

    def upscale(self, pil_image):
        output_image_2 = self.super_resolution_session.run([], {"image.1": [np.transpose(np.array(pil_image), (2, 0, 1)) / 255]})[0][0]
        pil_image2 = PIL.Image.fromarray(numpy.uint8(numpy.rint(np.transpose(output_image_2[[2, 1, 0], :, :], (1, 2, 0)) * 255)), mode='RGB')
        return pil_image2

import copy

from anime_face_detector import create_detector


def run_inference(onnx_session, input_size, image):
    # リサイズ
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size, input_size))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

    # 前処理
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1).astype('float32')
    x = x.reshape(-1, 3, input_size, input_size)

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # 後処理
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype('uint8')

    return onnx_result

def get_shortest_rotvec_between_two_vector(a, b):
    """Get shortest rotation between two vectors.
    Args:
        a - starting vector of rotation
        b - destination vector of rotation
    
    Returns:
        rotation_axis - axis of rotation
        theta - theta of rotation (in radian)
    """
    
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = a.dot(b)
    # st.write(dot)
    if (1 - dot) < 1e-10:
        return None

    # Because they are unit vectors.
    theta = np.arccos(dot)

    rotation_axis = np.cross(a, b)

    return rotation_axis, theta


class TsukuyomichanVisualizationGenerator:
    def __init__(self, image_path, clock, config, background_color=(0,0,0), transparent_background=False, upscale=True):
        self.device = torch.device("cuda:0")

        self.poser = tha2.poser.modes.mode_20.create_poser(self.device)
        self.pose_parameters = tha2.poser.modes.mode_20.get_pose_parameters()
        self.pose_size = self.poser.get_num_parameters()
        self.pose = torch.zeros(1, self.pose_size).to(self.device)


        self.blink_controller = BlinkController(clock)
        self.mouth_shape_controller = MouthShapeController(clock, None)
        self.body_controller = BodyController(clock)

        self.background_color = background_color
        self.transparent_background = transparent_background

        self.clock = clock

        self.saying_something_ = False
        self.do_blink = True
        self.upscale = upscale
        if upscale:
            self.upscaler = RealESRGANUpscaler(config.UPSCALER)

        # image_path = "character_images/reitsu.png"
        # image_path = "character_images/sozai-rei-yumesaki-mini-open-blank.png"
        # image_path = "character_images/sample6.png"
        image = Image.open(image_path)
        self.whole_image = Image.fromarray(np.array(image))
        if image.mode == 'RGB':
            import IPython; IPython.embed()
            # rgba_image = image.convert('RGBA')
            image_np = np.array(image)
            onnx_session = onnxruntime.InferenceSession("u2net.onnx", execution_provider=["CPUExecutionProvider"])
            out = run_inference(
                onnx_session,
                320,
                image_np,
            )
            resize_image = cv2.resize(out, dsize=(image_np.shape[1], image_np.shape[0]))
            resize_image[resize_image > 255] = 255
            resize_image[resize_image < 125] = 0

            mask = Image.fromarray(resize_image)

            rgba_image = Image.fromarray(image_np).convert('RGBA')
            rgba_image.putalpha(mask)
            whole_image_rgba = np.array(rgba_image)
        else:
            whole_image_rgba = np.array(image)

        whole_image_rgb = np.array(image.convert("RGB"))

        detector = create_detector('yolov3')
        preds = detector(whole_image_rgb)
        first_detection = [pred for pred in preds if pred['bbox'][4] > 0.5][0]
        box = first_detection['bbox'][:4]
        box = [int(p) for p in box]

        face_vec = first_detection['keypoints'][3][:2] - first_detection['keypoints'][1][:2]
        base_vec = [1, 0]

        val = get_shortest_rotvec_between_two_vector(base_vec, face_vec)
        theta = 0
        direction = -1 if val[0] > 0 else 1
        if val is not None:
            theta = val[1]
        # import IPython; IPython.embed()

        # TARGET_SIZE = 120
        # Stable setting
        TARGET_SIZE = 90
        WHOLE_SIZE = 256
        width_ratio = TARGET_SIZE / (box[2] - box[0])
        height_ratio = TARGET_SIZE / (box[3] - box[1])

        self.original_to_face_width_ratio = width_ratio
        self.original_to_face_height_ratio = height_ratio

        margin_width = int((WHOLE_SIZE /width_ratio - (box[2] - box[0])) // 2)
        margin_height = int((WHOLE_SIZE / height_ratio - (box[3] - box[1])) // 2)

        start_y = box[1]-int(margin_height * 1.5)
        start_x = box[0]-margin_width
        end_y = box[3]+int(margin_height * 0.5)
        end_x = box[2]+margin_width

        self.original_start_y = start_y
        self.original_start_x = start_x
        self.original_end_x = end_x
        self.original_end_y = end_y

        buffer = np.zeros((end_y - start_y, end_x - start_x, 4), np.uint8)

        offset_y = 0
        height = end_y - start_y
        width = end_x - start_x
        offset_x = 0
        if start_y < 0:
            offset_y = -start_y
            start_y = 0
        if end_y >= whole_image_rgb.shape[0]:
            height -= (end_y - whole_image_rgb.shape[0])
        if start_x < 0:
            offset_x = -start_x
            start_x = 0
        if end_x >= whole_image_rgb.shape[1]:
            width -= (end_x - whole_image_rgb.shape[1])

        buffer[offset_y:height, offset_x:width] = whole_image_rgba[
            start_y:end_y,
            start_x:end_x
        ]

        self.face_image = cv2.resize(buffer, (256, 256))

        self.pil_face_image = Image.fromarray(self.face_image)
        self.face_theta = direction * np.rad2deg(theta)
        self.rotated_pil_face_image = self.pil_face_image.rotate(-self.face_theta, resample=Image.BICUBIC)

        # self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike("neutral_face.png")).to(self.device)
        self.torch_input_image = extract_pytorch_image_from_PIL_image(
            self.rotated_pil_face_image
        ).to(self.device)

    def saying_something(self):
        return self.saying_something_

    def set_mouth_shape_sequenece(self, mouth_shape_sequence, emotion_label):
        self.saying_something_ = True
        self.mouth_shape_controller = MouthShapeController(self.clock, mouth_shape_sequence)
        print(emotion_label)
        if emotion_label == "happy":
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 1.0
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 1.0
            self.do_blink = False
        elif emotion_label is None:
            return
        else:
            if emotion_label == "awate":
                self.do_blink = False
                self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 0.0
                self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 0.0
            # self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike(f"{emotion_label}_face.png")).to(self.device)
            # self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike).to(self.device)
            self.torch_input_image = extract_pytorch_image_from_PIL_image(self.rotated_pil_face_image).to(self.device)
            # import IPython; IPython.embed()

    def generate(self):
        if self.do_blink:
            blink_rate = self.blink_controller.blink_rate()
            # This is workaround for a bug.
            if blink_rate is None:
                blink_rate = 0.0
                
            # assert blink_rate >= 0.0 and blink_rate <= 1.0, blink_rate
            self.pose[0, self.pose_parameters.get_parameter_index("eye_wink_right")] = blink_rate
            self.pose[0, self.pose_parameters.get_parameter_index("eye_wink_left")] = blink_rate

        # self.body_controller.control(self.pose, self.pose_parameters)
        finished = self.mouth_shape_controller.control(self.pose, self.pose_parameters)
        if finished:
            # self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike(f"neutral_face.png")).to(self.device)
            # self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike("zunda_256.png")).to(self.device)
            self.torch_input_image = extract_pytorch_image_from_PIL_image(self.rotated_pil_face_image).to(self.device)
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 0.0
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 0.0

            self.do_blink = True
            self.saying_something_ = False

            for mouth_shape in mouth_shapes:
                self.pose[0, self.pose_parameters.get_parameter_index(mouth_shape)] = 0.0


        s = time.time()
        # s = time.time()
        # self.rotated_pil_face_image.save("/tmp/before_edit.png")
        output_image = self.poser.pose(self.torch_input_image, self.pose)[0]
        # print(time.time() - s)
        # import IPython; IPython.embed()
        # print(1 / (time.time() - s))

        s = time.time()
        output_image = output_image.detach().cpu()
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        # pil_image.save("/tmp/after_edit.png")
        # # if not self.transparent_background:
        # #     background = PIL.Image.new('RGBA', pil_image.size, self.background_color)
        #     pil_image = PIL.Image.alpha_composite(background, pil_image).convert('RGB')
        # print(1 / (time.time() - s))
    
        if self.upscale:
            # pil_image = pil_image.resize((pil_image.width * 4, pil_image.height * 4), PIL.Image.ANTIALIAS)
            s = time.time()
            pil_image = self.upscaler.upscale(pil_image)
            
            pil_image = pil_image.rotate(self.face_theta)
            pil_image = pil_image.resize((
                int(pil_image.size[0] / 4 / self.original_to_face_width_ratio),
                int(pil_image.size[1] / 4 / self.original_to_face_height_ratio)
                ))

            # import IPython; IPython.embed()

            empty_image = Image.new('RGBA', self.whole_image.size, (0,0,0,0))
            empty_image.paste(pil_image, (self.original_start_x, self.original_start_y))
            composite_image = Image.alpha_composite(self.whole_image.convert('RGBA'), empty_image)

            # composite_image.save("/tmp/merged.png")
    
            if not self.transparent_background:
                background = PIL.Image.new('RGBA', self.whole_image.size, (0, 255,0,255))
                composite_image = PIL.Image.alpha_composite(background, composite_image).convert('RGB')

            print("Real ESRGan", 1 / (time.time() - s))
            # output_image_2 *= 255
            # output_image_2 = output_image_2[:,:,::-1]

            # pil_image.show()
            
            return composite_image

        return pil_image

import cv2
import sentencepiece as spm


class WallClock():
    def time(self):
        return time.time()

class Speaker():
    def output(self, wav, fs):
        play_obj = sa.play_buffer(wav, 1, 2, fs)

# from english_to_kana import EnglishToKana

class SentimentJaFeelingEstimator:
    def __init__(self):
        self.emotion_analyzer = onnxruntime.InferenceSession(
            get_model_file_from_gdrive("sentiment.onnx", "https://drive.google.com/uc?id=1ij9WEObAUJir60qpR1RERlB4-ewiPFVZ"), 
            providers = ['CPUExecutionProvider'])
        
        self._sp = spm.SentencePieceProcessor()
        self._sp.load("sp.model")
        self._maxlen = 281
        self.emotion_label = [
            "happy", "sad", "angry", "disgust", "surprise", "fear"
        ]
    def get_feeling(self, sentence):
        word_ids = self._sp.EncodeAsIds(sentence)
        padded = np.pad(word_ids, (self._maxlen - len(word_ids), 0), 'constant', constant_values=(0, 0))
        emotions = self.emotion_analyzer.run([], {"embedding_input": [padded]})[0][0]
        emotion_index = np.argmax(emotions)
        emotion_label = self.emotion_label[emotion_index] if emotions[emotion_index] > 0.9 else None

        print(emotions)

        return emotion_label


from simpletransformers.classification import (ClassificationArgs,
                                               ClassificationModel)


class FeelingJaFeelingEstimator:
    CLASS_NAMES = [
        'angry_face', 'crying_face', 'face_with_crossed-out_eyes', 'face_with_open_mouth', 
        'flushed_face', 'grinning_face_with_smiling_eyes', 'loudly_crying_face', 'pouting_face', 
        'slightly_smiling_face', 'smiling_face_with_smiling_eyes', 'sparkles', 'tired_face']

    feeling_face_map = {
        'angry_face': 'angry',
        'crying_face': 'sad',
        "face_with_crossed-out_eyes": "awate",
        "face_with_open_mouth": "neutral",
        "flushed_face": "embarrassed",
        "grinning_face_with_smiling_eyes": "happy",
        "loudly_crying_face": "cry",
        "pouting_face": "angry",
        "slightly_smiling_face": "neutral",
        "smiling_face_with_smiling_eyes": "happy",
        "sparkles": "sparkles",
        "tired_face": "awate"
    }

    def __init__(self):
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id="xiongjie/face-expression-ja")
        model_args = ClassificationArgs()
        model_args.onnx = True
        self.model_onnx = ClassificationModel(
            'auto',
            path,
            use_cuda=False,
            args=model_args
        )

    def get_feeling(self, sentence):
        class_id = self.model_onnx.predict([sentence])[0][0]
        return self.feeling_face_map[self.CLASS_NAMES[class_id]]

class TsukuyomichanWrapper:
    MAX_WAV_VALUE = 32768.0
    fs = 24000

    def __init__(self, talksoft, nlp):
        self.talksoft = talksoft
        self.nlp = nlp

        d = ModelDownloader()
        aaa = d.download_and_unpack("kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave")


        self.aligner = CTCSegmentation( **aaa , fs=self.talksoft.fs )

        self.aligner.set_config( gratis_blank=True, kaldi_style_text=False )

    def _get_mouth_map(self, sentence):
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

        sentence = "".join(uttrs)

        for uttr, reading in zip(uttrs, readings):
            print(uttr, reading)

        segments = None
        try:
            segments = self.aligner(wav, uttrs)
        except IndexError:
            import traceback
            traceback.print_exc()
        if segments:
            time_mouth_map = []

            for segment, reading in zip(segments.segments, readings):
                position = segment[0]
                div = (segment[1] - segment[0]) / len(reading)
                for char in reading:
                    mouth_shape = None
                    for mouth_item in mouth_map:
                        if char in mouth_item[0]:
                            mouth_shape = mouth_item[1]
                            break

                    time_mouth_map.append(
                        [position, position+div, mouth_shape]
                    )
                    position += div

            print(time_mouth_map)
            if time_mouth_map is not None:
                self.visualization_generator.set_mouth_shape_sequenece(
                    time_mouth_map, emotion_label
                )
                time_mouth_map = None


    def generate(self, alphabet_replaced_sentence):

        s = time.time()
        print(f"audio is generated from {alphabet_replaced_sentence}")
        if len(alphabet_replaced_sentence) > 200:
            all_wavs = []
            for partial_sentence in alphabet_replaced_sentence.split("。"):
                wav = self.talksoft.generate_voice(partial_sentence, 1)
                wav = wav * self.MAX_WAV_VALUE
                wav = wav.astype(np.int16)
                all_wavs.append(wav)
            wav = np.concatenate(wav)
            print(time.time() - s)
        else:
            wav = self.talksoft.generate_voice(alphabet_replaced_sentence, 1)
            wav = wav * self.MAX_WAV_VALUE
            wav = wav.astype(np.int16)
            print(time.time() - s)
        return wav

import json
from io import BytesIO
from urllib.parse import urlencode

import requests
from scipy.io.wavfile import read


class VoicevoxTalksoft:
    fs = 24000
    def __init__(self, type):
        self.type = type

    def generate(self, text, speech_option={}):
        speaker_id = None
        if self.type == "zundamon":
            speaker_id = 3
        elif self.type == "tsumugi":
            speaker_id = 8
        r = requests.post('http://localhost:50021/audio_query', params={"speaker": speaker_id, 'text': text})
        # import IPython; IPython.embed()
        res_json = json.loads(r.text)
        #import IPython; IPython.embed()

        speed = 1.0
        if 'length' in speech_option:
            speeds = list(np.arange(1.0, 1.4, 0.05))
            closest_wav = None
            closest_diff_length_s = 1000000000000.0
            closest_speed = None
            for speed in speeds:
                res_json['speedScale'] = speed
                r = requests.post('http://localhost:50021/synthesis', params={"speaker": speaker_id}, json=res_json)
                rate, data = read(BytesIO(r.content))
                diff_length_s = abs(speech_option["length"] - (len(data) / self.fs))
                if diff_length_s < closest_diff_length_s:
                    closest_wav = data
                    closest_diff_length_s = diff_length_s
                    closest_speed = speed
            print(closest_diff_length_s)
            print(closest_speed)
            data = closest_wav
            speed = closest_speed
        else:
            r = requests.post('http://localhost:50021/synthesis', params={"speaker": speaker_id}, json=res_json)
            rate, data = read(BytesIO(r.content))

        time_mouth_map = []
        current_time = res_json['prePhonemeLength']
        for accent_phrase in res_json['accent_phrases']:
            if accent_phrase['pause_mora'] is not None:
                current_time += accent_phrase['pause_mora']['vowel_length']
            for moras in accent_phrase['moras']:
                c_length = moras['consonant_length'] if moras['consonant_length'] is not None else 0
                v_length = moras['vowel_length'] if moras['vowel_length'] is not None else 0
                sound_length = (c_length + v_length) / speed
                mouth_shape = None
                for mouth_item in mouth_map:
                    if moras['text'] in mouth_item[0]:
                        mouth_shape = mouth_item[1]
                        break
                time_mouth_map.append([current_time, current_time + sound_length, mouth_shape])
                current_time += sound_length

        # import IPython; IPython.embed()

        return data, time_mouth_map

class TsukuyomichanVisualizer:
    def __init__(self, 
        talksoft, config, clock=WallClock(), wav_output=Speaker(), background_color=None,
        feeling_estimator=FeelingJaFeelingEstimator, transparent_background=False, 
        character="zunda1"
    ):
        self.nlp = spacy.load('ja_ginza_electra')

        if character == "tsumugi1":
            image_path = "character_images/tsumugi1.png"
            self.talksoft = VoicevoxTalksoft("tsumugi")
        elif character == "tsukuyomichan1":
            self.talksoft = TsukuyomichanWrapper(talksoft, self.nlp)
            image_path = "character_images/tsukuyomi3_small.png"

        elif character == "zunda1":
            self.talksoft = VoicevoxTalksoft("zundamon")
            image_path = "character_images/zunda1.png"

        self.clock = clock
        self.wav_output = wav_output

        self.visualization_generator = TsukuyomichanVisualizationGenerator(image_path, self.clock, config, background_color, transparent_background)
        if feeling_estimator is None:
            self.feeling_estimator = None
        else:
            self.feeling_estimator = feeling_estimator()
        base_dictionary = {
            "github": "ギットハブ",
            "FastSRGAN": "ファストエスアールガン",
            "GAN": "ガン",
            "ESRGAN": "イーエスアールガン",
            "Real": "リアル",
            "Bicubic": "バイキュービック",
            "Realtime": "リアルタイム",
            "GB": "ギガバイト",
            "ORT": "オーアールティー",
            "1GB": "イチギガバイト",
            "3D": "スリーディー",
            "Live2D": "ライブツーディー"
        }
        for line in open("english_to_kana_dictionary.txt"):
            english, kana = line[:-1].split(",")
            english = english.strip()
            kana = kana.strip()
            base_dictionary[english] = kana

        self.english_to_kana_dictionary = {}
        # TODO: decide it is necessary.
        for key in base_dictionary.keys():
            self.english_to_kana_dictionary[key.lower()] = base_dictionary[key]
            self.english_to_kana_dictionary[key] = base_dictionary[key]

        # self.english2kana = EnglishToKana()
        self.config = config

    def saying_something(self):
        return self.visualization_generator.saying_something()

    def visualize(self, sentence, generate_visual=True, speech_option={}):
        # print("wait visualization")
        if sentence is not None:
            print(f"got {sentence}")
            
            sentence = ''.join(['' if c in emoji.UNICODE_EMOJI['en'] else c for c in sentence])
            sentence = re.sub("<unk>", "", sentence)
            sentence = re.sub(r":.+", "", sentence)
            sentence = sentence.strip()

            emotion_label = ""
            if self.feeling_estimator is not None:
                emotion_label = self.feeling_estimator.get_feeling(sentence )

            alphabet_replaced_sentence = ""
            doc = self.nlp(sentence)
            for sent in doc.sents:
                for token in sent:
                    if re.match(r"[a-zA-Z]+", token.orth_):
                        yomi = token.orth_
                        if re.match(r"[a-zA-Z]+", yomi) and yomi.lower() in self.english_to_kana_dictionary:
                            yomi = self.english_to_kana_dictionary[yomi.lower()]
                        if re.match(r"[a-zA-Z]+", yomi):
                            yomi = ginza.reading_form(token)
                        # another_yomi = self.english2kana.convert(yomi.lower())
                        # if re.match(r"[a-zA-Z]+", yomi) and another_yomi is not None:
                        #     yomi = another_yomi
                        # import IPython; IPython.embed()
                        alphabet_replaced_sentence += yomi
                    else:
                        alphabet_replaced_sentence += token.orth_

            print(alphabet_replaced_sentence)

            sentence = alphabet_replaced_sentence


            wav, time_mouth_map = self.talksoft.generate(alphabet_replaced_sentence, speech_option)
            if generate_visual:
                self.visualization_generator.set_mouth_shape_sequenece(
                    time_mouth_map, emotion_label
                )
                
            self.wav_output.output(wav, self.talksoft.fs)

        if generate_visual:
            pil_image = self.visualization_generator.generate()
            return pil_image
        else:
            None
        # pil_image = pil_image.resize((pil_image.width // 2, pil_image.height // 2), PIL.Image.ANTIALIAS)
        # import io
        # s = time.time()
        # bio = io.BytesIO()
        # pil_image.save(bio, format="PNG")
        # del next_animation_image
    
        
        # window["tsukuyomi_image"].update(bio.getvalue())
        # print(time.time() - s)


        # print("outputting image")

        # image_queue.put_nowait(pil_image)
        # print(time.time() - s)
    # cv2.imshow("tsukuyomi chan", np.array(pil_image)[:, :, ::-1])
    # # cv2.imshow("tsukuyomi chan", np.array(pil_image)[:, :, ::-1])
    # cv2.waitKey(3)
