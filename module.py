import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import parselmouth
import hashlib
from ast import literal_eval
from tqdm import tqdm

from .slicer import Slicer
from .ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from .ddsp.core import upsample
from .diffusion.vocoder import load_model_vocoder

def check_args(ddsp_args, diff_args):
    if ddsp_args.data.sampling_rate != diff_args.data.sampling_rate:
        print("Unmatch data.sampling_rate!")
        return False
    if ddsp_args.data.block_size != diff_args.data.block_size:
        print("Unmatch data.block_size!")
        return False
    if ddsp_args.data.encoder != diff_args.data.encoder:
        print("Unmatch data.encoder!")
        return False
    return True
    
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-diff",
        "--diff_ckpt",
        type=str,
        required=True,
        help="path to the diffusion model checkpoint",
    )
    parser.add_argument(
        "-ddsp",
        "--ddsp_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to the DDSP model checkpoint (for shallow diffusion)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='rmvpe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe, fcpe, rmvpe (default)",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-diffid",
        "--diff_spk_id",
        type=str,
        required=False,
        default='auto',
        help="diffusion speaker id (for multi-speaker model) | default: auto",
    )
    parser.add_argument(
        "-speedup",
        "--speedup",
        type=str,
        required=False,
        default='auto',
        help="speed up | default: auto",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='auto',
        help="ddim, pndm, dpm-solver or unipc | default: auto",
    )
    parser.add_argument(
        "-kstep",
        "--k_step",
        type=str,
        required=False,
        default=None,
        help="shallow diffusion steps | default: None",
    )
    return parser.parse_args(args=args, namespace=namespace)

    
def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result

class DDSP_SVC:
    def __init__(
            self, 
            sample_rate,
            diff_ckpt=None, device=None,
            pitch_extractor='rmvpe', f0_min=50, f0_max=1100, threshold=-60, 
            ddsp_ckpt = None, k_step = None
        ):
        # パラメタ
        self.here = os.path.dirname(os.path.abspath(__file__))
        if diff_ckpt is None:
            diff_ckpt = os.path.join(
                self.here,
                "exp/diffusion-test/model_100000.pt"
            )
        print("DDSP-SVC ckpt:", diff_ckpt)
        print('Pitch extractor type: ' + pitch_extractor)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.sample_rate = sample_rate
        self.threshold = threshold

        # vocoder
        self.model, self.vocoder, self.args = load_model_vocoder(diff_ckpt, self.here, device=self.device)

        # pitch
        self.hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        self.pitch_extractor = F0_Extractor(
                            pitch_extractor, 
                            sample_rate, 
                            self.hop_size, 
                            float(f0_min), 
                            float(f0_max),
                            ddsp_path=self.here)

        # ddsp
        ddsp = None
        ddsp_is_not_identified = False
        if self.args.model.type == 'DiffusionNew' or self.args.model.type == 'DiffusionFast':
            if k_step is not None:
                k_step = int(k_step)
                if k_step > self.args.model.k_step_max:
                    k_step = self.args.model.k_step_max
            else:
                k_step = self.args.model.k_step_max
            print('Shallow diffusion step: ' + str(k_step))
            if ddsp_ckpt is not None:
                # load ddsp model
                ddsp, ddsp_args = load_model(ddsp_ckpt, device=self.device)
                if not check_args(ddsp_args, self.args):
                    print("Cannot use this DDSP model for shallow diffusion, the built-in DDSP model will be used!")
                    ddsp = None
            else:
                print("DDSP model is not identified, the built-in DDSP model will be used!")
        else:
            if k_step is not None:
                k_step = int(k_step)
                print('Shallow diffusion step: ' + str(k_step))
                if ddsp_ckpt is not None:
                    # load ddsp model
                    ddsp, ddsp_args = load_model(ddsp_ckpt, device=self.device)
                    if not check_args(ddsp_args, self.args):
                        print("Cannot use this DDSP model for shallow diffusion, gaussian diffusion will be used!")
                        ddsp = None
                else:
                    print('DDSP model is not identified!')
                    print('Extracting the mel spectrum of the input audio for shallow diffusion...')
                    ddsp_is_not_identified = True
                    audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                    input_mel = vocoder.extract(audio_t, sample_rate)
                    input_mel = torch.cat((input_mel, input_mel[:,-1:,:]), 1)
            else:
                print('Shallow diffusion step is not identified, gaussian diffusion will be used!')
        self.ddsp = ddsp
        self.ddsp_is_not_identified = ddsp_is_not_identified
        self.k_step = k_step

    def __call__(
            self, 
            ifname, spk_id, key, 
            ofname=None, 
            formant_shift_key=0, diff_spk_id='auto', spk_mix_dict="None",
            speedup='auto', method='auto'
        ):  
        # load input
        audio, sample_rate = librosa.load(ifname, sr=None)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
    
        # extract f0
        f0 = self.pitch_extractor.extract(audio, uv_interp = True, device = self.device)        
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # key change
        f0 = f0 * 2 ** (float(key) / 12)
        
        # formant change
        formant_shift_key = torch.from_numpy(np.array([[float(formant_shift_key)]])).float().to(self.device)
        
        # extract volume 
        print('Extracting the volume envelope of the input audio...')
        volume_extractor = Volume_Extractor(self.hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(self.threshold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # load units encoder
        if self.args.data.encoder == 'cnhubertsoftfish':
            cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
        units_encoder = Units_Encoder(
                            self.args.data.encoder, 
                            self.args.data.encoder_ckpt, 
                            self.args.data.encoder_sample_rate, 
                            self.args.data.encoder_hop_size,
                            cnhubertsoft_gate=cnhubertsoft_gate,
                            device = self.device,
                            ddsp_path=self.here)
                                
        # speaker id or mix-speaker dictionary
        spk_mix_dict = literal_eval(spk_mix_dict)
        spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)
        if diff_spk_id == 'auto':
            diff_spk_id = spk_id
        else:
            diff_spk_id = torch.LongTensor(np.array([[int(diff_spk_id)]])).to(self.device)
        if spk_mix_dict is not None:
            print('Mix-speaker mode')
        else:
            print('DDSP Speaker ID: '+ str(int(spk_id)))
            print('Diffusion Speaker ID: '+ str(diff_spk_id)) 
        
        # speed up
        if speedup == 'auto':
            infer_speedup = self.args.infer.speedup
        else:
            infer_speedup = int(speedup)
        if method == 'auto':
            method = self.args.infer.method
        else:
            method = method
        if infer_speedup > 1:
            print('Sampling method: '+ method)
            print('Speed up: '+ str(infer_speedup))
        else:
            print('Sampling method: DDPM')
        
        input_mel = None
        if self.ddsp_is_not_identified:
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            input_mel = self.vocoder.extract(audio_t, sample_rate)
            input_mel = torch.cat((input_mel, input_mel[:,-1:,:]), 1)
            
        # forward and save the output
        result = np.zeros(0)
        current_length = 0
        segments = split(audio, sample_rate, self.hop_size)
        print('Cut the input audio into ' + str(len(segments)) + ' slices')
        with torch.no_grad():
            for segment in tqdm(segments):
                start_frame = segment[0]
                seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
                seg_units = units_encoder.encode(seg_input, sample_rate, self.hop_size)
            
                seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
                seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
                if self.ddsp is not None:
                    seg_ddsp_f0 = 2 ** (-float(cmd.formant_shift_key) / 12) * seg_f0
                    seg_ddsp_output, _ , (_, _) = self.ddsp(seg_units, seg_ddsp_f0, seg_volume, spk_id = spk_id, spk_mix_dict = spk_mix_dict)
                    seg_input_mel = self.vocoder.extract(seg_ddsp_output, self.args.data.sampling_rate, keyshift=float(cmd.formant_shift_key))
                elif input_mel is not None:
                    seg_input_mel = input_mel[:, start_frame : start_frame + seg_units.size(1), :]
                else:
                    seg_input_mel = None
                    
                seg_mel = self.model(
                        seg_units, 
                        seg_f0, 
                        seg_volume, 
                        spk_id = diff_spk_id, 
                        spk_mix_dict = spk_mix_dict,
                        aug_shift = formant_shift_key,
                        vocoder=self.vocoder,
                        gt_spec=seg_input_mel,
                        infer=True, 
                        infer_speedup=infer_speedup, 
                        method=method,
                        k_step=self.k_step)
                seg_output = self.vocoder.infer(seg_mel, seg_f0)
                seg_output *= mask[:, start_frame * self.args.data.block_size : (start_frame + seg_units.size(1)) * self.args.data.block_size]
                seg_output = seg_output.squeeze().cpu().numpy()
                
                silent_length = round(start_frame * self.args.data.block_size) - current_length
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = cross_fade(result, seg_output, current_length + silent_length)
                current_length = current_length + silent_length + len(seg_output)
            if ofname is not None:
                sf.write(ofname, result, self.args.data.sampling_rate)
        