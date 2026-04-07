import os
import json
import torch
import numpy as np
import pandas as pd
import h5py
import librosa
import string
import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from moviepy.editor import VideoFileClip
from torch.cuda.amp import autocast
from transformers import BertTokenizer, BertModel
from torchvision.models.feature_extraction import create_feature_extractor

def get_vision_model(device):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor, model_layer

def get_language_model(device):
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return model, tokenizer

def extract_visual_features(episode_path, tr, feature_extractor, model_layer, transform, device, save_dir_temp, save_dir_features, use_fp16=True):
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    visual_features = []
    with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
        for start in start_times:
            clip_chunk = clip.subclip(start, start+tr)
            chunk_path = os.path.join(temp_dir, 'visual_chunk.mp4')
            clip_chunk.write_videofile(chunk_path, verbose=False, audio=False, logger=None)
            video_clip = VideoFileClip(chunk_path)
            chunk_frames = [frame for frame in video_clip.iter_frames()]
            frames_array = np.transpose(np.array(chunk_frames), (3, 0, 1, 2))
            inputs = torch.from_numpy(frames_array).float()
            inputs = transform(inputs).unsqueeze(0).to(device)

            with torch.no_grad():
                if use_fp16 and device.type == 'cuda':
                    with autocast(dtype=torch.float16):
                        preds = feature_extractor(inputs)
                else:
                    preds = feature_extractor(inputs)

            feat = preds[model_layer]
            feat_pooled = torch.mean(feat, dim=(2, 3), keepdim=False)
            feat_pooled = torch.squeeze(feat_pooled).cpu().numpy()
            visual_features.append(feat_pooled.astype('float32'))
            pbar.update(1)

    return np.array(visual_features, dtype='float32')

def extract_audio_features(episode_path, tr, sr, device, save_dir_temp, save_dir_features):
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    audio_features = []
    with tqdm(total=len(start_times), desc="Extracting audio features") as pbar:
        for start in start_times:
            clip_chunk = clip.subclip(start, start+tr)
            chunk_audio_path = os.path.join(temp_dir, 'audio_chunk.wav')
            clip_chunk.audio.write_audiofile(chunk_audio_path, verbose=False, logger=None)
            y, _ = librosa.load(chunk_audio_path, sr=sr, mono=True)
            mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
            audio_features.append(mfcc_features.astype('float32'))
            pbar.update(1)

    return np.array(audio_features, dtype='float32')

def extract_language_features(episode_path, model, tokenizer, num_used_tokens, kept_tokens_last_hidden_state, device, save_dir_features, use_fp16=True):
    df = pd.read_csv(episode_path, sep='\t')
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())
    tokens, np_tokens, pooler_output, last_hidden_state = [], [], [], []

    for i in tqdm(range(df.shape[0]), desc="Extracting language features"):
        if not df.iloc[i]["is_na"]:
            tr_text = df.iloc[i]["text_per_tr"]
            tokens.extend(tokenizer.tokenize(tr_text))
            tr_np_tokens = tokenizer.tokenize(tr_text.translate(str.maketrans('', '', string.punctuation)))
            np_tokens.extend(tr_np_tokens)

        if len(tokens) > 0:
            used_tokens = tokenizer.convert_tokens_to_ids(tokens[-(num_used_tokens):])
            input_ids = [101] + used_tokens + [102]
            tensor_tokens = torch.tensor(input_ids).unsqueeze(0).to(device)
            with torch.no_grad():
                if use_fp16 and device.type == 'cuda':
                    with autocast(dtype=torch.float16):
                        outputs = model(tensor_tokens)
                else:
                    outputs = model(tensor_tokens)
                pooler_output.append(outputs['pooler_output'][0].cpu().numpy().astype('float32'))
        else:
            pooler_output.append(np.full(768, np.nan, dtype='float32'))

        if len(np_tokens) > 0:
            np_feat = np.full((kept_tokens_last_hidden_state, 768), np.nan, dtype='float32')
            used_tokens = tokenizer.convert_tokens_to_ids(np_tokens[-(num_used_tokens):])
            np_input_ids = [101] + used_tokens + [102]
            np_tensor_tokens = torch.tensor(np_input_ids).unsqueeze(0).to(device)
            with torch.no_grad():
                if use_fp16 and device.type == 'cuda':
                    with autocast(dtype=torch.float16):
                        np_outputs = model(np_tensor_tokens)
                else:
                    np_outputs = model(np_tensor_tokens)
                np_outputs = np_outputs['last_hidden_state'][0][1:-1].cpu().numpy().astype('float32')
            tk_idx = min(kept_tokens_last_hidden_state, len(np_tokens))
            np_feat[-tk_idx:, :] = np_outputs[-tk_idx:]
            last_hidden_state.append(np_feat)
        else:
            last_hidden_state.append(np.full((kept_tokens_last_hidden_state, 768), np.nan, dtype='float32'))

    return np.array(pooler_output, dtype='float32'), np.array(last_hidden_state, dtype='float32')

class ExtractionCheckpoint:
    def __init__(self, checkpoint_dir, log_dir):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, "extraction_progress.json")
        self.log_file = os.path.join(log_dir, f"extraction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.progress = self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'completed_episodes': {}, 'failed_episodes': {}, 'total_episodes_processed': 0}

    def save_checkpoint(self):
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def mark_episode_complete(self, episode_name, modalities_extracted):
        self.progress['completed_episodes'][episode_name] = {'modalities': modalities_extracted, 'status': 'success'}
        self.progress['total_episodes_processed'] += 1
        self.save_checkpoint()

    def mark_episode_failed(self, episode_name, error_msg, attempted_modalities):
        self.progress['failed_episodes'][episode_name] = {'error': error_msg, 'attempted_modalities': attempted_modalities, 'status': 'failed'}
        self.save_checkpoint()

    def is_episode_completed(self, episode_name):
        return episode_name in self.progress['completed_episodes']
