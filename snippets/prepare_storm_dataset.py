#!/usr/bin/env python3

import argparse
import os
import numpy as np
import soundfile as sf
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print('Musisz zainstalować sklearn: pip install scikit-learn')
    exit(1)
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'SC-Wind-Noise-Generator')))
try:
    import librosa
except ImportError:
    print('Musisz zainstalować librosa: pip install librosa')
    exit(1)
from tqdm import tqdm
import torch

try:
    from sc_wind_noise_generator import WindNoiseGenerator
except ImportError:
    print('Brak pliku sc_wind_noise_generator.py w katalogu SC-Wind-Noise-Generator lub błąd importu.')
    exit(1)

from concurrent.futures import ThreadPoolExecutor, as_completed


def load_wav_list(wav_list_path):
    with open(wav_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def extract_fragments(audio, sr, fragment_sec, overlap):
    fragment_len = int(fragment_sec * sr)
    step = int(fragment_len * (1 - overlap))
    fragments = []
    for start in range(0, len(audio) - fragment_len + 1, step):
        fragments.append(audio[start:start+fragment_len])
    return fragments


def generate_wind_noise(fs, duration, wind_levels, gustiness, seed=None):
    # Wybierz losowy poziom wiatru z listy
    wind_profile = np.random.choice(wind_levels, size=int(duration))
    wn = WindNoiseGenerator(fs=fs, duration=duration, generate=True,
                           wind_profile=wind_profile, gustiness=gustiness, start_seed=seed)
    wn.samples = int(fs * duration)
    wn_signal, _ = wn.generate_wind_noise()
    return wn_signal


def mix_with_wind(clean, wind, snr_db):
    # Dopasuj długość
    min_len = min(len(clean), len(wind))
    clean = clean[:min_len]
    wind = wind[:min_len]
    # Oblicz skalowanie SNR
    clean_power = np.mean(clean ** 2)
    wind_power = np.mean(wind ** 2)
    target_wind_power = clean_power / (10 ** (snr_db / 10))
    wind = wind * np.sqrt(target_wind_power / (wind_power + 1e-8))
    return clean + wind


def worker_task(args_tuple):
    frag, sr, split, idx, args = args_tuple
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    wind = generate_wind_noise(fs=sr, duration=args.fragment_sec, wind_levels=args.wind_levels, gustiness=args.gustiness, seed=args.seed+idx)
    snr_db = random.choice(args.snr)
    noisy = mix_with_wind(frag, wind, snr_db)
    fname = f'{split}_{idx:06d}.wav'
    out_clean = os.path.join(args.out_dir, split, 'clean', fname)
    out_noisy = os.path.join(args.out_dir, split, 'noisy', fname)
    sf.write(out_clean, frag, sr)
    sf.write(out_noisy, noisy, sr)
    return fname


def main():
    parser = argparse.ArgumentParser(description='Przygotuj zbiór treningowy/testowy dla storm z szumem wiatru.')
    parser.add_argument('--wav-list', type=str, required=True, help='Ścieżka do pliku z listą plików wav (jeden na linię)')
    parser.add_argument('--out-dir', type=str, required=True, help='Katalog wyjściowy na dane')
    parser.add_argument('--fragment-sec', type=float, required=True, help='Długość fragmentu w sekundach')
    parser.add_argument('--overlap', type=float, default=0.5, help='Procent nakładania się fragmentów (0-1)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporcja zbioru testowego (np. 0.2)')
    parser.add_argument('--n-samples', type=int, required=True, help='Całkowita liczba próbek (łącznie train+test)')
    parser.add_argument('--wind-levels', type=float, nargs='+', default=[3, 6, 9, 12], help='Poziomy wiatru (m/s)')
    parser.add_argument('--gustiness', type=float, default=5, help='Zmienność wiatru (gustiness)')
    parser.add_argument('--snr', type=float, nargs='+', default=[-6, 0, 6, 12], help='SNR-y do miksowania')
    parser.add_argument('--sr', type=int, default=16000, help='Docelowa częstotliwość próbkowania')
    parser.add_argument('--seed', type=int, default=42, help='Seed do losowania')
    parser.add_argument('--num-workers', type=int, default=4, help='Liczba wątków do generacji danych')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    files = load_wav_list(args.wav_list)
    train_files, test_files = train_test_split(files, test_size=args.test_size, random_state=args.seed)

    splits = [('train', train_files), ('test', test_files)]
    n_train = int(args.n_samples * (1 - args.test_size))
    n_test = args.n_samples - n_train
    n_per_split = {'train': n_train, 'test': n_test}

    for split, split_files in splits:
        out_clean = os.path.join(args.out_dir, split, 'clean')
        out_noisy = os.path.join(args.out_dir, split, 'noisy')
        os.makedirs(out_clean, exist_ok=True)
        os.makedirs(out_noisy, exist_ok=True)
        tasks = []
        idx = 0
        while len(tasks) < n_per_split[split]:
            file = random.choice(split_files)
            audio, sr = sf.read(file)
            if sr != args.sr:
                # Próbkuj do docelowego sr
                audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sr)
                sr = args.sr
            fragments = extract_fragments(audio, sr, args.fragment_sec, args.overlap)
            for frag in fragments:
                if len(tasks) >= n_per_split[split]:
                    break
                tasks.append((frag, sr, split, idx, args))
                idx += 1
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(worker_task, tasks), total=len(tasks), desc=f'Preparing {split}'))

if __name__ == '__main__':
    main() 