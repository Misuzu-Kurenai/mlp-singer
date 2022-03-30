import argparse
import csv

import torch
import os
import numpy as np
import subprocess
import pathlib
from preprocess_koko import seq2frame
from utils import load_model
from collections import namedtuple

KOKO2022_labels = [
    'sil',
    'pau',
    'br',
    'a', 'i', 'u', 'e', 'o', 
    'k', 's', 't', 'n', 'h', 'm', 'y', 'r', 'w',
    'g', 'z', 'd', 'b', 'p', 
    'ky', 'sh', 'ch', 'ty', 'ny', 'hy', 'f' , 'my', 'ry', 'ts', 
    'gy', 'j' , 'dy', 'by', 'py', 
    'v' , 'N' , 'cl']

HIRA_TO_PHONEME = {
    "あ": ["a"], "い": ["i"], "う": ["u"], "え": ["e"], "お": ["o"],
    "か": ["k", "a"], "き": ["k", "i"], "く": ["k", "u"], "け": ["k", "e"], "こ": ["k", "o"],
    "さ": ["s", "a"], "し": ["sh", "i"], "す": ["s", "u"], "せ": ["s", "e"], "そ": ["s", "o"],
    "た": ["t", "a"], "ち": ["ch", "i"], "つ": ["ts", "u"], "て": ["t", "e"], "と": ["t", "o"],
    "な": ["n", "a"], "に": ["n", "i"], "ぬ": ["n", "u"], "ね": ["n", "e"], "の": ["n", "o"],
    "は": ["h", "a"], "ひ": ["h", "i"], "ふ": ["f", "u"], "へ": ["h", "e"], "ほ": ["h", "o"],
    "ま": ["m", "a"], "み": ["m", "i"], "む": ["m", "u"], "め": ["m", "e"], "も": ["m", "o"],
    "や": ["y", "a"], "ゆ": ["y", "u"], "よ": ["y", "o"],
    "ら": ["r", "a"], "り": ["r", "i"], "る": ["r", "u"], "れ": ["r", "e"], "ろ": ["r", "o"],
    "わ": ["w", "a"], "を": ["o"], "ん": ["N"], "うぉ": ["w", "o"],

    "が": ["g", "a"], "ぎ": ["g", "i"], "ぐ": ["g", "u"], "げ": ["g", "e"], "ご": ["g", "o"],
    "ざ": ["z", "a"], "じ": ["j", "i"], "ず": ["z", "u"], "ぜ": ["z", "e"], "ぞ": ["z", "o"],
    "だ": ["d", "a"], "で": ["d", "e"], "ど": ["d", "o"],
    "ば": ["b", "a"], "び": ["b", "i"], "ぶ": ["b", "u"], "べ": ["b", "e"], "ぼ": ["b", "o"],
    "ぱ": ["p", "a"], "ぴ": ["p", "i"], "ぷ": ["p", "u"], "ぺ": ["p", "e"], "ぽ": ["p", "o"],

    "きゃ": ["ky", "a"], "きゅ": ["ky", "u"], "きょ": ["ky", "o"],
    "しゃ": ["sh", "a"], "しゅ": ["sh", "u"], "しょ": ["sh", "o"],
    "ちゃ": ["ch", "a"], "ちゅ": ["ch", "u"], "ちょ": ["ch", "o"],
    "てゃ": ["ty", "a"], "てゅ": ["ty", "u"], "てょ": ["ty", "o"],
    "にゃ": ["ny", "a"], "にゅ": ["ny", "u"], "にょ": ["ny", "o"],
    "ひゃ": ["hy", "a"], "ひゅ": ["hy", "u"], "ひょ": ["hy", "o"],
    "ふぁ": ["f", "a"], "ふぃ": ["f", "i"], "ふぇ": ["f", "e"], "ふぉ": ["f", "o"],
    "みゃ": ["my", "a"], "みゅ": ["my", "u"], "みょ": ["my", "o"],
    "りゃ": ["ry", "a"], "りゅ": ["ry", "u"], "りょ": ["ry", "o"],
    "つぁ": ["ts", "a"], "つぃ": ["ts", "i"], "つぇ": ["ts", "e"], "つぉ": ["ts", "o"],

    "ぎゃ": ["gy", "a"], "ぎゅ": ["gy", "u"], "ぎょ": ["gy", "o"],
    "じゃ": ["j", "a"], "じゅ": ["j", "u"], "じょ": ["j", "o"],
    "ずぃ": ["z", "i"],
    "でゃ": ["dy", "a"], "でぃ": ["dy", "i"], "でゅ": ["dy", "u"], "でぇ": ["dy", "e"], "でょ": ["dy", "o"],
    "びゃ": ["by", "a"], "びゅ": ["by", "u"], "びょ": ["by", "o"],
    "ぴゃ": ["py", "a"], "ぴゅ": ["py", "u"], "ぴょ": ["py", "o"],
    "ゔぁ": ["v", "a"], "ゔぃ": ["v", "i"], "ゔ": ["v", "u"], "ゔぇ": ["v", "e"], "ゔぉ": ["v", "o"],

    "<sil>": ["sil"],
    "<pau>": ["pau"],
    "<br>": ["br"],
    "<cl>": ["cl"],
    "ー": ["cont"]
}

def load_phoneme_dist(phoneme_dist_path):
    phoneme_dist = {}
    PhonemeConnection = namedtuple(
        "PhonemeConnection",
        ["phoneme", "prev_mean", "prev_std", "next_mean", "next_std"])
    with open(phoneme_dist_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            """
            phoneme means compound phoneme of consonant and vowel,
            prev_mean means mean of duration of the consotant,
            prev_std means (unbiased) standard deviation of the consotant,
            next_mean means mean of duration of the vowel, and
            next_std means (unbiased) standard deviation of the vowel.
            """
            phoneme = row["phoneme"]
            prev_mean = float(row["prev_mean"])
            prev_std  = float(row["prev_std"])
            next_mean = float(row["next_mean"])
            next_std  = float(row["next_std"])
            phoneme_connection = PhonemeConnection(
                phoneme, prev_mean, prev_std, next_mean, next_std
            )
            phoneme_dist[phoneme] = phoneme_connection
    
    return phoneme_dist


def to_int(int_str):
    if len(int_str) == 0:
        return 0
    else:
        return int(int_str)

def to_float(float_str):
    if len(float_str) == 0:
        return 0
    else:
        return float(float_str)

def to_midi_num(midi_str):
    # parse string
    base_midi = midi_str[0]
    if len(midi_str) > 1:
        octave = int(midi_str[1])
    else:
        octave = 4
    
    if len(midi_str) > 2:
        sharp_str = midi_str[2]
        if sharp_str == "#":
            sharp = 1
        elif sharp_str == "b":
            sharp = -1
    else:
        sharp = 0
    
    # construct midi num
    note_dict = {
        "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11
    }
    midi_num = note_dict[base_midi] + octave * 12 + 12 + sharp

    return midi_num

def append_phoneme(phoneme_list, phoneme_dist, frame_len, phoneme_seq, start_frame, end_frame):
    if len(phoneme_list) == 1:
        phoneme = phoneme_list[0]
        phoneme_idx = KOKO2022_labels.index(phoneme)
        phoneme_seq.append( (phoneme_idx, start_frame, end_frame) )
        pass
    elif len(phoneme_list) >= 2:
        compound_phoneme = "".join(phoneme_list)
        phoneme_connection = phoneme_dist[compound_phoneme]
        consonant_mean = phoneme_connection.prev_mean
        consonant_frame_len = round(consonant_mean / frame_len)

        if consonant_frame_len > end_frame - start_frame:
            phoneme = phoneme_list[0]
            phoneme_idx = KOKO2022_labels.index(phoneme)
            phoneme_seq.append( (phoneme_idx, start_frame, end_frame) )
        else:
            phoneme = phoneme_list[0]
            phoneme_idx = KOKO2022_labels.index(phoneme)
            phoneme_seq.append( (phoneme_idx, start_frame, start_frame + consonant_frame_len) )

            phoneme = phoneme_list[1]
            phoneme_idx = KOKO2022_labels.index(phoneme)
            phoneme_seq.append( (phoneme_idx, start_frame + consonant_frame_len, end_frame) )


        #print(phoneme_connection)
    pass

def main(args):
    input_csv = pathlib.Path(args.input_csv)
    output_npy = input_csv.stem + ".npy"
    tempo = float(args.tempo)

    phoneme_dist = load_phoneme_dist(args.phoneme_dist)
    #print(phoneme_dist)
    phoneme_note_seq = []
    phoneme_seq = []
    note_seq = []
    frame_len = 256 / 22050
    with open(input_csv) as f:
        reader = csv.DictReader(f)
        current = 0
        for row in reader:
            
            sec_str = row["sec"]
            dur_str = row["duration"]
            midi_note_str = row["notes"]
            hira = row["phoneme"]

            phoneme_list = HIRA_TO_PHONEME.get(hira, None)


            sec = to_float(sec_str)
            dur = to_float(dur_str) * tempo
            mid_num = to_midi_num(midi_note_str)
            rel_mid_num = mid_num - 52
            
            
            #print(sec, current, dur)
            start_frame = round(current / frame_len)
            end_frame   = round((current + dur) / frame_len)
            note_seq.append( (rel_mid_num, start_frame, end_frame) )
            append_phoneme(phoneme_list, phoneme_dist, frame_len, phoneme_seq, start_frame, end_frame)
            print(hira, phoneme_list, dur, midi_note_str, mid_num, rel_mid_num, dur, start_frame, end_frame - start_frame)
            current += dur
            continue
    
    total_length = round(current / frame_len)

    notes = seq2frame(note_seq, total_length)
    phonemes = seq2frame(phoneme_seq, total_length)

    # load model
    device = args.device
    print("device: ", device)
    
    model = load_model(args.checkpoint_path, eval=True)
    chunk_size = model.seq_len
    notes = notes.to(device)
    phonemes = phonemes.to(device)
    remainder = total_length % chunk_size
    mel_dim = 80
    if remainder:
        pad_size = chunk_size - remainder
        padding = torch.zeros(pad_size, dtype=int).to(device)
        phonemes = torch.cat((phonemes, padding))
        notes = torch.cat((notes, padding))
        batch_phonemes = phonemes.reshape(-1, chunk_size)
        batch_notes = notes.reshape(-1, chunk_size)
        preds = model(batch_notes, batch_phonemes)
        preds = preds.reshape(-1, mel_dim)[:-pad_size]
    else:
        batch_phonemes = phonemes.reshape(-1, chunk_size)
        batch_notes = notes.reshape(-1, chunk_size)
        preds = model(batch_notes, batch_phonemes)
        mel_dim = preds.size(-1)
        preds = preds.reshape(-1, mel_dim)
    preds = preds.transpose(0, 1).unsqueeze(0)
    np.save(os.path.join(args.mel_path, output_npy), preds.detach().numpy())
    
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="little_star.csv")
    parser.add_argument("--phoneme_dist", default="../phoneme_dist.utf-8.csv")
    parser.add_argument("--tempo", default=0.5)
    parser.add_argument(
        "--mel_path",
        type=str,
        default=os.path.join("hifi-gan", "test_mel_files"),
        help="path to save synthesized mel-spectrograms",
    )
    parser.add_argument("--checkpoint_path",
        type=str,
        default="checkpoints/default/last.pt",
        help="path to checkpoint file")
    parser.add_argument(
        "--hifi_gan",
        type=str,
        default="g_02500000",
        help="path to hifi gan generator checkpoint file",
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    args = parser.parse_args()
    main(args)