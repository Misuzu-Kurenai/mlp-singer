import argparse
import pathlib
import csv

from data.dsp import TacotronSTFT
import mido
import torch
from utils import load_config

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

class PreprocessorKoKo2022:
    def __init__(self, config):
        self.stft = TacotronSTFT(
            config.n_fft,
            config.hop_length,
            config.win_length,
            config.n_mel_channels,
            config.sampling_rate,
            config.mel_fmin,
            config.mel_fmax,
            config.max_wav_value,
        )
        self.frame_len = self.stft.frame_len
        self.consonant_emphasis = config.consonant_emphasis
        self.min_note = config.min_note

    def __call__(self, midi_path, text_path, wav_path):
        mel = self.preprocess_audio(wav_path)
        total_length = len(mel)
        note_sequence = get_note_sequence(midi_path, self.frame_len)
        label_sequence = get_label_sequence(text_path, self.frame_len)

        notes = seq2frame(note_sequence, total_length, min_data=self.min_note)
        phonemes = seq2frame(label_sequence, total_length)
        return notes, phonemes, mel

    def preprocess_audio(self, wav_path: str):
        wav = self.stft.load_wav(wav_path)
        mel = self.stft.mel_spectrogram(wav)
        return mel

    def prepare_inference(self, midi_path, text_path):
        midi_file = mido.MidiFile(midi_path)
        note_sequence = get_note_sequence(midi_path, self.frame_len)
        
        total_length = round(1000 * midi_file.length / self.frame_len)
        text = get_phonemes(text_path)
        notes, phonemes = align(
            text, note_sequence, total_length, self.consonant_emphasis, self.min_note
        )
        return notes, phonemes

def seq2frame(
    data_seq: list,
    total_length: int,
    min_data=0
):
    expanded_data = torch.zeros(total_length)

    for data_event in data_seq:
        data, original_start, original_end = data_event
        data -= min_data
        assert data >= 0

        expanded_data[original_start:original_end] = data
    
    expanded_data = expanded_data.to(int)
    return expanded_data

def get_note_sequence(
    midi_path: str, frame_len: float, count_frames: bool = True
) -> list:
    """generate a list containing midi events of form (note, start, end)"""
    """specialized to koko dataset"""

    midi_file = mido.MidiFile(midi_path)
    tempo = 500000
    max_idx = 0
    max_num_messages = 0
    for i, track in enumerate(midi_file.tracks):
        if i == 0:
            tempo = get_tempo(track)
        else:
            num_messages = len(track)
            if num_messages > max_num_messages:
                max_num_messages = num_messages
                max_idx = i
    
    track = midi_file.tracks[max_idx]
    note_sequence = []

    current_tick = 0
    note_tick  = [ [ [] for i in range(256)] for j in range(16)]
    
    unit = tick2milisecond(tempo, midi_file.ticks_per_beat)
    for message in track:
        #pointer += message.time
        current_tick += message.time
        event = message.type
        if event == "note_on" and message.velocity != 0:
            channel = message.channel
            note     = message.note

            note_tick[channel][note].append(current_tick)
        elif event == "note_off" or (event == "note_on" and message.velocity == 0):
            channel = message.channel
            note     = message.note

            start_tick = note_tick[channel][note].pop()
            start = start_tick * unit
            end = current_tick * unit
            if count_frames:
                start = round(start / frame_len)
                end = round(end / frame_len)
            note_sequence.append((note, start, end))
    
    # sanity check
    remaining_notes = 0
    for i in range(256):
        for j in range(16):
            remaining_notes += len(note_tick[j][i])
    print("remaining: %d" % remaining_notes)
    return note_sequence

def get_label_sequence(labfile, frame_len, count_frames=True):
    sequence = []
    with open(labfile) as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            label = row[2]
            label_idx = KOKO2022_labels.index(label)
            start_time_ms = int(row[0]) / 1e4
            end_time_ms   = int(row[1]) / 1e4

            if count_frames:
                start_time = round(start_time_ms / frame_len)
                end_time   = round(end_time_ms   / frame_len)
            else:
                start_time = start_time_ms
                end_time   = end_time_ms
            sequence.append( (label_idx, start_time, end_time) )
    return sequence

def get_tempo(track) -> int:
    """find track tempo, in microseconds per beat"""
    for message in track:
        if message.type == "set_tempo":
            return message.tempo
    return 500000

def tick2milisecond(tempo: int, ticks_per_beat: int) -> float:
    """calculate how many miliseconds are in one tick"""
    return tempo / (1000 * ticks_per_beat)

def main(args):
    print(args)
    config = load_config(args.config_path)
    print(config)
    preprocessor = PreprocessorKoKo2022(config)
    result = preprocessor(args.midfile, args.labfile, args.wavfile)

    midipath = pathlib.Path(args.midfile)
    basename = midipath.stem
    savedir_path = pathlib.Path(args.savedir)
    torch.save(result, savedir_path / ("%s.pt" % basename) )
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavfile")
    parser.add_argument("--midfile")
    parser.add_argument("--labfile")
    parser.add_argument("--savedir", default="data/bin/train")
    parser.add_argument(
        "--config_path", type=str, default="configs/preprocess.json"
    )
    args = parser.parse_args()
    main(args)