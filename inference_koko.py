import sys

sys.path.append("hifi-gan")

import argparse
import os
import subprocess
import pathlib

import numpy as np
import torch

from data.preprocess import Preprocessor
from utils import load_config, load_model, make_directory, set_seed


@torch.no_grad()
def main(args):
    # setup model
    device = args.device
    assert device in {"cpu", "cuda"}, "device must be one of `cpu` or `cuda`"
    model = load_model(args.checkpoint_path, eval=True)
    preprocessor_config = load_config(args.preprocessor_config_path)
    mel_dim = preprocessor_config.n_mel_channels

    # load pt data
    notes, phonemes, mel = torch.load(args.input)
    chunk_size = model.seq_len
    preds = []
    total_len = len(notes)
    notes = notes.to(device)
    phonemes = phonemes.to(device)
    remainder = total_len % chunk_size
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
    np.save(args.output, preds.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/default/last.pt",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--preprocessor_config_path",
        type=str,
        default=os.path.join("configs", "preprocess.json"),
        help="path to preprocessor config file",
    )
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    set_seed()
    main(args)
