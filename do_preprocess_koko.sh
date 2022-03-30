#! /bin/bash

wavdir=../no7singing/wav_PT_22k
middir=../no7singing/midi_label
labdir=../no7singing_labels/mono_label

for idx in $(seq 1 51); do
#for idx in "5"; do
    base=$(printf "%02d" $idx)
    wavfile=$wavdir/$base.wav
    midfile=$middir/$base.mid
    labfile=$labdir/$base.lab

    python3 preprocess_koko.py --wavfile $wavfile --midfile=$midfile --labfile=$labfile
done