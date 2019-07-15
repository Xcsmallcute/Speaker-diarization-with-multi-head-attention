# Speaker diarization with multi-head attention
## Introduction
Speaker verification (SV) is the process of verifying whether an utterance
belongs to a specific speake, based on that speaker’s known
utterances (i.e., enrollment utterances), with applications such as
Voice Match.<br>
Depending on the restrictions of the utterances used for enrollment
and verification, speaker verification models usually fall into
one of two categories: `text-dependent speaker verification (TD-SV)`
and `text-independent speaker verification (TI-SV)`. In TD-SV, the
transcript of both enrollment and verification utterances is phonetially
constrained, while in TI-SV, there are no lexicon constraints
on the transcript of the enrollment or verification utterances, exposing
a larger variability of phonemes and utterance durations.
In this work, we focus on TI-SV situation and `compare different attention-based model` 
for TI-SV speaker verification.

## Requirements
![](https://img.shields.io/badge/python-v3.6.6-blue.svg)<br>
![](https://img.shields.io/badge/tensorflow-v1.12.0-orange.svg)<br>
![](https://img.shields.io/badge/librosa-v0.6.3-lightgrey.svg)<br>
![](https://img.shields.io/badge/CUDA-v9.0.176-green.svg)

## Baseline model
<div align=center>
<img src="https://gitlab-bpit.huawei.com/xiaocheng/speaker-diarization-with-multi-head-attention/raw/master/images/basemodel.PNG"/>
</div>

<div align="center">
Fig.1 Baseline model
</div>
&nbsp;

### Preprocessing
For each utterance, firstly use VAD to remove slience and low dB noise,
then transformed these utterances to `log-mel-filterbank energies`.<br>
```Python
# voice activity detection
intervals = librosa.effects.split(utter, top_db=20)   

# log mel spectrogram of utterances
S = librosa.core.stft(y=utter_part, n_fft=config.nfft,win_length=int(config.window * sr), hop_length=int(config.hop * sr))
S = np.abs(S) ** 2
mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
S = np.log10(np.dot(mel_basis, S) + 1e-6)           
```
For each training step, a N x M batch is fed into our LSTM network, 
where N represents the number of speakers, and M represents number
of utterances of each speaker, every utterance has been transformed into
log-mel-filterbank energies.<br>
### Loss function
As for loss function, we use [GE2E loss function](https://arxiv.org/abs/1710.10467),
which is proposed by google in 2017.<br>

<div align=center>
<img src="https://gitlab-bpit.huawei.com/xiaocheng/speaker-diarization-with-multi-head-attention/raw/master/images/d-vector.PNG"/>
</div>

<div align="center">
Fig.2 Extracting embedding vector
</div>
&nbsp;

For each utterance, let the output of the LSTM’s last layer at
frame t be a fixed dimensional vector ht, where 1 ≤ t ≤ T. We
take the last frame output as the d-vector ω = hT.

<div align=center>
<img src="https://gitlab-bpit.huawei.com/xiaocheng/speaker-diarization-with-multi-head-attention/raw/master/images/loss1.PNG"/>
</div>
The centroid represents the discrimitive feature of a speaker.
<div align=center>
<img src="https://gitlab-bpit.huawei.com/xiaocheng/speaker-diarization-with-multi-head-attention/raw/master/images/loss2.PNG"/>
</div>
The similarity is defined using the cosine similarity function.
<div align=center>
<img src="https://gitlab-bpit.huawei.com/xiaocheng/speaker-diarization-with-multi-head-attention/raw/master/images/loss3.PNG"/>
</div>
With learnable w and b, the loss function is defined as above.

## Multi-head attention
<div align=center>
<img src="https://gitlab-bpit.huawei.com/xiaocheng/speaker-diarization-with-multi-head-attention/raw/master/images/mulhead.PNG" width="500" height="600"/>
</div>

<div align="center">
Fig.3 An example for multi-head attention
</div>
&nbsp;
