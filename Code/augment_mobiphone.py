# This script augments the MOBIPHONE dataset by adding gaussian noise, background chattering noise and reverberation
# to each wav file of the dataset, effectively quadrupling the dataset

import os
import torch
import torchaudio
import torchaudio.functional as F

# phone models
phones = ["HTC desire c", "HTC sensation xe", "LG GS290", "LG L3", "LG Optimus L5", "LG Optimus L9", "Nokia 5530", "Nokia C5", "Nokia N70", "Samsung E2121B", \
          "Samsung E2600", "Samsung GT-I8190 mini", "Samsung GT-N7100 (galaxy note2)", "Samsung Galaxy GT-I9100 s2", "Samsung Galaxy Nexus S", "Samsung e1230", \
          "Samsung s5830i", "Sony Ericson c902", "Sony ericson c510i", "Vodafone joy 845", "iPhone5"]

# make the folders where the augmented wav files will be stored
if(not os.path.isdir("MOBIPHONE_GAUSSIAN")):
    os.mkdir("MOBIPHONE_GAUSSIAN")
    for phone in phones:
        os.mkdir("MOBIPHONE_GAUSSIAN/" + phone)

if(not os.path.isdir("MOBIPHONE_BACKGROUND")):
    os.mkdir("MOBIPHONE_BACKGROUND")
    for phone in phones:
        os.mkdir("MOBIPHONE_BACKGROUND/" + phone)

if(not os.path.isdir("MOBIPHONE_REVERBERATION")):
    os.mkdir("MOBIPHONE_REVERBERATION")
    for phone in phones:
        os.mkdir("MOBIPHONE_REVERBERATION/" + phone)

# use Room Impulse Response (RIR) to make speech sound as though it has been uttered in a conference room
rir_raw, sample_rate = torchaudio.load("noise files/clap.wav") # from VOiCES dataset
# clean up the RIR, extract the main impulse (between 1.01 seconds to 1.3 seconds) and normalize it by its power
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.linalg.vector_norm(rir, ord=2)

# background noise (royalty free sound effect from https://pixabay.com/)
background_noise, _ = torchaudio.load("noise files/chattering.wav")

for phone in phones:
    for i in range(1, 25):
        waveform, sample_rate = torchaudio.load("MOBIPHONE/" + phone + "/speaker" + str(i) + ".wav")
        # keep only one channel
        if(waveform.shape[0] > 1):
            # unsqueeze is needed to convert tensor to shape [1, num_samples] instead of [num_samples]
            waveform = waveform[1, :].unsqueeze(0)

        ### add gaussian noise ###
        # random SNR between 10 and 20 db
        target_snr_db = torch.randint(10, 21, (1, 1)).item()
        signal_power = torch.mean(torch.pow(waveform, 2))
        # calculate power for gaussian noise to achieve target SNRdb
        noise_power = signal_power / (10 ** (target_snr_db / 10.0))
        gaussian = waveform + torch.normal(mean=0, std=torch.sqrt(noise_power), size=waveform.shape)
        torchaudio.save("MOBIPHONE_GAUSSIAN/" + phone + "/speaker" + str(i) + ".wav", gaussian, sample_rate=sample_rate)

        ### add background noise ###
        # random SNR between 10 and 20 db
        target_snr_db = torch.randint(10, 21, (1, ))
        # keep background noise with equal duration to the audio signal
        b = background_noise[:, :waveform.shape[1]]
        background_power = torch.mean(torch.pow(b, 2))
        # scaling factor of background noise to achieve target SNRdb
        a = torch.sqrt((signal_power / 10 ** (target_snr_db / 10)) / background_power)
        background = waveform + a * b
        torchaudio.save("MOBIPHONE_BACKGROUND/" + phone + "/speaker" + str(i) + ".wav", background, sample_rate=sample_rate)

        ### add reverberation ###
        reverberation = F.fftconvolve(waveform, rir)
        torchaudio.save("MOBIPHONE_REVERBERATION/" + phone + "/speaker" + str(i) + ".wav", reverberation, sample_rate=sample_rate)