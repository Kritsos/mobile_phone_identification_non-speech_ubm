# the following script augments the MOBIPHONE database by performing 5 augmentations on it
# 1. Random croppings in the middle of the signal
# 2. Randomly increase or decrease the loudness of the signal (from half to double loudness level)
# 3. Randomly change the pitch of the speech (with a factor from -10 to 10)
# 4. Randomly change the speed of the signal (from half speed to double speed)
# 5. Randomly apply vocal tract length perturbation with a factor from 0.8 to 1.2

import os
import numpy as np
import nlpaug.augmenter.audio as naa
from scipy.io import wavfile

# phone models
phones = ["HTC desire c", "HTC sensation xe", "LG GS290", "LG L3", "LG Optimus L5", "LG Optimus L9", "Nokia 5530", "Nokia C5", "Nokia N70", "Samsung E2121B", \
          "Samsung E2600", "Samsung GT-I8190 mini", "Samsung GT-N7100 (galaxy note2)", "Samsung Galaxy GT-I9100 s2", "Samsung Galaxy Nexus S", "Samsung e1230", \
          "Samsung s5830i", "Sony Ericson c902", "Sony ericson c510i", "Vodafone joy 845", "iPhone5"]

# make the folders where the augmented wav files will be stored
if(not os.path.isdir("MOBIPHONE_CROP")):
    os.mkdir("MOBIPHONE_CROP")
    for phone in phones:
        os.mkdir("MOBIPHONE_CROP/" + phone)

if(not os.path.isdir("MOBIPHONE_LOUDNESS")):
    os.mkdir("MOBIPHONE_LOUDNESS")
    for phone in phones:
        os.mkdir("MOBIPHONE_LOUDNESS/" + phone)

if(not os.path.isdir("MOBIPHONE_PITCH")):
    os.mkdir("MOBIPHONE_PITCH")
    for phone in phones:
        os.mkdir("MOBIPHONE_PITCH/" + phone)

if(not os.path.isdir("MOBIPHONE_SPEED")):
    os.mkdir("MOBIPHONE_SPEED")
    for phone in phones:
        os.mkdir("MOBIPHONE_SPEED/" + phone)

if(not os.path.isdir("MOBIPHONE_VTLP")):
    os.mkdir("MOBIPHONE_VTLP")
    for phone in phones:
        os.mkdir("MOBIPHONE_VTLP/" + phone)

# randomly pick a zone in which the augmentation will take place
# for example if the zone is (0.2, 0.8) then the augmentation will 
# take place between the 20% and 80% of the samples of the signal
def pick_zone():
    zone_start_a = 0.05
    zone_start_b = 0.3
    zone_start = (zone_start_b - zone_start_a) * np.random.random() + zone_start_a

    zone_end_a = 0.7
    zone_end_b = 0.95
    zone_end = (zone_end_b - zone_end_a) * np.random.random() + zone_end_a

    return (zone_start, zone_end)

# randomly pick the percent of the selected zone on which the augmentation
# will take place
def pick_coverage():
    a = 0.3
    b = 0.7
    return (b - a) * np.random.random() + a

for phone in phones:
    for i in range(1, 25):
        sample_rate, waveform = wavfile.read("MOBIPHONE/" + phone + "/speaker" + str(i) + ".wav")
        # keep only one channel if there is more than one
        if(waveform.ndim > 1):
            waveform = waveform[:, 1]

        # CROP
        aug = naa.CropAug(sampling_rate=sample_rate, zone=pick_zone(), coverage=pick_coverage())
        waveform_crop = np.array(aug.augment(waveform)).reshape(-1)
        wavfile.write("MOBIPHONE_CROP/" + phone + "/speaker" + str(i) + ".wav", sample_rate, waveform_crop)

        # LOUDNESS
        aug = naa.LoudnessAug(zone=pick_zone(), coverage=1, factor=(0.5, 2))
        waveform_loudness = np.array(aug.augment(waveform)).reshape(-1)
        wavfile.write("MOBIPHONE_LOUDNESS/" + phone + "/speaker" + str(i) + ".wav", sample_rate, waveform_loudness)

        # PITCH
        aug = naa.PitchAug(sampling_rate=sample_rate, zone=pick_zone(), coverage=1, factor=(-10, 10))
        # this augmentation needs the data in float form so we convert it into the [-1, 1] region
        # and cast it back to an int16 after the augmentation
        waveform_pitch = np.array(aug.augment(waveform / 32767.0)).reshape(-1)
        waveform_pitch = (waveform_pitch * 32767).astype(np.int16)
        wavfile.write("MOBIPHONE_PITCH/" + phone + "/speaker" + str(i) + ".wav", sample_rate, waveform_pitch)

        # SPEED
        aug = naa.SpeedAug(zone=pick_zone(), coverage=1, factor=(0.5, 2))
        # this augmentation needs the data in float form so we convert it into the [-1, 1] region
        # and cast it back to an int16 after the augmentation
        waveform_speed = np.array(aug.augment(waveform / 32767.0)).reshape(-1)
        waveform_speed = (waveform_speed * 32767).astype(np.int16)
        wavfile.write("MOBIPHONE_SPEED/" + phone + "/speaker" + str(i) + ".wav", sample_rate, waveform_speed)

        # VLTP
        aug = naa.VtlpAug(sampling_rate=sample_rate, zone=pick_zone(), coverage=pick_coverage(), fhi=5000, factor=(0.8, 1.2))
        # this augmentation needs the data in float form so we convert it into the [-1, 1] region
        # and cast it back to an int16 after the augmentation
        waveform_vtlp = np.array(aug.augment(waveform / 32767.0)).reshape(-1)
        waveform_vtlp = (waveform_vtlp * 32767).astype(np.int16)
        wavfile.write("MOBIPHONE_VTLP/" + phone + "/speaker" + str(i) + ".wav", sample_rate, waveform_vtlp)