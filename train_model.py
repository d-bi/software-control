# Libraries for data import, preprocessing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.ioff()

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# OS, Control
import os

# Path properties
train_root = 'csv_sa_2/'
test_root = 'csv_sa_2/'
prefix = 'db-rec-'
label = ['GO_', 'STOP_']
train_number = range(1, 4)
# test_number = range(3, 4)
suffix = '.csv'

# Data properties
SAMPLE_FREQ = 256
INTERVAL = 0.4
N_COMPONENTS = 4 # raw data
TRAIN_SAMPLES = 60
TEST_SAMPLES = 36

# Spectrogram
N_SPECT_FREQS, N_SPECT_TIMES = 33, 39

########################################################################

# Audio
from scipy.io import wavfile
from scipy import signal

# Audio Path Properties
audio_root = 'sound_2/'
audio_prefix = ''
audio_suffix = '.wav'

# Audio Data Properties
AUDIO_SAMPLE_FREQ = 44100
AUDIO_EVENT_THRESHOLD_LO = 1
AUDIO_EVENT_THRESHOLD_HI = 1000
MIN_AUDIO_EVENT_LENGTH = 10000

########################################################################
# A note on labels: 1 is "go," 0 is "stop."

print('imports done, start reading data')

def to_eeg_index(audio_index):
    return int(audio_index / AUDIO_SAMPLE_FREQ * SAMPLE_FREQ)

END = to_eeg_index(INTERVAL * AUDIO_SAMPLE_FREQ) # Produce training samples of uniform length.

df_columns = np.empty((0, int(END * N_COMPONENTS)))
data_stack = np.empty((0, END, N_COMPONENTS))
spects_stack = np.empty((0, N_COMPONENTS, N_SPECT_FREQS, N_SPECT_TIMES)) # Spectrogram dimensions are particular.

NUM_SAMPLES = []

for l in label:
    print(l)
    event_cts = 0
    for trial in train_number:
        print('####', end=' ')
        ### READ AUDIO DATA
        audio_path = audio_root + audio_prefix + l + str(trial) + audio_suffix
        _, samples = wavfile.read(audio_path)
        envelope = np.abs(signal.hilbert(samples))

        ### GET AUDIO EVENTS
        events = np.empty((0, 2))
        in_event = False
        new_event = np.zeros((1, 2))
        hi_threshold_check = False
        for s in range(len(samples)):
            if (not in_event) and (envelope[s] >= AUDIO_EVENT_THRESHOLD_LO):
                in_event = True
                new_event[0][0] = s
            elif (in_event) and (envelope[s] < AUDIO_EVENT_THRESHOLD_LO):
                in_event = False
                new_event[0][1] = s
                event_length = new_event[0][1] - new_event[0][0]
                if hi_threshold_check and event_length >= MIN_AUDIO_EVENT_LENGTH:
                    print('*', end='') # Simplified progress bar
                    events = np.concatenate((events, new_event), axis=0)
                    hi_threshold_check = False
            elif (in_event) and (envelope[s] >= AUDIO_EVENT_THRESHOLD_HI):
                hi_threshold_check = True

        ### READ EEG DATA
        cortical_path = train_root + prefix + l + str(trial) + suffix
        df = pd.read_csv(cortical_path)


        df = df.drop(columns=df.columns[25:])
        df = df.drop(columns=df.columns[0:21])  # Only keep raw EEG data.

#         df = df.drop(columns=df.columns[21:])
#         df = df.drop(columns=df.columns[0]) # Keep only DTABG filtered components.

        ### EXTRACT EEG EVENTS FROM AUDIO EVENTS (EACH EVENT IS A DATAPOINT)
        cts = 0
        for e in events:
            cts += 1
            event_cts += 1

            START = to_eeg_index(e[0])

            raw = df.iloc[START:]
            if (raw.equals(raw.dropna()) == False):
                raw = raw.dropna()
            raw = raw.iloc[:END]
            raw_vals = raw.values
            raw_vals = np.reshape(
                raw_vals, (1, END, N_COMPONENTS)) # Split into N_COMPONENTS channels
            data_stack = np.concatenate((data_stack, raw_vals), axis=0)

            ### GENERATE SPECTROGRAMS
            spects = np.empty((0, N_SPECT_FREQS, N_SPECT_TIMES))
            for ch in range(N_COMPONENTS):
                ch_raw = raw_vals[0, :, ch]
                ch_spect, freqs_spect, times_spect, image_spect = plt.specgram(
                    ch_raw, NFFT=64, Fs=SAMPLE_FREQ, noverlap=63)
                ch_spect = np.reshape(ch_spect, (1, N_SPECT_FREQS, N_SPECT_TIMES))
                spects = np.concatenate((spects, ch_spect), axis=0) # Channels in one datapoint

            spects = np.reshape(spects, (1, N_COMPONENTS, N_SPECT_FREQS, N_SPECT_TIMES))
            spects_stack = np.concatenate((spects_stack, spects), axis=0) # The tensor corresponding to this datapoint

            ### FOR CLUSTERING PURPOSES
            df_col = np.reshape(
                raw_vals, (1, int(END * N_COMPONENTS))) # One vector of df_stack
            df_columns = np.concatenate((df_columns, df_col), axis=0)


        print('*' + str(cts), end=' ') # Progress bar

    NUM_SAMPLES = NUM_SAMPLES + [event_cts]
    print('\n')

df_stack = pd.DataFrame(df_columns) # Will only work when data is combined.
# print(df_stack.head())
print("data_stack shape: ", data_stack.shape)
print("spects_stack shape: ", spects_stack.shape)
print("confirm no NaN:", len(np.argwhere(np.isnan(data_stack))) == 0)  # Should be empty

# Labels: 1=GO, 0=STOP
data_labels = [1] * NUM_SAMPLES[0] + [0] * NUM_SAMPLES[1]
print(NUM_SAMPLES)

NUM_TRAIN_SAMPLES = [150, 150]
NUM_TEST_SAMPLES = [36, 36]

### SPECTROGRAM
train_stack = spects_stack[np.r_[
    0:NUM_TRAIN_SAMPLES[0],
    NUM_SAMPLES[0]:NUM_SAMPLES[0]+NUM_TRAIN_SAMPLES[1]], :, :, :]
test_stack = spects_stack[np.r_[
    NUM_TRAIN_SAMPLES[0]:NUM_TRAIN_SAMPLES[0]+NUM_TEST_SAMPLES[0],
    NUM_SAMPLES[0]+NUM_TRAIN_SAMPLES[1]:NUM_SAMPLES[0]+NUM_TRAIN_SAMPLES[1]+NUM_TEST_SAMPLES[1]], :, :, :]
print(train_stack.shape, test_stack.shape)
train_data_ch_split = np.split(train_stack, N_COMPONENTS, axis=1)
test_data_ch_split = np.split(test_stack, N_COMPONENTS, axis=1)
print(len(train_data_ch_split), train_data_ch_split[0].shape)
print(len(test_data_ch_split), test_data_ch_split[0].shape)

### CONVNET
train_data_spect = np.swapaxes(train_stack, 1, 3)
test_data_spect = np.swapaxes(test_stack, 1, 3)
print(train_data_spect.shape, test_data_spect.shape)

# Labels: 1=GO, 0=STOP
train_labels = np.asarray([1] * NUM_TRAIN_SAMPLES[0] + [0] * NUM_TRAIN_SAMPLES[1])
test_labels = np.asarray([1] * NUM_TEST_SAMPLES[0] + [0] * NUM_TEST_SAMPLES[1])

print(train_labels.shape, test_labels.shape)

print('done reading data, start building model')

sg_ch_spectrogram_shape = (1, N_SPECT_FREQS, N_SPECT_TIMES)
LSTM_IN_SHAPE = (N_SPECT_TIMES, 1)
LSTM_OUT_DIM = N_SPECT_TIMES

spects_inputs, recurr_combined = [], []
for ch in range(N_COMPONENTS):
    # spectrogram from each EEG channel (4 spectrograms)
    spects_inputs = spects_inputs + [keras.Input(shape=sg_ch_spectrogram_shape)]
    recurr_this_channel = []
    for freq in range(N_SPECT_FREQS):
        slice_freq = layers.Lambda(lambda x: x[:, 0, freq, :])(spects_inputs[ch]) # slice to (N_SPECT_TIMES,)
        reshape_for_recurr = layers.Reshape(LSTM_IN_SHAPE)(slice_freq)
        # multi-channel recurrent net (transformer?)
        recurr_this_channel = recurr_this_channel + [layers.LSTM(LSTM_OUT_DIM)(reshape_for_recurr)]
        print('*', end='') # progress bar
    # concat recurrent layers
    recurr_combined = recurr_combined + [layers.Lambda(
        lambda x: tf.stack(x, axis=2))(
            recurr_this_channel)]
    print('#', recurr_combined[ch].shape)
# feature image with 4 "spatial channels" (2d * 4)
transformed_spect_combined = layers.Lambda(
        lambda x: tf.stack(x, axis=3))(
                recurr_combined)
print(transformed_spect_combined.shape)
# ConvNet
first_conv = layers.Conv2D(16, (3, 3), activation='relu')(transformed_spect_combined)
first_pool = layers.MaxPooling2D((2, 2))(first_conv)
second_conv = layers.Conv2D(32, (3, 3), activation='relu')(first_pool)
second_pool = layers.MaxPooling2D((2, 2))(second_conv)
third_conv = layers.Conv2D(64, (3, 3), activation='relu')(second_pool)
flatten = layers.Flatten()(third_conv)
first_dense = layers.Dense(64, activation='relu')(flatten)
output = layers.Dense(1)(first_dense)

model = keras.Model(inputs=spects_inputs, outputs=output)


model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('done building model, start training')

history = model.fit(train_data_ch_split, train_labels, batch_size=8, epochs=20, validation_split=0.2, verbose=1)

model.save('models/SpectRecurrConvNet')

