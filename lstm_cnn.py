import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import pywt
from loess.loess_1d import loess_1d

import argparse
from os import listdir
from os.path import join


def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser(description='BDLSTM + CNN model trainer')

    parser.add_argument('--df_audio_save_path', help='Path to save the audio signals dataframe')
    parser.add_argument('--df_audio_read_path', help='Path to read the audio signals dataframe')

    return parser.parse_args()


def read_audio_filenames(directory: str) -> list[str]:
    '''

    '''
    return [ filename for filename in listdir(directory) if filename.endswith('.wav') ]


def create_audio_dataframe(filenames: list[str]) -> pd.DataFrame:
    '''
    '''
    parts = [ filename.split('.')[0].split('_') + [filename] for filename in filenames ]

    df = pd.DataFrame(parts, columns=['ID', 'code', 'local1', 'local2', 'device', 'filename'])
    df['ID'] = df['ID'].astype(int)

    return df


def read_audio_content(filepath: str, sample_rate=22050, offset=0, duration=5):
    '''
    '''
    data, sr = librosa.load(filepath, sr=sample_rate, offset=offset, duration=duration)
    return data


def wdenoise(audio):
    '''Wavelet denoise
    '''
    return pywt.wavedec(audio, 'db5', mode='smooth', level=4)[0]


def apply_loess(audio):
    '''
    '''
    xout, yout, wout = loess_1d(np.array(range(len(audio))), audio) # type: ignore
    return yout


if __name__ == '__main__':

    DATASET_BASEDIR = 'Respiratory_Sound_Database'
    AUDIO_TXT_DIR = join(DATASET_BASEDIR, 'audio_and_txt_files')

    # Parse program arguments
    args = parse_args()

    if args.df_audio_read_path is not None:
        df_audio_diagnosis = pd.DataFrame(
            np.load(args.df_audio_read_path, allow_pickle=True),
            columns=['ID', 'code', 'local1', 'local2', 'device', 'filename', 'Diagnosis', 'audio_data']
            )
    else:
        # Read diagnosis file
        df_diagnosis = pd.read_table('ICBHI_Challenge_diagnosis.txt', sep='\t')

        # Read list of wav filenames
        audio_filenames = read_audio_filenames(AUDIO_TXT_DIR)

        # Create dataframe for audio files
        df_audio = create_audio_dataframe(audio_filenames)

        # Join audio and diagnosis dataframes
        df_audio_diagnosis = df_audio.join(df_diagnosis, on='ID')

        # Read audio signals
        df_audio_diagnosis['audio_data'] = df_audio_diagnosis['filename']\
            .apply(lambda filename: read_audio_content(join(AUDIO_TXT_DIR, filename)))

        # Apply Wavelet denoising
        df_audio_diagnosis['audio_data'] = df_audio_diagnosis['audio_data'].apply(lambda audio: wdenoise(audio))

        # Apply LOESS
        df_audio_diagnosis['audio_data'] = df_audio_diagnosis['audio_data'].apply(lambda audio: apply_loess(audio))

    # Save partial dataframe
    if args.df_audio_save_path is not None:
        np.save(args.df_audio_save_path, df_audio_diagnosis.to_numpy())

    print(df_audio_diagnosis)