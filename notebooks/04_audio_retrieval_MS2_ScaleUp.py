import os
import time

import numpy as np
import librosa
from pandas import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
    """Compute constellation map from a spectrogram Y."""
    result = ndimage.maximum_filter(Y, size=[2 * dist_freq + 1, 2 * dist_time + 1], mode='constant')
    Cmap = np.logical_and(Y == result, result > thresh)
    return Cmap

def compute_spectrogram(fn_wav, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    """Compute spectrogram for a given audio file"""
    x, Fs = librosa.load(fn_wav, sr=Fs)
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    if bin_max is None:
        bin_max = X.shape[0]
    if frame_max is None:
        frame_max = X.shape[0]
    Y = np.abs(X[:bin_max, :frame_max])
    return Y


def get_files_from_tarballs_range(start, end, directory='../data', endswith='.mp3'):
    """Function to get all files with the specified extension in a directory and its subdirectories."""
    files = []
    print(f"Getting files from tarballs {start} to inclusive {end}")
    for i in range(start, end + 1):
        tarball = f"{i:02d}"
        tarball_files = get_files(directory=f'{directory}/{tarball}', endswith=endswith)
        files.extend(tarball_files)
    return files


from pympler import asizeof
import pickle
import os
import pympler


def save_db_hash_index(db_hash_index, file_path='db_hash_index.pkl'):
    """Saves the db_hash_index in a file"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        pickle.dump(db_hash_index, f)
    print(f"saved db_hash_index in {file_path}")


def load_db_hash_index(file_path='db_hash_index.pkl'):
    """Loads the db_hash_index from a file."""
    with open(file_path, 'rb') as f:
        db_hash_index = pickle.load(f)
    print(f"loaded db_hash_index from {file_path}")
    return db_hash_index

def compute_hash(freq1, freq2, delta_time, dist_freq, dist_time, exclusion_zone):
    if (
            delta_time > exclusion_zone and  # Exclude points within exclusion zone
            freq1 + dist_freq / 2 > freq2 > freq1 - dist_freq / 2 and
            freq2 >= 0 and
            exclusion_zone <= delta_time <= dist_time + exclusion_zone
    ):
        # Compute 32-bit hash as described in Wang's paper
        hash_value = (
                (freq1 & 0x3FF) << 22 |  # 10 bits for freq1
                (freq2 & 0x3FF) << 12 |  # 10 bits for freq2
                (delta_time & 0xFFF)  # 12 bits for delta_time
        )
        return hash_value
    return None

import sys


def get_sizeof(obj):
    """Calculates the size of an object recursively."""
    return asizeof.asizeof(obj)


def prepare_hash_index_parallel(file_list, configuration, target_zone, exclusion_zone=5, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    """
    Process files in parallel to create a hash index without storing the constellation maps.

    Parameters:
    - file_list (list): List of file paths to process.
    - configuration (tuple): Configuration for constellation map (dist_freq, dist_time).
    - target_zone (tuple): Target zone for hash computation (dist_freq, dist_time).
    - exclusion_zone (int): Exclusion zone for anchor points.
    - Fs, N, H, bin_max, frame_max: Parameters for spectrogram computation.

    Returns:
    - db_hash_index (dict): Hash index with {hash_value: [(file_name, time1), ...]}.
    """
    db_hash_index = defaultdict(list)
    print(f"Processing {len(file_list)} files in parallel...")
    progress_counter = 0
    total_files = len(file_list)

    def process_file(file):
        """Process a single file to compute hashes."""
        nonlocal progress_counter
        try:
            # Compute spectrogram
            Y = compute_spectrogram(file, Fs=Fs, N=N, H=H, bin_max=bin_max, frame_max=frame_max)
            file_key = re.split(r'[\\/]', file)[-1]
            dist_freq, dist_time = configuration

            # Compute constellation map
            cmap = compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time)

            # Compute hashes from constellation map
            freq_bins, time_bins = np.where(cmap == 1)  # Anchor points
            local_hash_index = defaultdict(list)

            for i, (freq1, time1) in enumerate(zip(freq_bins, time_bins)):
                for j in range(i + 1, len(freq_bins)):
                    freq2 = freq_bins[j]
                    time2 = time_bins[j]

                    delta_time = time2 - time1
                    hash_value = compute_hash(freq1, freq2, delta_time, target_zone[0], target_zone[1], exclusion_zone)
                    if hash_value is not None:
                        local_hash_index[hash_value].append((file_key, time1))
            progress_counter += 1 # it could sometimes be the case that concurrent processing does not increment the counter correctly, but progress_counter is just for the user to see how far the processing is and if the results are off by a few files it is not a problem
            if progress_counter % 250 == 0:
                print(f"Processed {progress_counter}/{total_files} files...")

            return local_hash_index
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return {}

    # Parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, file_list))

    # Merge all local hash indices into the global index
    for local_index in results:
        for hash_value, entries in local_index.items():
            db_hash_index[hash_value].extend(entries)

    print("Processing done.")
    return db_hash_index


def scaleUp(db_hash_index, start, end, target_zone, exclusion_zone=5):
    results = {}

    new_database_files = get_files_from_tarballs_range(start, end)

    start_time = time.perf_counter()
    new_db_hash_index = prepare_hash_index_parallel(new_database_files, configuration, target_zone, exclusion_zone)
    end_time = time.perf_counter()
    results["time_to_compute_cmaps_and_hashes"] = end_time - start_time
    results["number_of_new_files"] = len(new_database_files)

    db_hash_index.update(new_db_hash_index)
    results["size_of_db_hash_index"] = get_sizeof(db_hash_index)
    save_db_hash_index(db_hash_index, f'hashes/db_hash_index_to_tarball_{end}.pkl')
    results["size_of_db_hash_index_on_disk"] = os.path.getsize(f'hashes/db_hash_index_tz0_{end}.pkl')

    return db_hash_index, results


def save_results_to_table_scale_up(results):
    """
    Convert scale-up results into a table for analysis.

    Parameters:
    - results (dict): Dictionary containing scale-up results.

    Returns:
    - summary_df (pd.DataFrame): Dataframe summarizing the results for each scale-up step.
    """
    summary_data = []
    for step, result in results.items():
        summary_data.append({
            "Step": step,
            "Time to Compute Constellation Maps and Hashes (s)": result["time_to_compute_cmaps_and_hashes"],
            "Number of New Files": result["number_of_new_files"],
            "Size of DB Hash Index (MB)": result["size_of_db_hash_index"] / (1024 ** 2),
            "Size of DB Hash Index on Disk (MB)": result["size_of_db_hash_index_on_disk"] / (1024 ** 2)
        })
    summary_df = DataFrame(summary_data)
    return summary_df

configuration = (8, 2) # (κ=8, τ=2)
(dist_freq, dist_time) = configuration
target_zones = [(5,10), (10, 5), (10, 10), (5, 5)]



index_scale_up, result = scaleUp({}, 0, 9, target_zones[0])
print(save_results_to_table_scale_up(result))

