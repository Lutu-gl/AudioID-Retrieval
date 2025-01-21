import hashlib
import time
import re
import os
import numpy as np
import librosa
from scipy import ndimage


def get_files(directory='../data', endswith='.wav'):
    """Function to get all files with the specified extension in a directory and its subdirectories."""
    files = []
    for root, _, fs in os.walk(directory):
        for f in fs:
            if f.endswith(endswith):
                files.append(os.path.join(root, f))
    return files

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

def process_file(file, configuration, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    # Compute spectrogram for the current file
    Y = compute_spectrogram(file, Fs=Fs, N=N, H=H, bin_max=bin_max, frame_max=frame_max)

    # Initialize nested structure for the current file
    #file_key = os.path.basename(file)
    file_key = re.split(r'[\\/]', file)[-1]
    (dist_freq, dist_time) = configuration

    # Compute constellation maps
    cmap = compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time)

    return file_key, cmap


def prepare_database_parallel(file_list, configuration, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    database = {}
    print(f"Processing {len(file_list)} files in parallel...")

    # Parallelisierung mit ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda file: process_file(file, configuration, Fs, N, H, bin_max, frame_max), file_list
            )
        )

    for file_key, data in results:
        database[file_key] = data

    print("Processing done.")
    return database


def prepare_query_constellation_maps(file_list, configuration, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    """Compute query constellation maps for all configurations and save the results in a nested structure."""
    query_maps = {}
    for file in file_list:

        # Compute spectrogram for the current file
        Y = compute_spectrogram(file, Fs=Fs, N=N, H=H, bin_max=bin_max, frame_max=frame_max)

        # Retrieve metadata for the file
        file_size = os.path.getsize(file)  # Size in bytes

        # Initialize nested structure for the current query
        file_key = os.path.basename(file)
        query_maps[file_key] = {
            "metadata": {
                "size": file_size,
                "original_query_id": file_key.split('_')[0],
            },
            "constellation_map": {}
        }

        # Compute constellation maps for all configurations
        (dist_freq, dist_time) = configuration
        cmap = compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time)
        query_maps[file_key]["constellation_map"] = cmap
    return query_maps

def compute_hashes_list_and_index_parallel(files_dict, target_zone, exclusion_zone=5):
    """
    Compute hashes for all files' constellation maps and return a list of hashes
    along with an index for fast lookup.

    Parameters:
    - files_dict (dict): Dictionary of files, each containing a "constellation_map" entry.
                         Example: {"file1.wav": {"constellation_map": <numpy_array>}, ...}.
    - target_zone (tuple): Tuple (dist_freq, dist_time) defining the rectangle of the target zone.
                           dist_freq: Range of frequency bins to consider.
                           dist_time: Range of time frames to consider.
    - exclusion_zone (int): Minimum time frames to skip after the anchor point.

    Returns:
    - hashes_list (list): List of all hashes across all files.
                          Each entry is a tuple (hash_value, track_id, anchor_time).
    - db_hash_index (dict): Dictionary for fast lookup of database hashes as:
                            {hash_value: [(track_id, anchor_time), ...]}.
    """

    def file_processing_for_hashes(file_name, constellation_map, tz, ez):
        hashes = []
        df, dt = tz
        freq_bins, time_bins = np.where(constellation_map == 1)  # Anchor points

        for i, (freq1, time1) in enumerate(zip(freq_bins, time_bins)):
            for j in range(i + 1, len(freq_bins)):
                freq2 = freq_bins[j]
                time2 = time_bins[j]
                delta_time = time2 - time1

                hs = compute_hash(freq1, freq2, delta_time, df, dt, ez)
                if hs is not None:
                    hashes.append((hs, file_name, time1))

        return hashes

    # hashes_list = []
    db_hash_index = defaultdict(list)  # To store the index for fast lookups
    finished_hashes = 0
    print(f"Computing {len(files_dict.items())} hashes...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(file_processing_for_hashes, file_name, constellation_map, target_zone, exclusion_zone): file_name
            for file_name, constellation_map in files_dict.items()
        }

        for future in as_completed(futures):
            file_name = futures[future]
            finished_hashes += 1
            print(f"finished hashing {file_name} {finished_hashes}")
            try:
                file_hashes = future.result()
                for hash_value, file_name, anchor_time in file_hashes:
                    db_hash_index[hash_value].append((file_name, anchor_time))
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return db_hash_index


def compute_hashes_list_and_index(files_dict, target_zone, exclusion_zone=5):
    """
    Compute hashes for all files' constellation maps and return a list of hashes
    along with an index for fast lookup.

    Parameters:
    - files_dict (dict): Dictionary of files, each containing a "constellation_map" entry.
                         Example: {"file1.wav": {"constellation_map": <numpy_array>}, ...}.
    - target_zone (tuple): Tuple (dist_freq, dist_time) defining the rectangle of the target zone.
                           dist_freq: Range of frequency bins to consider.
                           dist_time: Range of time frames to consider.
    - exclusion_zone (int): Minimum time frames to skip after the anchor point.

    Returns:
    - hashes_list (list): List of all hashes across all files.
                          Each entry is a tuple (hash_value, track_id, anchor_time).
    - db_hash_index (dict): Dictionary for fast lookup of database hashes as:
                            {hash_value: [(track_id, anchor_time), ...]}.
    """
    # hashes_list = []
    db_hash_index = defaultdict(list)  # To store the index for fast lookups
    dist_freq, dist_time = target_zone

    finished_hashes = 0
    for file_name, constellation_map in files_dict.items():
        freq_bins, time_bins = np.where(constellation_map == 1)  # Anchor points

        for i, (freq1, time1) in enumerate(zip(freq_bins, time_bins)):
            for j in range(i + 1, len(freq_bins)):
                freq2 = freq_bins[j]
                time2 = time_bins[j]

                delta_time = time2 - time1

                hash_value = compute_hash(freq1, freq2, delta_time, dist_freq, dist_time, exclusion_zone)
                if hash_value is not None:
                    db_hash_index[hash_value].append((file_name, time1))
        finished_hashes += 1
        print(f"finished hashing {file_name} {finished_hashes}")

    return db_hash_index


def compute_hash(freq1, freq2, delta_time, dist_freq, dist_time, exclusion_zone):
    if (
            delta_time > exclusion_zone and  # Exclude points within exclusion zone
            0 <= (freq2 - freq1) and  # Frequency constraint
            freq1 + dist_freq / 2 > freq2 > freq1 - dist_freq / 2 and
            exclusion_zone <= delta_time <= dist_time + exclusion_zone  # Time constraint
    ):
        # Compute 32-bit hash as described in Wang's paper
        hash_value = (
                (freq1 & 0x3FF) << 22 |  # 10 bits for freq1
                (freq2 & 0x3FF) << 12 |  # 10 bits for freq2
                (delta_time & 0xFFF)  # 12 bits for delta_time
        )
        return hash_value
    return None

def compute_hashes_for_query(constellation_map, target_zone, exclusion_zone=5):
    """
    Compute hashes for a single query's constellation map.

    Parameters:
    - constellation_map (numpy array): Binary constellation map of the query.
                                        Example: <numpy_array> with 1s at anchor points.
    - target_zone (tuple): Tuple (dist_freq, dist_time) defining the rectangle of the target zone.
                           dist_freq: Range of frequency bins to consider.
                           dist_time: Range of time frames to consider.
    - exclusion_zone (int): Minimum time frames to skip after the anchor point.

    Returns:
    - hashes_list (list): List of hashes for the query.
                          Each entry is a tuple (hash_value, anchor_time).
    """
    hashes_list = []
    dist_freq, dist_time = target_zone

    # Extract anchor points from the constellation map
    freq_bins, time_bins = np.where(constellation_map == 1)

    for i, (freq1, time1) in enumerate(zip(freq_bins, time_bins)):
        # Define the target zone relative to the anchor point
        for j in range(i + 1, len(freq_bins)):
            freq2 = freq_bins[j]
            time2 = time_bins[j]
            delta_time = time2 - time1
            hash_value = compute_hash(freq1, freq2, delta_time, dist_freq, dist_time, exclusion_zone)
            if hash_value is not None:
                hashes_list.append(hash_value)

    return hashes_list


def compute_minhash(hashes_list, num_minhashes=100):
    """
    Compute MinHash signatures for a list of hashes.

    Parameters:
    - hashes_list (list): List of all hashes across all files.
                          Each entry is a tuple (hash_value, track_id, anchor_time).
                          Example: [(hash_value, track_id, anchor_time), ...].
    - num_minhashes (int): Number of MinHash signatures to compute.

    Returns:
    - minhash_dict (dict): Dictionary of MinHash signatures for each track.
                           Example: {track_id: [minhashes]}.
    """
    # Initialize MinHash dictionary
    minhash_dict = {}

    # Group hashes by track_id
    track_hashes = {}
    for hash_value, track_id, anchor_time in hashes_list:
        if track_id not in track_hashes:
            track_hashes[track_id] = []
        track_hashes[track_id].append(hash_value)

    # Compute MinHash signatures for each track
    for track_id, hashes in track_hashes.items():
        minhashes = [float('inf')] * num_minhashes
        for h in hashes:
            for i in range(num_minhashes):
                # Use a hash function to generate permutations (simulated by hashlib)
                hash_func = hashlib.md5(f"{h}_{i}".encode()).hexdigest()
                permuted_value = int(hash_func, 16)
                if permuted_value < minhashes[i]:
                    minhashes[i] = permuted_value
        minhash_dict[track_id] = minhashes

    return minhash_dict

def match_query_to_database(query_hash_list, db_hash_index, threshold=5):
    """
    Match a query against the precomputed database index and return the top-scoring track.

    Parameters:
    - query_hash_list (list): List of hashes for the query.
                              Each entry is a tuple (hash_value, anchor_time).
    - db_hash_index (dict): Precomputed hash index for the database as:
                            {hash_value: [(track_id, db_time), ...]}.
    - threshold (int): Minimum number of matching hashes in a time bin to consider it a match.

    Returns:
    - top_match (dict): Dictionary containing the best match as:
                        {"track_id": track_id, "score": match_score, "delta_t_histogram": histogram}.
    """
    # Step 1: Initialize a structure to store time differences for each track
    time_pair_bins = defaultdict(lambda: defaultdict(int))  # {track_id: {delta_t: count}}

    # Step 2: Match query hashes against the database index
    for query_hash, query_time in query_hash_list:
        if query_hash in db_hash_index:
            for db_track_id, db_time in db_hash_index[query_hash]:
                # Compute delta_t (time offset between query and database hash)
                delta_t = db_time - query_time
                # Increment the count for this delta_t and track_id
                time_pair_bins[db_track_id][delta_t] += 1

    # Step 3: Calculate scores for each track based on the maximum delta_t bin count
    track_scores = {}
    for track_id, delta_t_histogram in time_pair_bins.items():
        max_count = max(delta_t_histogram.values())  # Find the highest peak in the histogram
        track_scores[track_id] = {"score": max_count, "delta_t_histogram": delta_t_histogram}

    # Step 4: Identify the top-scoring track
    if track_scores:
        best_track_id = max(track_scores, key=lambda k: track_scores[k]["score"])
        best_score = track_scores[best_track_id]["score"]

        # Return the best match if it meets the threshold
        if best_score >= threshold:
            return {
                "track_id": best_track_id,
                "score": best_score,
                "delta_t_histogram": track_scores[best_track_id]["delta_t_histogram"],
            }
        else:
            return {"track_id": None, "score": 0, "delta_t_histogram": None}
    else:
        return {"track_id": None, "score": 0, "delta_t_histogram": None}


from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

def match_query_hash_executor(query_hash, query_time, db_hash_index):
    """
    Matches a single query hash against the database and computes delta_t for matching hashes.
    """
    results = defaultdict(int)
    if query_hash in db_hash_index:
        for db_track_id, db_time in db_hash_index[query_hash]:
            delta_t = db_time - query_time
            results[(db_track_id, delta_t)] += 1
    return results

def merge_results_executor(results_list):
    """
    Merges results from all parallel executions into a single dictionary.
    """
    merged = defaultdict(lambda: defaultdict(int))
    for results in results_list:
        for (track_id, delta_t), count in results.items():
            merged[track_id][delta_t] += count
    return merged

def match_query_to_database_threaded(query_hash_list, db_hash_index, threshold=5, max_workers=4):
    """
    Parallel version of the matching function using ThreadPoolExecutor.
    """
    results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {
            executor.submit(match_query_hash_executor, query_hash, query_time, db_hash_index): (query_hash, query_time)
            for query_hash, query_time in query_hash_list
        }

        for future in as_completed(future_to_query):
            results.append(future.result())

    # Merge results from all threads
    time_pair_bins = merge_results_executor(results)

    # Calculate scores for each track based on the maximum delta_t bin count
    track_scores = {}
    for track_id, delta_t_histogram in time_pair_bins.items():
        max_count = max(delta_t_histogram.values())  # Find the highest peak in the histogram
        track_scores[track_id] = {"score": max_count, "delta_t_histogram": delta_t_histogram}

    # Identify the top-scoring track
    if track_scores:
        best_track_id = max(track_scores, key=lambda k: track_scores[k]["score"])
        best_score = track_scores[best_track_id]["score"]

        if best_score >= threshold:
            return {
                "track_id": best_track_id,
                "score": best_score,
                "delta_t_histogram": track_scores[best_track_id]["delta_t_histogram"],
            }
        else:
            return {"track_id": None, "score": 0, "delta_t_histogram": None}
    else:
        return {"track_id": None, "score": 0, "delta_t_histogram": None}


import sys
def deep_sizeof(obj, seen=None):
    """Berechnet die Gesamtspeichergröße eines Objekts rekursiv."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Vermeidet Doppelzählung
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_sizeof(i, seen) for i in obj)

    return size

import pickle

def save_db_hash_index(db_hash_index, file_path='db_hash_index.pkl'):
    """Speichert den db_hash_index in einer Datei."""
    with open(file_path, 'wb') as f:
        pickle.dump(db_hash_index, f)
    print(f"db_hash_index gespeichert in {file_path}")

def load_db_hash_index(file_path='db_hash_index.pkl'):
    """Lädt den db_hash_index aus einer Datei."""
    with open(file_path, 'rb') as f:
        db_hash_index = pickle.load(f)
    print(f"db_hash_index geladen von {file_path}")
    return db_hash_index


configuration = (8, 2) # (κ=8, τ=2)
(dist_freq, dist_time) = configuration

loadHashesFromFile = False
if not loadHashesFromFile:
    database_files = get_files(directory='../data/04', endswith='.mp3')

#database = prepare_database(database_files, configuration)
if not loadHashesFromFile:
    print ("Database preparation starts now")
    time_now = time.time()
    database = prepare_database_parallel(database_files, configuration)
    print ("Database preparation took", time.time()-time_now, "seconds")

print("Query preparation starts now")
query_list_original = get_files(directory='../queries/cut_output', endswith=".wav")
query_list_noise = get_files(directory='../queries/noise_output', endswith=".wav")
query_list_coding = get_files(directory='../queries/coding_output', endswith=".wav")
query_list_mobile = get_files(directory='../queries/mobile_output', endswith=".wav")


query_C_original = prepare_query_constellation_maps(query_list_original, configuration)
query_C_noise = prepare_query_constellation_maps(query_list_noise, configuration)
query_C_coding = prepare_query_constellation_maps(query_list_coding, configuration)
query_C_mobile = prepare_query_constellation_maps(query_list_mobile, configuration)
print("Query preparation done")

target_zones = [(20,50), (50, 20), (50, 50), (20, 20)]

if not loadHashesFromFile:
    print("computing hashes for database started")
    time_now = time.time()
    db_hash_index = compute_hashes_list_and_index(database, (20,50), exclusion_zone=5)
    print("computing hashes for database took", time.time()-time_now, "seconds")

if not loadHashesFromFile:
    save_db_hash_index(db_hash_index, 'db_hash_index.pkl')
else:
    print ("loading db_hash_index from file")
    db_hash_index = load_db_hash_index('db_hash_index.pkl')
    print ("db_hash_index loaded")

fst_query = query_C_original['1009604_cut.wav']['constellation_map']
query_hash_list = compute_hashes_for_query(fst_query, (20,50), exclusion_zone=5)

# Matching durchführen
time_now = time.time()
print("Now matching query to database non threaded")
result = match_query_to_database(query_hash_list, db_hash_index , threshold=5)
print("Matching took", time.time()-time_now, "seconds")

# Ergebnis ausgeben
if result["track_id"]:
    print(f"Best Match: {result['track_id']}")
    print(f"Score: {result['score']}")
    #print(f"Delta-T Histogram: {result['delta_t_histogram']}")
else:
    print("No significant match found.")

# Matching durchführen2
time_now = time.time()
print("Now matching query to database threaded")
result = match_query_to_database_threaded(query_hash_list, db_hash_index , threshold=5)
print("Threaded Matching took", time.time()-time_now, "seconds")

# Ergebnis ausgeben
if result["track_id"]:
    print(f"Best Match: {result['track_id']}")
    print(f"Score: {result['score']}")
    #print(f"Delta-T Histogram: {result['delta_t_histogram']}")
else:
    print("No significant match found.")

