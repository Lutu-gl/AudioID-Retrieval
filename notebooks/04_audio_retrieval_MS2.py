import os, sys
import numpy as np
import librosa
from scipy import ndimage
import matplotlib.pyplot as plt

configuration = (8, 2) # (κ=8, τ=2)

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


def plot_constellation_map(Cmap, Y=None, xlim=None, ylim=None, title='',
                           xlabel='Time (sample)', ylabel='Frequency (bins)',
                           s=5, color='r', marker='o', figsize=(7, 3), dpi=72):
    """Plot constellation map on top of a spectrogram"""
    if Cmap.ndim > 1:
        (K, N) = Cmap.shape
    else:
        K = Cmap.shape[0]
        N = 1
    if Y is None:
        Y = np.zeros((K, N))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    im = ax.imshow(Y, origin='lower', aspect='auto', cmap='gray_r', interpolation='nearest')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    Fs = 1
    if xlim is None:
        xlim = [-0.5 / Fs, (N - 0.5) / Fs]
    if ylim is None:
        ylim = [-0.5 / Fs, (K - 0.5) / Fs]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    n, k = np.argwhere(Cmap == 1).T
    ax.scatter(k, n, color=color, s=s, marker=marker)
    plt.tight_layout()
    return fig, ax, im


def compute_spectrogram(fn_wav, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    """Compute spectrogram for a given audio file"""
    # x, Fs = librosa.load(fn_wav, Fs)
    x, Fs = librosa.load(fn_wav, sr=Fs, duration=30)
    x_duration = len(x) / Fs
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    if bin_max is None:
        bin_max = X.shape[0]
    if frame_max is None:
        frame_max = X.shape[0]
    Y = np.abs(X[:bin_max, :frame_max])
    return Y

def prepare_database(file_list, configuration, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    """Prepare database by computing constellation maps for all configurations"""
    database = {}
    print(f"Processing {len(file_list)} files...")
    for file in file_list:

        # Compute spectrogram for the current file
        Y = compute_spectrogram(file, Fs=Fs, N=N, H=H, bin_max=bin_max, frame_max=frame_max)

        # Retrieve metadata for the file
        file_size = os.path.getsize(file)  # Size in bytes
        duration = librosa.get_duration(path=file)  # Duration in seconds

        # Initialize nested structure for the current file
        file_key = os.path.basename(file)
        database[file_key] = {
            "metadata": {
                "size": file_size,
                "original_query_id": os.path.splitext(file_key)[0],
            },
            "constellation_maps": {}
        }
        (dist_freq, dist_time) = configuration
        # Compute constellation maps for all configurations
        cmap = compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time)
        database[file_key]["constellation_maps"][(dist_freq, dist_time)] = cmap
    print("Processing done.")
    return database


def compute_hash(constellation_map, configuration, fan_out=5):
    """
    Compute hashes from a constellation map using anchor points.

    Parameters:
    - constellation_map (ndarray): The constellation map to generate hashes from.
    - configuration (tuple): Tuple (dist_freq, dist_time) used for computation.
    - fan_out (int): Number of target points per anchor point.

    Returns:
    - hashes (list): A list of hashes, each represented as (hash_value, anchor_time, track_id).
    """
    hashes = []
    freq_bins, time_bins = np.where(constellation_map == 1)

    # Loop through all anchor points
    for i, (freq1, time1) in enumerate(zip(freq_bins, time_bins)):
        # Pair anchor point with target points
        for j in range(1, fan_out + 1):
            if i + j < len(freq_bins):  # Ensure target exists
                freq2 = freq_bins[i + j]
                time2 = time_bins[i + j]

                # Compute hash as a combination of anchor and target
                delta_time = time2 - time1
                if delta_time > 0:  # Only consider forward target points
                    hash_value = (freq1, freq2, delta_time)
                    hashes.append((hash_value, time1))  # Append hash with anchor time
    return hashes


def add_hashes_to_database(database, configuration, fan_out=5):
    """
    Compute and store hashes for all files in the database.

    Parameters:
    - database (dict): The database containing constellation maps.
    - configuration (tuple): Tuple (dist_freq, dist_time) used for computation.
    - fan_out (int): Number of target points per anchor point.
    """
    for file_key, file_data in database.items():
        # Retrieve the constellation map for the given configuration
        constellation_map = file_data["constellation_maps"].get(configuration)
        if constellation_map is not None:
            # Compute hashes
            hashes = compute_hash(constellation_map, configuration, fan_out=fan_out)
            # Store hashes in the database
            database[file_key]["hashes"] = hashes


def compute_hash_with_target_zones_32bit(constellation_map, target_zones, fan_out=5):
    """
    Compute hashes from a constellation map using target zones, storing them as 32-bit unsigned integers.

    Parameters:
    - constellation_map (ndarray): The constellation map to generate hashes from.
    - target_zones (list): List of tuples, each defining a target zone:
        [(freq_range, time_range), ...]
        freq_range = (min_freq, max_freq)
        time_range = (min_time, max_time)
    - fan_out (int): Maximum number of target points per anchor point.

    Returns:
    - hashes (list): A list of hashes, each represented as a 32-bit unsigned integer.
    """
    hashes = []
    freq_bins, time_bins = np.where(constellation_map == 1)

    for target_zone in target_zones:
        freq_range, time_range = target_zone
        min_freq, max_freq = freq_range
        min_time, max_time = time_range

        # Loop through all anchor points
        for i, (freq1, time1) in enumerate(zip(freq_bins, time_bins)):
            # Filter target points based on the target zone
            for j in range(1, fan_out + 1):
                if i + j < len(freq_bins):  # Ensure target exists
                    freq2 = freq_bins[i + j]
                    time2 = time_bins[i + j]

                    # Check if target point falls within the target zone
                    delta_time = time2 - time1
                    if (min_freq <= freq2 <= max_freq) and (min_time <= delta_time <= max_time):
                        # Compute 32-bit hash: Pack (f1, f2, deltaTime) into a single 32-bit integer
                        hash_value = (
                                (freq1 & 0x3FF) << 22 |  # 10 bits for freq1
                                (freq2 & 0x3FF) << 12 |  # 10 bits for freq2
                                (delta_time & 0xFFF)  # 12 bits for deltaTime
                        )
                        hashes.append((np.uint32(hash_value), time1))  # Store as 32-bit unsigned integer
    return hashes


def add_hashes_with_target_zones_32bit_to_database(database, target_zones, fan_out=5):
    """
    Compute and store 32-bit hashes for all files in the database using target zones.

    Parameters:
    - database (dict): The database containing constellation maps.
    - target_zones (list): List of target zones for hash generation.
    - fan_out (int): Maximum number of target points per anchor point.
    """
    for file_key, file_data in database.items():
        constellation_map = file_data["constellation_maps"].get(configuration)
        if constellation_map is not None:
            # Compute hashes for each target zone
            for idx, target_zone in enumerate(target_zones):
                hashes = compute_hash_with_target_zones_32bit(constellation_map, [target_zone], fan_out=fan_out)
                file_data[f"hashes_zone_{idx + 1}"] = hashes


def prepare_query_constellation_maps(file_list, configurations, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
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
            "constellation_maps": {}
        }

        # Compute constellation maps for all configurations
        (dist_freq, dist_time) = configuration
        cmap = compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time)
        query_maps[file_key]["constellation_maps"][(dist_freq, dist_time)] = cmap
    return query_maps

def match_query_to_database(queries, database):
    """
    Match the query hashes against the database to find the best matching track.

    Parameters:
    - query_hashes (list): List of query hashes as (hash_value, query_time).
    - database (dict): Database containing track hashes as
      {track_id: [(hash_value, db_time), ...]}.

    Returns:
    - match_scores (dict): Dictionary of match scores for each track ID.
    - best_match (str): Track ID with the highest match score.
    """
    time_pair_bins = {}

    # Step 1: Match each query hash with database hashes
    for query_hash, query_time in queries["hashes_zone_1"]:
        for track_id, track_data in database.items():
            for db_hash, db_time in track_data["hashes_zone_1"]:
                if query_hash == db_hash:
                    # Calculate the time difference (delta_t)
                    delta_t = db_time - query_time

                    # Group by (track_id, delta_t)
                    if (track_id, delta_t) not in time_pair_bins:
                        time_pair_bins[(track_id, delta_t)] = 0
                    time_pair_bins[(track_id, delta_t)] += 1

    # Step 2: Aggregate scores by track_id
    match_scores = {}
    for (track_id, delta_t), count in time_pair_bins.items():
        if track_id not in match_scores:
            match_scores[track_id] = 0
        match_scores[track_id] += count

    # Step 3: Identify the best matching track
    best_match = max(match_scores, key=match_scores.get, default=None)

    return match_scores, best_match


# Database holen.
file_list = get_files(directory='../data/04', endswith=".mp3")

# Prepare database
database = prepare_database(file_list, configuration)

# Display constellation maps for the first file with all configurations
first_file = file_list[0]

# Compute the spectrogram for the first file
Y = compute_spectrogram(first_file, Fs=22050, N=2048, H=1024, bin_max=128)

# Plot constellation maps for all configurations
(dist_freq, dist_time) = configuration
cmap = compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time)
title = f"Constellation Map for {os.path.basename(first_file)} (κ={dist_freq}, τ={dist_time})"
fig, ax, im = plot_constellation_map(cmap, Y=Y, title=title)

#plt.show()

#print(database)
print(database['12304.mp3']["constellation_maps"][configuration].shape)
print(database['1604.mp3']["constellation_maps"][configuration].shape)

# Compute and store hashes in the database
add_hashes_to_database(database, configuration, fan_out=5)

# Prüfen, ob die Hashes korrekt generiert wurden
first_file_key = list(database.keys())[0]
print(f"Hashes for {first_file_key}:")
print(database[first_file_key]["hashes"][:10])  # Zeige die ersten 10 Hashes

# Definiere die Target Zones
target_zones = [
    ((0, 50), (1, 10)),  # Frequenzen 0-50 Hz, Zeitdifferenzen 1-10 Samples
    ((50, 100), (1, 10)),  # Frequenzen 50-100 Hz, Zeitdifferenzen 1-10 Samples
    ((0, 50), (10, 20)),  # Frequenzen 0-50 Hz, Zeitdifferenzen 10-20 Samples
    ((50, 100), (10, 20))  # Frequenzen 50-100 Hz, Zeitdifferenzen 10-20 Samples
]

# Berechne und speichere die Hashes
add_hashes_with_target_zones_32bit_to_database(database, target_zones)

# Überprüfe die generierten Hashes
first_file_key = list(database.keys())[0]
print(f"Hashes for {first_file_key}, Zone 1:")
print(database[first_file_key]["hashes_zone_1"][:10])  # Zeige die ersten 10 Hashes


### QUERIES
query_list_original = get_files(directory='../queries/cut_output', endswith=".wav")

# Print the total number of files to be processed
query_C_original = prepare_query_constellation_maps(query_list_original, configuration)
add_hashes_with_target_zones_32bit_to_database(query_C_original, target_zones)


match_scores, best_match = match_query_to_database(query_C_original['903604_cut.wav'], database)

print("Match Scores:", match_scores)
print("Best Match:", best_match)