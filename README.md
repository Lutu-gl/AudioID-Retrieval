# MusicFingerprinting 🎵  
**An Audio Identification System for Music Retrieval**  

---

## **Overview**  
This project implements a **music retrieval system** capable of identifying audio tracks from short, distorted queries.  
Inspired by **Avery Wang's Shazam algorithm** (ISMIR 2003), the system processes audio data, generates constellation maps, and matches queries against a database of tracks.  

The project is part of the **Information Retrieval course**, focusing on content-based audio retrieval and efficient matching strategies.

---

## **Features**  
- **Audio Query Processing**: Extract 10-second segments and simulate various distortions:
  - Original (unmodified)
  - Noise (Gaussian noise added)
  - Coding (compressed audio with artifacts)
  - Mobile recording (outdoor playback and re-recording)  
- **Constellation Maps**: Generate time-frequency representations for efficient matching.  
- **Matching and Retrieval**: Perform query matching against a music database using precision, recall, and other metrics.  
- **Scalability**: Extend the system to larger datasets and more efficient hashing approaches in future milestones.

---

## **Project Structure**  

```plaintext
├── data/                  # Raw music database (excluded from Git)
├── queries/               # Processed query audio files (10-second segments)
│   ├── random20/          # 20 Randomly music pieces for queries
│   ├── noise_output/      # Queries with Gaussian noise added (10s long)
│   ├── coding_output/     # Compressed queries (10s long)
│   ├── cut_output/        # Segmented and extracted audio cuts (10s long)
│   ├── moible_output/     # Queries played on a phone in an outdoor environment (10s long)
│   └── process_mp3.sh     # Script to process the audio files with noise and coding
├── notebooks/             # Jupyter notebooks for analysis and experiments
├── evaluation/            # CSV files with evaluation results
├── README.md              # Project documentation (this file)
├── requirements.txt       # Python dependencies
└── .gitignore             # Files and folders to exclude from Git

```

---

## **Setup and Installation**
Clone the repository and navigate to the project folder:
```bash
git clone https://github.com/Lutu-gl/AudioID-Retrieval.git
cd AudioID-Retrieval
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Download the Dataset
```bash
wget https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/raw_30s_audio-04.tar
tar -xvf raw_30s_audio-04.tar -C data/
```

## **Acknowledgements**
- Avery Wang: For pioneering the [Shazam algorithm](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf) and audio fingerprinting.
- Book [Fundamentals of Music Processing](https://link.springer.com/book/10.1007/978-3-030-69808-9)
- Jupiter notebook for [Audio Identification](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.ipynb)
- [MTG-Jamendo Dataset](https://mtg.github.io/mtg-jamendo-dataset/)