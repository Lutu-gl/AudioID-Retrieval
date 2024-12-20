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
├── data/                # Raw music database (excluded from Git)
├── queries/             # Query audio files (10-second segments)
├── notebooks/           # Jupyter notebooks for analysis and experiments
├── src/                 # Python scripts for audio processing and matching
├── reports/             # Tables, figures, and findings
├── README.md            # Project documentation (this file)
├── requirements.txt     # Python dependencies
└── .gitignore           # Files and folders to exclude from Git
```

---

## **Setup and Installation**
Clone the repository and navigate to the project folder:
```bash
git clone <repository-url>
cd <repository-folder>
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Download the Dataset
```bash
wget https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/raw_30s_audio-04.tar
tar -xvf raw_30s_audio-XX.tar -C data/
```