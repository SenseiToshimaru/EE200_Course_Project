import os
import glob
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import Counter
import pickle
import gzip 

# ==========================================
# 1. CORE PIPELINE & DATABASE LOGIC
# ==========================================

@st.cache_data
def load_audio(filepath):
    audio, fs = librosa.load(filepath, sr=None, mono=True)
    return fs, audio

def get_constellation(Sxx, percentile_threshold=85):
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    local_max = maximum_filter(Sxx_log, size=20) == Sxx_log
    dynamic_threshold = np.percentile(Sxx_log, percentile_threshold)
    background = (Sxx_log > dynamic_threshold)
    peaks = local_max & background
    freq_bins, time_frames = np.where(peaks)
    return list(zip(time_frames, freq_bins)), Sxx_log

def generate_hashes(constellation, delay_limit=50):
    hashes = []
    constellation = sorted(constellation, key=lambda x: x[0])
    for i in range(len(constellation)):
        for j in range(i + 1, len(constellation)):
            t1, f1 = constellation[i]
            t2, f2 = constellation[j]
            delta_t = t2 - t1
            if delta_t <= delay_limit:
                hashes.append(((f1, f2, delta_t), t1))
            else:
                break 
    return hashes

def fingerprint_song(fs, audio):
    f_bins, t_frames, Sxx = spectrogram(audio, fs, nperseg=1024, noverlap=512)
    constellation, Sxx_log = get_constellation(Sxx, percentile_threshold=85)
    hashes = generate_hashes(constellation, delay_limit=50)
    return hashes, Sxx_log, f_bins, t_frames, constellation

def match_hashes(query_hashes, db):
    """Refactored matching logic to be reused in both modes."""
    matches = []
    for hash_val, t_query in query_hashes:
        if hash_val in db:
            for song_name, t_db in db[hash_val]:
                matches.append((song_name, t_db - t_query))
                
    if not matches:
        return "No match found", 0, 0, []
        
    match_counts = Counter(matches)
    best_match_tuple, highest_score = match_counts.most_common(1)[0]
    return best_match_tuple[0], highest_score, best_match_tuple[1], matches


@st.cache_resource
def load_or_build_database(dataset_folder):
    import gzip
import pickle
import glob
import os
import streamlit as st

@st.cache_resource
def load_or_build_database(dataset_folder):
    # 1. Look for the chunked files we just created
    chunk_files = sorted(glob.glob("db_chunk_*.bin"))
    
    if chunk_files:
        print("Stitching database chunks together in memory...")
        full_binary_data = b""
        
        # Read each piece and glue the bytes together
        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as f:
                full_binary_data += f.read()
                
        # Decompress the glued bytes and load the dictionary
        return pickle.loads(gzip.decompress(full_binary_data))

    # Notice we changed the file extension to .pkl.gz
    cache_path = "app_db_cache.pkl.gz" 
    
    if os.path.exists(cache_path):
        # We use gzip.open instead of standard open
        with gzip.open(cache_path, 'rb') as f: 
            return pickle.load(f)
            
    song_db = {}
    search_pattern = os.path.join(dataset_folder, "*.*")
    files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.wav', '.mp3'))]
    
    for filepath in files:
        song_name = os.path.splitext(os.path.basename(filepath))[0]
        fs, audio = load_audio(filepath)
        hashes, _, _, _, _ = fingerprint_song(fs, audio)
        
        for hash_val, t_db in hashes:
            if hash_val not in song_db:
                song_db[hash_val] = []
            song_db[hash_val].append((song_name, t_db))
            
    # Save using gzip.open to compress it on the fly
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(song_db, f)
    
    return song_db

# ==========================================
# 2. STREAMLIT UI & MODES
# ==========================================

st.set_page_config(page_title="Audio Identifier", layout="wide")
st.title("🎵 Interactive Audio Identifier")

# Initialize DB
dataset_folder = "EE200 Project Song Database"
with st.spinner("Loading Database..."):
    db = load_or_build_database(dataset_folder)
st.sidebar.success(f"Database Active: {len(db)} unique hashes loaded.")

# Create the two required modes via Tabs
tab_single, tab_batch = st.tabs(["Single-Clip Mode", "Batch Mode"])

# --- TAB 1: SINGLE CLIP MODE ---
with tab_single:
    st.markdown("Upload a single query clip to view the visual identification pipeline.")
    single_file = st.file_uploader("Upload a Query Clip (.wav/.mp3)", type=["wav", "mp3"], key="single")

    if single_file is not None:
        temp_path = "temp_single" + os.path.splitext(single_file.name)[1]
        with open(temp_path, "wb") as f:
            f.write(single_file.getbuffer())
            
        fs, audio = load_audio(temp_path)
        q_hashes, q_Sxx_log, q_f_bins, q_t_frames, q_constellation = fingerprint_song(fs, audio)
        
        best_song, score, best_offset, all_matches = match_hashes(q_hashes, db)
        
        # Display Visuals
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Step 1: Spectrogram")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.pcolormesh(q_t_frames, q_f_bins, q_Sxx_log, shading='gouraud', cmap='viridis')
            ax1.set_ylabel('Frequency [Hz]')
            ax1.set_xlabel('Time [sec]')
            st.pyplot(fig1)

        with col2:
            st.subheader("Step 2: Constellation Map")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            t_peaks, f_peaks = zip(*q_constellation) if q_constellation else ([], [])
            if t_peaks:
                ax2.scatter(np.array(t_peaks) * (q_t_frames[1] - q_t_frames[0]), 
                            np.array(f_peaks) * (q_f_bins[1] - q_f_bins[0]), 
                            c='red', s=5)
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [sec]')
            ax2.set_xlim(0, max(q_t_frames) if len(q_t_frames) > 0 else 1)
            ax2.set_ylim(0, max(q_f_bins) if len(q_f_bins) > 0 else 1)
            st.pyplot(fig2)

        st.subheader("Step 3: Offset Histogram")
        if all_matches:
            songs = list(set([m[0] for m in all_matches]))
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            for song in songs:
                offsets = [m[1] for m in all_matches if m[0] == song]
                ax3.hist(offsets, bins=100, alpha=0.7, label=song)
            ax3.set_xlabel('Time Offset ($\Delta t$)')
            ax3.set_ylabel('Number of Matching Hashes')
            
            # Only show legend if there aren't too many random songs cluttering it
            if len(songs) < 15:
                ax3.legend()
            st.pyplot(fig3)
            
            st.success(f"🎉 **Final Verdict:** {best_song} (Score: {score} matches at offset {best_offset})")
        else:
            st.warning("No matches found in the database.")
            
        os.remove(temp_path)

# --- TAB 2: BATCH MODE ---
with tab_batch:
    st.markdown("Upload multiple query clips to generate an auto-grader compatible `results.csv`.")
    batch_files = st.file_uploader("Upload Multiple Query Clips", type=["wav", "mp3"], accept_multiple_files=True, key="batch")

    if st.button("Run Batch Processing") and batch_files:
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(batch_files):
            temp_path = "temp_batch_" + file.name
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
                
            fs, audio = load_audio(temp_path)
            q_hashes, _, _, _, _ = fingerprint_song(fs, audio)
            best_song, _, _, _ = match_hashes(q_hashes, db)
            
            # Append exact requested format
            results.append({"filename": file.name, "prediction": best_song})
            os.remove(temp_path)
            
            progress_bar.progress((i + 1) / len(batch_files))
            
        # Create DataFrame and CSV
        df = pd.DataFrame(results)
        st.dataframe(df) # Show preview to user
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download results.csv",
            data=csv_data,
            file_name='results.csv',
            mime='text/csv',
        )
