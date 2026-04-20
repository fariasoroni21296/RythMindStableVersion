import os
import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_mfcc(file_path, n_mfcc=40, max_len=130):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if mfcc.shape[1] < max_len:
            pad = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)))
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc
    except:
        return None


def load_audio(audio_dir, max_gtzan=150):
    features = []
    names = []

    gtzan_count = 0

    for root, dirs, files in os.walk(audio_dir):
        folder = os.path.basename(root)

        for file in files:
            if file.endswith(".wav") or file.endswith(".au"):

                path = os.path.join(root, file)

                # 🎯 GTZAN control (skip Rabindra folder)
                if "Rabindra" not in folder:
                    if gtzan_count >= max_gtzan:
                        continue
                    gtzan_count += 1

                mfcc = extract_mfcc(path)
                if mfcc is not None:
                    features.append(mfcc)

                    base_name = file.split(".")[0]
                    name = folder + "_" + base_name
                    names.append(name)

    return np.array(features), names

def load_lyrics(lyrics_dir, names):

    drake_text = ""
    rabindra_text = ""

    try:
        with open(os.path.join(lyrics_dir, "drake_lyrics.txt"), encoding="utf-8") as f:
            drake_text = f.read()
    except:
        drake_text = "english song music rhythm"

    try:
        with open(os.path.join(lyrics_dir, "Rabindra_Sangeet_data.txt"), encoding="utf-8") as f:
            rabindra_text = f.read()
    except:
        rabindra_text = "bangla gaan kobita prem"

    texts = []

    for name in names:
        if "Rabindra" in name:
            texts.append(rabindra_text)
        else:
            texts.append(drake_text)

    if all(t.strip() == "" for t in texts):
        texts = ["music data"] * len(texts)

    vectorizer = TfidfVectorizer(max_features=500)
    features = vectorizer.fit_transform(texts).toarray()

    return features

def combine_features(audio_features, lyrics_features):
    return np.concatenate([audio_features, lyrics_features], axis=1)


def get_final_data(audio_dir, lyrics_dir):
    print("Loading audio (.wav + .au)...")
    audio_features, names = load_audio(audio_dir)

    print("Total audio loaded:", len(names))

    print("Loading lyrics (big file)...")
    lyrics_features = load_lyrics(lyrics_dir, names)

    print("Combining features...")
    X = combine_features(audio_features, lyrics_features)

    print("Final shape:", X.shape)

    return X, names