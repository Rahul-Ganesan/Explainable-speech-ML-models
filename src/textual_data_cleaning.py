# ==========================================================
# textual_data_cleaning.py
# ==========================================================

import pandas as pd
import nltk
import re
import logging
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
import torch
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# ============================
# Logging Setup
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# Download NLTK Resources (if not already present)
# ============================
if not nltk.download('punkt', quiet=True):
    nltk.download('punkt')
if not nltk.download('averaged_perceptron_tagger', quiet=True):
    nltk.download('averaged_perceptron_tagger')
if not nltk.download('averaged_perceptron_tagger_eng', quiet=True):
    nltk.download('averaged_perceptron_tagger_eng')
if not nltk.download('stopwords', quiet=True):
    nltk.download('stopwords')
if not nltk.download('wordnet', quiet=True):
    nltk.download('wordnet')
if not nltk.download('omw-1.4', quiet=True):
    nltk.download('omw-1.4')
if not nltk.download('punkt_tab', quiet=True):
    nltk.download('punkt_tab')
if not nltk.download('vader_lexicon', quiet=True):
    nltk.download('vader_lexicon')
    
    
# ============================
# Preprocessing helpers
# ============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    pos_tags = nltk.pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return ' '.join(tokens)

def extract_text(text, sep='Interviewee:'):
    segments = str(text).split('|')
    interviewee_lines = [
        seg.replace(sep, '').strip() for seg in segments if seg.startswith(sep)
    ]
    return ' '.join(interviewee_lines)

def convert_pid(pid):
    pid = str(pid).upper()
    if pid.startswith("PP"):
        return int(re.sub(r'\D', '', pid)), 2
    elif pid.startswith("P"):
        return int(re.sub(r'\D', '', pid)), 1
    else:
        raise ValueError(f"Invalid participant format: {pid}")

# ==========================================================
# Main function
# ==========================================================
def process_textual_data(
    transcripts_path: str,
    score_path: str,
    output_path: str = None,
    score_cols=("Overall", "Excited"),
    sep="Interviewee:",
    tfidf_max_features=500
):
    # --- Load transcripts ---
    logger.info("Loading transcripts...")
    transcripts = pd.read_csv(transcripts_path, header=None)
    transcripts.columns = ["participant_raw", "transcript_text"]

    # --- Convert P/PP IDs ---
    transcripts[["Participant", "Interview"]] = transcripts["participant_raw"].apply(
        lambda x: pd.Series(convert_pid(x))
    )

    # --- Extract and preprocess text ---
    transcripts["interviewee_text"] = transcripts["transcript_text"].apply(lambda x: extract_text(x, sep=sep))
    transcripts["processed_interviewee_text"] = transcripts["interviewee_text"].apply(preprocess_text)

    # --- Sort by participant/interview ---
    transcripts = transcripts.sort_values(["Participant", "Interview"]).reset_index(drop=True)

    # --- Load and clean scores ---
    logger.info("Loading scores...")
    scores = pd.read_csv(score_path)
    scores = scores.rename(columns={"Participant": "participant_raw"})
    scores[["Participant", "Interview"]] = scores["participant_raw"].apply(lambda x: pd.Series(convert_pid(x)))

    # --- Merge scores ---
    merged_df = transcripts.merge(
        scores[["Participant", "Interview"] + list(score_cols)],
        on=["Participant", "Interview"],
        how="inner"
    )

    # --- Feature Engineering ---
    logger.info("Building TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,3))
    X_tfidf = tfidf_vectorizer.fit_transform(merged_df["processed_interviewee_text"])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    logger.info("Computing POS features...")
    def pos_distribution(text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        counts = Counter(tag for word, tag in pos_tags)
        total = sum(counts.values())
        for tag in counts: counts[tag] /= total
        return counts

    pos_features_df = merged_df["processed_interviewee_text"].apply(pos_distribution).apply(pd.Series).fillna(0)

    logger.info("Computing VADER sentiment features...")
    analyzer = SentimentIntensityAnalyzer()
    def compute_sentiment_scores(text):
        scores = analyzer.polarity_scores(text)
        return [scores["pos"], scores["neu"], scores["neg"], scores["compound"]]

    sentiment_df = pd.DataFrame(
        merged_df["processed_interviewee_text"].apply(compute_sentiment_scores).tolist(),
        columns=["sent_pos", "sent_neu", "sent_neg", "sent_compound"]
    )

    logger.info("Computing BERT embeddings (this may take a while)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    def get_bert_embedding(text):
        tokens = tokenizer(text, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding

    bert_features_df = pd.DataFrame(
        merged_df["processed_interviewee_text"].apply(get_bert_embedding).tolist(),
        columns=[f"bert_{i}" for i in range(768)]
    )

    logger.info("Combining all features into a single sparse matrix...")
    # X_combined = hstack([
    #     X_tfidf,
    #     csr_matrix(pos_features_df.values),
    #     csr_matrix(sentiment_df.values),
    #     csr_matrix(bert_features_df.values)
    # ])

    # feature_names = list(tfidf_feature_names) + list(pos_features_df.columns) + list(sentiment_df.columns) + list(bert_features_df.columns)

    # --- Convert TF-IDF to DataFrame ---
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_feature_names, index=merged_df.index)

    # --- Concatenate all features ---
    all_features_df = pd.concat([
        merged_df[["Participant", "Interview", "processed_interviewee_text"]].reset_index(drop=True),
        tfidf_df.reset_index(drop=True),
        pos_features_df.reset_index(drop=True),
        sentiment_df.reset_index(drop=True),
        bert_features_df.reset_index(drop=True)
    ], axis=1)

    logger.info("All features merged into single DataFrame with shape: %s", all_features_df.shape)

    # --- Optional save ---
    if output_path:
        all_features_df.to_csv(output_path, index=False)
        logger.info(f"Feature dataframe saved to {output_path}")

    return all_features_df


# ==========================================================
# Default run
# ==========================================================
if __name__ == "__main__":
    df, X_combined, feature_names = process_textual_data("transcripts.csv", "scores.csv")
    print(df.head())
    print("Combined feature matrix shape:", X_combined.shape)
    
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def get_wordnet_pos(tag):
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN

# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', ' ', text)
#     tokens = nltk.word_tokenize(text)
#     tokens = [t for t in tokens if t not in stop_words]
#     pos_tags = nltk.pos_tag(tokens)
#     tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
#     return ' '.join(tokens)

# def extract_text(text, sep='Interviewee:'):
#     segments = str(text).split('|')
#     interviewee_lines = [
#         seg.replace(sep, '').strip()
#         for seg in segments if seg.startswith(sep)
#     ]
#     return ' '.join(interviewee_lines)

# def split_id_interleaved(pid):
#     pid = str(pid)
#     match = re.match(r'([A-Za-z]+)(\d+)$', pid)
#     if match:
#         prefix = match.group(1)
#         number = int(match.group(2))
#         return number, prefix
#     else:
#         return float('inf'), pid
    
# def convert_pid(pid):
#     pid = str(pid).upper().strip()
#     if pid.startswith("PP"):
#         interview = 2
#         number = int(re.sub(r'\D', '', pid))
#     elif pid.startswith("P"):
#         interview = 1
#         number = int(re.sub(r'\D', '', pid))
#     else:
#         raise ValueError(f"Invalid participant format: {pid}")
#     return number, interview

# # ==========================================================
# # MAIN FUNCTION 
# # ==========================================================
# # def process_textual_data(
# #     transcripts_path: str,
# #     score_path: str,
# #     output_path: str = None,
# #     score_cols=("Overall", "Excited"),
# #     sep="Interviewee:"
# # ):
# #     """
# #     Cleans transcript text and merges with participant scores.

# #     Transcript IDs:
# #         P1  -> Participant = 1, Interview = 1
# #         PP1 -> Participant = 1, Interview = 2

# #     Score IDs:
# #         p1 -> Participant = 1

# #     Merges on (Participant, Interview).
# #     If score file has no Interview column, scores are duplicated
# #     across both interviews for that participant.
# #     """

# #     logger.info("Loading transcripts...")

# #     transcripts = pd.read_csv(
# #         transcripts_path,
# #         header=None,
# #         names=["participant&interview_id", "transcript_text"]
# #     )

# #     transcripts[["Participant", "Interview"]] = transcripts["participant&interview_id"].apply(
# #         lambda x: pd.Series(convert_pid(x))
# #     )

# #     # -----------------------------
# #     # Clean transcript text
# #     # -----------------------------
# #     logger.info("Extracting interviewee text...")
# #     transcripts["interviewee_text"] = transcripts["transcript_text"].apply(
# #         lambda x: extract_text(x, sep=sep)
# #     )

# #     logger.info("Preprocessing text...")
# #     transcripts["processed_interviewee_text"] = transcripts["interviewee_text"].apply(preprocess_text)

# #     transcripts = transcripts.sort_values(
# #         by=["Participant", "Interview"]
# #     ).reset_index(drop=True)

# #     # -----------------------------
# #     # Load and clean scores
# #     # -----------------------------
# #     logger.info("Loading scores...")
# #     scores = pd.read_csv(score_path)
# #     scores['id'] = scores['Participant']

# #     # Clean Participant column (p1 -> 1)
# #     scores[["Participant", "Interview"]] = transcripts["id"].apply(
# #         lambda x: pd.Series(convert_pid(x))
# #     )
# #     # scores["Participant"] = scores["Participant"].apply(
# #     #     lambda x: int(re.sub(r'\D', '', str(x)))
# #     # )

# #     # # If Interview is missing, replicate scores for both interviews
# #     # if "Interview" not in scores.columns:
# #     #     logger.info("No Interview column in scores — replicating for Interview 1 and 2.")
# #     #     scores = pd.concat([
# #     #         scores.assign(Interview=1),
# #     #         scores.assign(Interview=2)
# #     #     ]).reset_index(drop=True)

# #     # -----------------------------
# #     # Merge cleanly
# #     # -----------------------------
# #     logger.info("Merging transcripts with scores...")
# #     merged_df = transcripts.merge(
# #         scores[["Participant", "Interview"] + list(score_cols)],
# #         on=["Participant", "Interview"],
# #         how="inner"
# #     )

# #     merged_df = merged_df[[
# #         "Participant",
# #         "Interview",
# #         "transcript_text",
# #         "interviewee_text",
# #         "processed_interviewee_text"
# #     ] + list(score_cols)]
    
# #     merged_df = merged_df.drop(["transcript_text", "interviewee_text"], axis=1)
    
# #     merged_df = merged_df.sort_values(
# #         by=["Participant", "Interview"]
# #     ).reset_index(drop=True)
    

# #     logger.info("Textual cleaning and merge complete.")

# #     if output_path:
# #         merged_df.to_csv(output_path, index=False)
# #         logger.info(f"Saved merged dataset to {output_path}")

# #     return merged_df

# def process_textual_data(
#     transcripts_path: str,
#     score_path: str,
#     output_path: str = None,
#     score_cols=("Overall", "Excited"),
#     sep="Interviewee:"
# ):
#     """
#     Cleans transcript text and merges with participant scores.
#     Ensures both transcripts and scores use the same (Participant, Interview) keys
#     derived from 'p1' / 'pp1' style ids.

#     - transcripts_path: CSV of transcripts. Expected either:
#         1) two columns with no header: (id, transcript_text)
#         2) a headered CSV that contains an ID column and a transcript text column.
#     - score_path: CSV of scores with column 'Participant' containing 'p1'/'pp1' style ids.
#     - output_path: optional path to save merged csv
#     - score_cols: score columns to merge from scores file
#     - sep: speaker separator used to extract Interviewee lines
#     """

#     logger.info("Loading transcripts...")
#     # Try to read transcripts robustly:
#     raw_transcripts = pd.read_csv(transcripts_path, header=None)

#     # If there are exactly 2 columns and no header, interpret as [id, text].
#     if raw_transcripts.shape[1] == 2 and raw_transcripts.columns.tolist() == [0, 1]:
#         transcripts = raw_transcripts.copy()
#         transcripts.columns = ["id", "transcript_text"]
#     else:
#         # If the file has header rows, try to find sensible columns
#         transcripts = pd.read_csv(transcripts_path)
#         # try to locate likely id and text columns
#         # prefer exact names if present
#         if "participant&interview_id" in transcripts.columns and "transcript_text" in transcripts.columns:
#             transcripts = transcripts.rename(columns={"participant&interview_id": "id", "transcript_text": "transcript_text"})
#         elif "Participant" in transcripts.columns and transcripts.shape[1] >= 2:
#             # assume first non-Participant column is text
#             text_col = [c for c in transcripts.columns if c not in ("Participant", "participant_id")][-1]
#             transcripts = transcripts.rename(columns={"Participant": "id", text_col: "transcript_text"})
#         else:
#             # fall back: use first two columns
#             transcripts = transcripts.iloc[:, :2].copy()
#             transcripts.columns = ["id", "transcript_text"]

#     # helper to convert 'p1' / 'pp1' -> (participant_number:int, interview:int)
#     def convert_pid(pid):
#         pid = str(pid).strip().upper()
#         # remove whitespace
#         pid = re.sub(r"\s+", "", pid)
#         m = re.match(r'^(PP+|P+)(\d+)$', pid)
#         if not m:
#             raise ValueError(f"Invalid participant id format: '{pid}'")
#         prefix = m.group(1)
#         num = int(m.group(2))
#         # if prefix starts with 'PP' (two or more P's) treat as interview 2
#         # if prefix is single 'P' treat as interview 1
#         interview = 2 if prefix.startswith("PP") else 1
#         return num, interview

#     # Convert transcripts id -> Participant, Interview
#     transcripts[["Participant", "Interview"]] = transcripts["id"].apply(
#         lambda x: pd.Series(convert_pid(x))
#     )

#     # Extract interviewee text and preprocess
#     logger.info("Extracting interviewee text...")
#     transcripts["interviewee_text"] = transcripts["transcript_text"].apply(lambda x: extract_text(x, sep=sep))
#     logger.info("Preprocessing text...")
#     transcripts["processed_interviewee_text"] = transcripts["interviewee_text"].apply(preprocess_text)

#     # Normalize types
#     transcripts["Participant"] = transcripts["Participant"].astype(int)
#     transcripts["Interview"] = transcripts["Interview"].astype(int)

#     # ----- Load scores and convert IDs the same way -----
#     logger.info("Loading scores...")
#     scores = pd.read_csv(score_path)

#     # If the scores file uses 'Participant' as the column name (e.g. 'p1'), copy it to 'id'
#     if "Participant" in scores.columns:
#         scores["id"] = scores["Participant"]
#     elif "participant_id" in scores.columns:
#         scores["id"] = scores["participant_id"]
#     else:
#         # fallback: assume first column is id
#         scores["id"] = scores.iloc[:, 0]

#     # Apply same convert_pid to scores['id']
#     scores[["Participant", "Interview"]] = scores["id"].apply(lambda x: pd.Series(convert_pid(x)))

#     # Ensure Participant and Interview are ints
#     scores["Participant"] = scores["Participant"].astype(int)
#     scores["Interview"] = scores["Interview"].astype(int)

#     # If scores lack the requested score columns, raise a helpful error
#     missing_score_cols = [c for c in score_cols if c not in scores.columns]
#     if missing_score_cols:
#         raise ValueError(f"Score file missing columns: {missing_score_cols}. Found columns: {scores.columns.tolist()}")

#     # Drop transcipt and interviewee text to save memory
#     transcripts = transcripts.drop(["transcript_text", "interviewee_text"], axis=1)

#     # ---- Merge on both Participant and Interview ----
#     logger.info("Merging transcripts with scores on ['Participant','Interview'] ...")
#     merged_df = transcripts.merge(
#         scores[["Participant", "Interview"] + list(score_cols)],
#         on=["Participant", "Interview"],
#         how="inner"
#     )

#     # Reorder columns to match prosodic pipeline
#     merged_df = merged_df[[
#         "Participant",
#         "Interview",
#         "processed_interviewee_text"
#     ] + list(score_cols)]
    
#     # Sort final dataframe
#     merged_df = merged_df.sort_values(
#         by=["Participant", "Interview"]
#     ).reset_index(drop=True)

#     logger.info(f"Merged rows: {len(merged_df)}")
#     logger.info("Textual cleaning and merge complete.")

#     if output_path:
#         merged_df.to_csv(output_path, index=False)
#         logger.info(f"Saved merged dataset to {output_path}")

#     return merged_df


# ==========================================================
# Default Run (if executed directly)
# ==========================================================
# if __name__ == "__main__":
#     df = process_textual_data("transcripts.csv", "scores.csv")
#     print(df.head())
