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
    output_textual_path: str = None,
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

    # --- Convert TF-IDF to DataFrame ---
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_feature_names, index=merged_df.index)

    # --- Concatenate all features ---
    all_features_df = pd.concat([
        merged_df[["Participant", "Interview", "Overall", "Excited"]].reset_index(drop=True),
        tfidf_df.reset_index(drop=True),
        pos_features_df.reset_index(drop=True),
        sentiment_df.reset_index(drop=True),
        bert_features_df.reset_index(drop=True)
    ], axis=1)

    logger.info("All features merged into single DataFrame with shape: %s", all_features_df.shape)

    # --- Optional save ---
    if output_textual_path:
        all_features_df.to_csv(output_textual_path, index=False)
        logger.info(f"Feature dataframe saved to {output_textual_path}")

    return all_features_df


# ==========================================================
# Default run
# ==========================================================
if __name__ == "__main__":
    df, X_combined, feature_names = process_textual_data("transcripts.csv", "scores.csv")
    print(df.head())
    print("Combined feature matrix shape:", X_combined.shape)
    
