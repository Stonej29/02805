import pandas as pd
from langdetect import detect
import re, string, nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
stop_en = set(stopwords.words('english'))

def preprocess_posts_and_interactions(posts_file="../data/bluesky/ca_fire_20250101_20250207.jsonl", interact_file="../data/bluesky/interactions_20250101_20250207.jsonl"):
    """Preprocess posts and interactions from given jsonl files."""
    # ---------- load ----------
    posts  = pd.read_json(posts_file, lines=True)
    interact = pd.read_json(interact_file, lines=True)

    # ---------- language detection ----------
    def detect_language(text):
        try:
            return detect(text)
        except:
            return 'unknown'

    posts['langs'] = posts['text'].apply(detect_language)

    # ---------- filter to English only ----------
    posts = posts[posts['langs'] == 'en'].reset_index(drop=True)

    # ---------- regexes & cleaning fn (kept from your original) ----------
    url_re   = re.compile(r'https?://\S+')
    # NOTE: keep hashtag removal or not depending on whether you want to preserve hashtags
    mention_re = re.compile(r'[@#]\w+')
    emoji_re = re.compile("["                                
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    scanner_re = re.compile(r'\b(?:live audio feed|/webplayer/)\b')

    def clean(text):
        if not isinstance(text, str):
            return ''
        t = text.lower()
        t = url_re.sub(' ', t)
        t = mention_re.sub(' ', t)
        t = emoji_re.sub(' ', t)
        t = scanner_re.sub(' ', t)
        tokens = [w for w in t.split() if w not in stop_en and w not in string.punctuation]
        return ' '.join(tokens)

    # ---------- apply cleaning ----------
    posts['clean_text'] = posts['text'].apply(clean)

    # ---------- drop scanner spam ----------
    posts = posts[~posts['clean_text'].str.contains(
        r'live audio feed|/webplayer/|dispatch[\s,]*\d+(?:[,\s]+\d+)*', na=False)]

    # ---------- deduplicate by URI ----------
    # keep the first occurrence of each unique post URI
    before_rows = len(posts)
    posts = posts.drop_duplicates(subset='uri', keep='first').reset_index(drop=True)
    after_rows = len(posts)

    # ---------- create sets of unique URIs ----------
    posts_uris = set(posts['uri'].dropna())
    interacted_dst_uris_all = set(interact['dst_uri'].dropna())

    # ---------- interactions that point to posts *we have* ----------
    interacted_in_dataset = interacted_dst_uris_all & posts_uris

    # ---------- lonely posts (unique posts with zero interactions in our filtered interactions) ----------
    lonely_uris = posts_uris - interacted_in_dataset

    # ---------- optionally, filter interactions to only those relevant to our post-set ----------
    interact_filtered = interact[interact['dst_uri'].isin(posts_uris)].copy()

    # ---------- compute user set (authors in posts + sources of interactions that we kept) ----------
    users = set(posts['author_did'].dropna()) | set(interact_filtered['src_did'].dropna())

    # ---------- print summary stats (consistent) ----------
    print("=== dataset cleaning & dedupe ===")
    print(f"Rows after scanning-spam filter (before dedupe): {before_rows}")
    print(f"Rows after dedupe by uri: {after_rows}")
    print()

    print("=== URI / interaction summary (unique-URI basis) ===")
    total_unique_posts = len(posts_uris)
    print(f"Total unique clean posts: {total_unique_posts:,}")
    print(f"Unique post URIs that appear as interaction targets (in our interact file): {len(interacted_in_dataset):,}")
    print(f"Unique post URIs with zero interactions: {len(lonely_uris):,}")
    print(f"Check sum (unique targets + lonely) == total_unique_posts -> "
        f"{len(interacted_in_dataset) + len(lonely_uris)} == {total_unique_posts}")
    print()

    print("=== interactions (filtered to posts we retained) ===")
    print(f"Total interaction rows (original interactions file): {len(interact):,}")
    print(f"Interaction rows after filtering to dst_uri in our posts: {len(interact_filtered):,}")
    print("Interaction types (filtered):")
    if 'edge' in interact_filtered.columns:
        counts = interact_filtered['edge'].value_counts()
        for t, n in counts.items():
            print(f"{t:8s}: {n:,}")
    else:
        print("No 'edge' column found in interactions dataframe.")

    print()
    print(f"Distinct users (authors in posts ∪ src_did of filtered interactions): {len(users):,}")

    # ---------- show first 10 cleaned posts for inspection ----------
    print()
    print("=== first 10 cleaned posts (after filtering & dedupe) ===")
    with pd.option_context('display.max_colwidth', None):
        print(posts['clean_text'].head(10).to_string(index=False))

    return posts, interact_filtered
