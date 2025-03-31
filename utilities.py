import google.generativeai as genai
from google.api_core import retry

import pandas as pd
import numpy as np
import spacy
import json
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

# for stop words
nlp = spacy.load("en_core_web_sm")

def _remove_stop_words(text: str) -> str:
    doc = nlp(text) 
    filtered_tokens = [token for token in doc if not token.is_stop] 
    return ' '.join([token.text for token in filtered_tokens])


def _deserialize_embedding(row) -> list[float]:
    return np.array(json.loads(row['embeddings']))


def read_conversations(file_name: str, remove_stop_words: bool = True) -> pd.DataFrame:
    convs = pd.read_csv(
        file_name,
        # User questions don't have an answer_type
        usecols=lambda n: n != 'answer_type')
    
    if remove_stop_words:
        convs['content'] = convs['content'].apply(_remove_stop_words)
        # remove any records that are now empty
        convs = convs[convs['content'].str.len() > 0]

    # Remove all of the Bot's answers
    convs = convs[convs['author'] == 'USER']
    if 'embeddings' in convs.columns:
        convs['embeddings'] = convs.apply(_deserialize_embedding, axis=1)
    return convs


def _make_embed_text(model):
  @retry.Retry(timeout=300.0)
  def embed_fn(row) -> list[float]:
    if True: #  len(row['embeddings']) == 0:
        # Set the task_type to CLUSTERING.
        result = genai.embed_content(model=model,
                                   content=row['content'],
                                   task_type="CLUSTERING")
        return result['embedding']
    return row['embeddings']
  return embed_fn


def create_embeddings(df):
  model = 'models/embedding-001'
  df['embeddings'] = df.progress_apply(_make_embed_text(model), axis=1)
  return df


def combine_all_conversations():
    p = Path('conversations')
    combined = None
    for entry in p.iterdir():
        if entry.is_dir():
            file_name = entry.name
            file_path = Path(entry, f'{file_name}.csv')
            print(file_path)
            df = read_conversations(file_path, remove_stop_words=False)
            print(df.shape)
            if combined is None:
                combined = df[:]
            else:
                combined = pd.concat([combined, df])
    return combined


def compute_embeddings():
    p = Path('conversations')
    for entry in [Path('conversations/troy')]: # p.iterdir():
        if entry.is_dir():
            file_name = entry.name
            file_path = Path(entry, f'{file_name}.csv')
            print(f"Computing embeddings for: {file_path}")
            df = read_conversations(str(file_path), remove_stop_words=False)
            embeddings = create_embeddings(df)
            embeddings.to_csv(f'{file_path}', index=False)
