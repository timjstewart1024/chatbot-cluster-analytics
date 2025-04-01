import google.generativeai as genai
from google.api_core import retry

import pandas as pd
import numpy as np
import spacy
import json
from pathlib import Path
from tqdm import tqdm
from csv import QUOTE_STRINGS

tqdm.pandas()

# for stop words
nlp = spacy.load("en_core_web_sm")

def _remove_stop_words(text: str) -> str:
    doc = nlp(text) 
    filtered_tokens = [token for token in doc if not token.is_stop] 
    return ' '.join([token.text for token in filtered_tokens])


def read_conversations(file_name: str, remove_stop_words: bool = False) -> pd.DataFrame:
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
    return convs


def _make_embed_text(model):
  @retry.Retry(timeout=300.0)
  def embed_fn(row) -> str:
    if row['embeddings'] == '[]':
        result = genai.embed_content(model=model,
                                     content=row['content'],
                                     task_type="CLUSTERING")['embedding']
        return str(result).replace('\n','')
    return str(row['embeddings']).replace('\n','')
  return embed_fn


def create_embeddings(df):
  model = 'models/embedding-001'
  df['embeddings'] = df.progress_apply(_make_embed_text(model), axis=1)
  return df


def deserialize_embeddings(df):
  df['embeddings'] = df['embeddings'].progress_apply(lambda x: json.loads(x))
  return df


def combine_all_conversations():
    p = Path('conversations')
    combined = None
    for entry in p.iterdir():
        if entry.is_dir():
            file_name = entry.name
            file_path = Path(entry, f'{file_name}.csv')
            print(file_path)
            df = read_conversations(str(file_path), remove_stop_words=False)
            print(df.shape)
            if combined is None:
                combined = df[:]
            else:
                combined = pd.concat([combined, df])
    return combined


def compute_embeddings():
    p = Path('conversations')
    for entry in p.iterdir():
        if entry.is_dir():
            file_name = entry.name
            file_path = Path(entry, f'{file_name}.csv')
            print(f"Computing embeddings for: {file_path}")
            df = read_conversations(str(file_path), remove_stop_words=False)
            embeddings = create_embeddings(df)
            embeddings.to_csv(f'{file_path}', index=False, quoting=QUOTE_STRINGS)
