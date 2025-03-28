import google.generativeai as genai
import pandas as pd
import numpy as np
import spacy
import json
from tqdm import tqdm

tqdm.pandas()

# for stop words
nlp = spacy.load("en_core_web_sm")

def _remove_stop_words(text: str) -> str:
    doc = nlp(text) 
    filtered_tokens = [token for token in doc if not token.is_stop] 
    return ' '.join([token.text for token in filtered_tokens])


def _deserialize_embedding(row) -> list[float]:
    return np.array(json.loads(row['Embeddings']))


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
    convs['Embeddings'] = convs.apply(_deserialize_embedding, axis=1)
    
    return convs


def _make_embed_text(model):
  #@retry.Retry(timeout=300.0)
  def embed_fn(text: str) -> list[float]:
    # Set the task_type to CLUSTERING.
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="CLUSTERING")
    return embedding["embedding"]
  return embed_fn


def create_embeddings(df):
  model = 'models/embedding-001'
  df['Embeddings'] = df['content'].progress_apply(_make_embed_text(model))
  return df

