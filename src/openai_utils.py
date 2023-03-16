import constants
import openai
import pandas as pd
import numpy as np
import whisper
from openai.embeddings_utils import get_embedding, cosine_similarity

openai.api_key = constants.SECRET_KEY

# Searches on the column "text" 
def text_search_cosine_similarity(df, target, n=3, pprint=True):
   embedding = get_embedding(target, engine='text-embedding-ada-002')
   df['embedding'] = df.text.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
   df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res

def audio_to_text(filename, model_selection="base",pprint=True):
    model = whisper.load_model(model_selection)
    if pprint:
        print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )

    result = model.transcribe(filename)

    # print the recognized text
    if pprint:
        print(result["text"])

    return result