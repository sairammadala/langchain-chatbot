import os
from dotenv import load_dotenv; load_dotenv()

# show what we're about to use
print("ENV  =", repr(os.getenv("PINECONE_ENV")))
print("INDEX=", repr(os.getenv("PINECONE_INDEX")))
print("KEY startswith pcsk_ =", (os.getenv("PINECONE_API_KEY") or "").startswith("pcsk_"))

import pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY")
    #environment=os.getenv("PINECONE_ENV"),
)
print("SDK list =", pinecone.list_indexes())
