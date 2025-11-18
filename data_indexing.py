# data_indexing.py
"""
Data cleaning, Pinecone index creation, and vector upserting.
Run this script to populate the Pinecone index with career data.
"""

import os
import time
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv(".env")

# Load API keys
openapi_key = os.environ.get("OPENAI_API_KEY")
pineconeapi_key = os.environ.get("PINECONE_API_KEY")

# Initialize clients
INDEX_NAME = "entertainment-careers"
emb = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone(api_key=pineconeapi_key)


def create_index_if_needed():
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,  # text-embedding-3-large dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
        print("Index created. Waiting 10 seconds...")
        time.sleep(10)
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the roles dataset."""
    roles_df = pd.read_csv(csv_path, encoding='utf-8-sig').fillna("")
    
    # Ensure required columns exist
    required_cols = [
        "Title", "SOC_Code", "Cluster", "Description", "Skills", "Tools", 
        "Education_Level", "Experience_Needed", "Training", "Certifications", 
        "Salary_Range", "Resources", "R", "I", "A", "S", "E", "C"
    ]
    
    missing = [c for c in required_cols if c not in roles_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    print(f"Loaded {len(roles_df)} roles from {csv_path}")
    return roles_df


def build_text(row: pd.Series) -> str:
    """Build text for embedding from a role row."""
    return (
        f"Title: {row['Title']}. "
        f"Description: {row['Description']}. "
        f"Skills: {row['Skills']}. "
        f"Tools: {row['Tools']}. "
        f"Education: {row['Education_Level']}. "
        f"Experience: {row['Experience_Needed']}. "
        f"Training: {row['Training']}. "
        f"Certifications: {row['Certifications']}. "
        f"Salary Range: {row['Salary_Range']}. "
        f"Resources: {row['Resources']}. "
        f"Cluster: {row['Cluster']}."
    )


def embed_and_prepare_vectors(roles_df: pd.DataFrame) -> list:
    """Generate embeddings and prepare vectors for upserting."""
    vectors = []
    
    for i, row in roles_df.iterrows():
        text = build_text(row)
        vector = emb.embed_query(text)
        
        metadata = {
            "Title": row["Title"],
            "SOC_Code": row["SOC_Code"],
            "Cluster": row["Cluster"],
            "Description": row["Description"],
            "Skills": row["Skills"],
            "Tools": row["Tools"],
            "Education_Level": row["Education_Level"],
            "Experience_Needed": row["Experience_Needed"],
            "Training": row["Training"],
            "Certifications": row["Certifications"],
            "Salary_Range": row["Salary_Range"],
            "Resources": row["Resources"],
            # RIASEC as separate keys
            "R": int(row["R"]),
            "I": int(row["I"]),
            "A": int(row["A"]),
            "S": int(row["S"]),
            "E": int(row["E"]),
            "C": int(row["C"])
        }
        
        vectors.append({
            "id": f"{row['Title']}_{i}",
            "values": vector,
            "metadata": metadata
        })
    
    print(f"Prepared {len(vectors)} vectors for upserting")
    return vectors


def upsert_vectors(vectors: list, batch_size: int = 80):
    """Upsert vectors to Pinecone in batches."""
    index = pc.Index(INDEX_NAME)
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted {i + len(batch)} / {len(vectors)}")
    
    print("Indexing complete.")


def main():
    """Main function to orchestrate data loading, index creation, and upserting."""
    csv_path = "entertainment_roles_riasec.csv"
    
    # Step 1: Create index if needed
    create_index_if_needed()
    
    # Step 2: Load and clean data
    roles_df = load_and_clean_data(csv_path)
    
    # Step 3: Embed and prepare vectors
    vectors = embed_and_prepare_vectors(roles_df)
    
    # Step 4: Upsert to Pinecone
    upsert_vectors(vectors)


if __name__ == "__main__":
    main()
