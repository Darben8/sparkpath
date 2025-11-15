# app.py
import os
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy.linalg import norm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import LLMChain
#from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(".env")
# Load API keys from environment variables
openapi_key = os.environ.get("OPENAI_API_KEY")
pineconeapi_key = os.environ.get("PINECONE_API_KEY")

# Init clients
INDEX_NAME = "entertainment-careers"
emb = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone(api_key=pineconeapi_key)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openapi_key, temperature=0.3)

# # Create index if needed
# existing_indexes = [i["name"] for i in pc.list_indexes()]

# if INDEX_NAME not in existing_indexes:
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=3072,                 # text-embedding-3-large dimension
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         ),
#     )
#     print("Creating index... waiting 10s")
#     time.sleep(10)

index = pc.Index(INDEX_NAME)

# Load roles dataset
#roles_df = pd.read_csv("entertainment_roles_riasec.csv", encoding='utf-8-sig').fillna("")

# roles_df = pd.read_csv("entertainment_roles_riasec.csv")
# roles_df = roles_df.fillna("")

# # Ensure required columns exist
# required_cols = [
#     "Title","SOC_Code","Cluster","Description","Skills","Tools","Education_Level",
#     "Experience_Needed","Training","Certifications","Salary_Range","Resources",
#     "R","I","A","S","E","C"
# ]

# missing = [c for c in required_cols if c not in roles_df.columns]
# if missing:
#     raise ValueError(f"Missing columns: {missing}")

# def format_riasec(row):
#     return (
#         f"R:{int(row['R'])},"
#         f"I:{int(row['I'])},"
#         f"A:{int(row['A'])},"
#         f"S:{int(row['S'])},"
#         f"E:{int(row['E'])},"
#         f"C:{int(row['C'])}"
#     )

# #build text for embedding
# def build_text(row):
#     return (
#         f"Title: {row['Title']}. "
#         f"Description: {row['Description']}. "
#         f"Skills: {row['Skills']}. "
#         f"Tools: {row['Tools']}. "
#         f"Education: {row['Education_Level']}. "
#         f"Experience: {row['Experience_Needed']}. "
#         f"Training: {row['Training']}. "
#         f"Certifications: {row['Certifications']}. "
#         f"Salary Range: {row['Salary_Range']}. "
#         f"Resources: {row['Resources']}. "
#         f"Cluster: {row['Cluster']}."
#     )
# # Embed and upsert roles
# vectors = []

# for i, row in roles_df.iterrows():
#     text = build_text(row)
#     vector = emb.embed_query(text)

#     metadata = {
#         "Title": row["Title"],
#         "SOC_Code": row["SOC_Code"],
#         "Cluster": row["Cluster"],
#         "Description": row["Description"],
#         "Skills": row["Skills"],
#         "Tools": row["Tools"],
#         "Education_Level": row["Education_Level"],
#         "Experience_Needed": row["Experience_Needed"],
#         "Training": row["Training"],
#         "Certifications": row["Certifications"],
#         "Salary_Range": row["Salary_Range"],
#         "Resources": row["Resources"],
#         # RIASEC separate keys
#         "R": int(row["R"]),
#         "I": int(row["I"]),
#         "A": int(row["A"]),
#         "S": int(row["S"]),
#         "E": int(row["E"]),
#         "C": int(row["C"])
#     }

#     vectors.append({
#         "id": f"{row['Title']}_{i}",
#         "values": vector,
#         "metadata": metadata
#     })

# #Chunk uploads (Pinecone limit ≈ 100 vectors per batch recommended)
# BATCH = 80
# for i in range(0, len(vectors), BATCH):
#     batch = vectors[i:i+BATCH]
#     index.upsert(vectors=batch)
#     print(f"Upserted {i + len(batch)} / {len(vectors)}")

# print("Indexing complete.")

#RIASEC quiz questions (simplified)
quiz_questions = {
    "Do you enjoy working with your hands or technical equipment?": "R",
    "Do you enjoy analyzing and solving complex problems?": "I",
    "Do you enjoy creating art, music, or performances?": "A",
    "Do you enjoy helping and interacting with people?": "S",
    "Do you enjoy leading projects or persuading others?": "E",
    "Do you enjoy organizing, planning, and following rules?": "C"
}

st.title("🎤 SparkPath - Find Your Entertainment Career")

# Collect user answers
st.subheader("Personality Quiz (RIASEC)")
if "user_scores" not in st.session_state:
    st.session_state.user_scores = {k: 0 for k in "RIASEC"}

for question, key in quiz_questions.items():
    answer = st.radio(question, ["Not at all", "Sometimes", "Often"])
    st.session_state.user_scores[key] = {"Not at all": 0, "Sometimes": 1, "Often": 2}[answer]

user_scores = st.session_state.user_scores

#st.write("Your RIASEC Scores:", user_scores)
#map letters to full names
riasec_fullnames = {
    "R": "Realistic Careers",
    "I": "Investigative Careers",
    "A": "Artistic Careers",
    "S": "Social Careers",
    "E": "Enterprising Careers",
    "C": "Conventional Careers"
}

# Create display-friendly dictionary
user_scores_display = {riasec_fullnames[k]: v for k, v in user_scores.items()}

# Show scores table
st.write("### Your RIASEC Scores:")
st.write(user_scores_display)

# # Bar chart
# df_scores = pd.DataFrame.from_dict(user_scores_display, orient="index", columns=["Score"])
# df_scores = df_scores.sort_values("Score", ascending=True)  # optional: sort for chart
# fig, ax = plt.subplots(figsize=(8, 4))
# df_scores.plot(kind="barh", legend=False, ax=ax, color="skyblue")
# ax.set_xlabel("Score")
# ax.set_ylabel("")
# ax.set_title("Your RIASEC Personality Scores")
# ax.grid(axis="x", linestyle="--", alpha=0.7)
# st.pyplot(fig)

st.info(
    """
    The RIASEC model matches your personality and interests to six career types:
    - **Realistic (R):** Hands-on, technical, or practical work
    - **Investigative (I):** Analytical, scientific, or problem-solving tasks
    - **Artistic (A):** Creative work, arts, design, or expression
    - **Social (S):** Helping, teaching, or interacting with people
    - **Enterprising (E):** Leadership, persuasion, or business-oriented tasks
    - **Conventional (C):** Organized, structured, and rule-following roles
    """
)

st.subheader("Tell us about your interests and passions in entertainment")
user_text = st.text_area("Write a few sentences about what excites you:")

# --- EMBEDDING USER PROFILE ---
def get_user_embedding(text, riasec_scores):
    # Combine free-text and RIASEC
    profile_text = text + " | RIASEC: " + str(riasec_scores)
    return embeddings.embed_query(profile_text)
    
# --- Button Callback ---
if st.button("Find My Top Careers"):

    with st.spinner("Analyzing your profile..."):

        # 1. Get user embedding
        user_vector = get_user_embedding(user_text, user_scores)

        # 2. Query Pinecone
        query_results = index.query(
            vector=user_vector,
            top_k=10,
            include_metadata=True
        )
        # --- COSINE SIMILARITY
        def cosine_sim(v1, v2):
            return np.dot(v1, v2) / (norm(v1) * norm(v2))

        # 3. Combine with RIASEC similarity
        scored_roles = []
        for match in query_results["matches"]:
            meta = match["metadata"]
            riasec_vec = np.array([int(meta[k]) for k in "RIASEC"])
            user_vec = np.array([user_scores[k] for k in "RIASEC"])
            sim_score = cosine_sim(riasec_vec, user_vec)
            scored_roles.append({**meta, "riasec_score": sim_score, "pinecone_score": match["score"]})

        # 4. Combined ranking (50/50)
        for r in scored_roles:
            r["combined_score"] = 0.5*r["pinecone_score"] + 0.5*r["riasec_score"]
        top_roles = sorted(scored_roles, key=lambda x: x["combined_score"], reverse=True)[:3]

        # 5. Generate LLM reasoning
        prompt_template = """
        You are a friendly career coach. Suggest why the following role is a good match 
        for a user based on their RIASEC scores {riasec_scores} and interests: {user_text}.
        Return a short, clear reasoning snippet (2-3 sentences max).
        Role: {role_title} ({role_cluster})
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["riasec_scores", "user_text", "role_title", "role_cluster"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        
        top_careers = []
        for r in top_roles:
            reasoning = chain.run(
                riasec_scores=user_scores,
                user_text=user_text,
                role_title=r['Title'],
                role_cluster=r['Cluster']
            ).strip()
            top_careers.append({**r, "Reasoning": reasoning})

        # --- UTILITY FUNCTIONS ---
        def clean_text(text):
            """Fix common encoding artifacts and normalize text."""
            if not text or pd.isna(text):
                return ""
            text = str(text)
            # Fix common mojibake
            text = text.replace("â€™", "'")  # apostrophe
            text = text.replace("â€“", "-")  # en-dash
            text = text.replace("â€”", "-")  # em-dash
            text = text.replace("\xa0", " ") # non-breaking space
            text = text.strip()
            return text

        def format_career_field(career):
            """Format salary and education for display, fixing encoding issues."""

            # --- Salary ---
            salary = clean_text(career.get("Salary_Range", ""))
            if not salary:
                salary_str = "N/A"
            else:
                # Keep only digits, commas, hyphens
                salary_clean = re.sub(r"[^\d\-,]", "", salary)
                # Split ranges on hyphen
                parts = [p.strip() for p in salary_clean.split("-") if p.strip()]
                try:
                    if len(parts) == 2:
                        salary_str = f"${int(parts[0].replace(',','')):,} - ${int(parts[1].replace(',','')):,}"
                    elif len(parts) == 1:
                        salary_str = f"${int(parts[0].replace(',','')):,}"
                    else:
                        salary_str = "N/A"
                except:
                    salary_str = "N/A"

            # --- Education ---
            edu = clean_text(career.get("Education_Level", ""))
            if not edu:
                edu_str = "N/A"
            else:
                edu_lower = edu.lower()
                if any(k in edu_lower for k in ["bachelor", "bsc", "bs"]):
                    edu_str = "Bachelor's Degree"
                elif any(k in edu_lower for k in ["master", "msc", "ms"]):
                    edu_str = "Master's Degree"
                elif any(k in edu_lower for k in ["phd", "doctor"]):
                    edu_str = "Doctorate"
                elif any(k in edu_lower for k in ["high school", "secondary"]):
                    edu_str = "High School Diploma"
                elif "associate" in edu_lower:
                    edu_str = "Associate Degree"
                else:
                    edu_str = edu.title()

            return salary_str, edu_str
                
        # Store careers in session state
        st.session_state.top_careers = top_careers

# ------------------ DISPLAY TOP CAREERS ------------------
# Show careers if they exist in session state
if "top_careers" in st.session_state and st.session_state.top_careers:
    st.subheader("🌟 Top Career Matches for You")
    for i, career in enumerate(st.session_state.top_careers, 1):
        with st.expander(f"{i}. {career['Title']} ({career['Cluster']})"):
            # --- Cleaned fields ---
            def clean_text(text):
                """Fix common encoding artifacts and normalize text."""
                if not text or pd.isna(text):
                    return ""
                text = str(text)
                # Fix common mojibake
                text = text.replace("\u2019", "'")  # apostrophe
                text = text.replace("\u2013", "-")  # en-dash
                text = text.replace("\u2014", "-")  # em-dash
                text = text.replace("\xa0", " ")  # non-breaking space
                text = text.strip()
                return text

            def format_career_field(career):
                """Format salary and education for display, fixing encoding issues."""

                # --- Salary ---
                salary = clean_text(career.get("Salary_Range", ""))
                if not salary:
                    salary_str = "N/A"
                else:
                    # Keep only digits, commas, hyphens
                    salary_clean = re.sub(r"[^\d\-,]", "", salary)
                    # Split ranges on hyphen
                    parts = [p.strip() for p in salary_clean.split("-") if p.strip()]
                    try:
                        if len(parts) == 2:
                            salary_str = f"${int(parts[0].replace(',','')):,} - ${int(parts[1].replace(',','')):,}"
                        elif len(parts) == 1:
                            salary_str = f"${int(parts[0].replace(',','')):,}"
                        else:
                            salary_str = "N/A"
                    except:
                        salary_str = "N/A"

                # --- Education ---
                edu = clean_text(career.get("Education_Level", ""))
                if not edu:
                    edu_str = "N/A"
                else:
                    edu_lower = edu.lower()
                    if any(k in edu_lower for k in ["bachelor", "bsc", "bs"]):
                        edu_str = "Bachelor's Degree"
                    elif any(k in edu_lower for k in ["master", "msc", "ms"]):
                        edu_str = "Master's Degree"
                    elif any(k in edu_lower for k in ["phd", "doctor"]):
                        edu_str = "Doctorate"
                    elif any(k in edu_lower for k in ["high school", "secondary"]):
                        edu_str = "High School Diploma"
                    elif "associate" in edu_lower:
                        edu_str = "Associate Degree"
                    else:
                        edu_str = edu.title()

                return salary_str, edu_str
            
            salary_str, edu_str = format_career_field(career)
            
            st.write(f"**Why this fits:** {career['Reasoning']}")
            st.write(f"**Skills:** {career.get('Skills','N/A')}")
            st.write(f"**Tools:** {career.get('Tools','N/A')}")
            st.write(f"**Education:** {edu_str}")
            st.write(f"**Salary Range:** {salary_str}")
            st.write(f"**Resources:** {career.get('Resources','N/A')}")

# ------------------ FOLLOW-UP CHAT ------------------
# Only show chat if careers have been found
if "top_careers" in st.session_state and st.session_state.top_careers:
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Ask Follow-Up Questions")
    
    # Display chat history first
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Coach:** {msg['content']}")
    
    # Input
    follow_question = st.text_input("Ask about a career, skill, or next step:")

    # Button required to prevent auto-refresh
    if st.button("Send"):
        if follow_question.strip():

            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": follow_question})

            # Build conversation string
            history_text = ""
            for msg in st.session_state.chat_history:
                prefix = "User:" if msg["role"]=="user" else "Coach:"
                history_text += f"{prefix} {msg['content']}\n"

            # Retrieve stored top careers
            top_list = st.session_state.get("top_careers", [])

            # LLM prompt with conversation + career context
            titles = ', '.join([c['Title'] for c in top_list])
            prompt_text = """
            You are a career coach responding to user questions.
            Top career matches are : {titles}.

            Conversation:
            {history_text}

            Respond as Coach:
            """

            # Generate response
            prompt = PromptTemplate(
                template=prompt_text,
                input_variables=["titles", "history_text"]
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            assistant_reply = chain.run(
                titles=', '.join([c['Title'] for c in top_list]),
                history_text=history_text
            ).strip()

            # Store assistant reply
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
            # st.markdown(f"**Coach:** {assistant_reply}")
            
            # Rerun to display the new message
            st.rerun()