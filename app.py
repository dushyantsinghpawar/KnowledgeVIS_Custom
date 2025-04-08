import streamlit as st
import pandas as pd
import torch
import numpy as np
import nltk
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import spacy
import subprocess
import umap
import io
import time

# Ensure necessary NLTK data is available
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Ensure spaCy model is available
model_name_spacy = "en_core_web_md"
try:
    nlp = spacy.load(model_name_spacy)
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", model_name_spacy])
    nlp = spacy.load(model_name_spacy)

# Streamlit page setup
st.set_page_config(page_title="KnowledgeVIS: Language Model Explorer", layout="wide")
st.title("KnowledgeVIS: Fill-in-the-Blank Language Model Explorer")

st.expander("How to Use This App").markdown("""
This app explores how different language models complete fill-in-the-blank statements.

**How to get started:**
- Use the sidebar to choose a model, pick a theme or enter your own prompts, and define subjects.
- The app will show predicted words with confidence scores and visualizations.
""")

# Preset templates with corresponding subjects and icons
preset_options = {
    "Public Health (COVID-19)": {
        "prompts": [
            "[subject] is _ to receive the COVID-19 vaccine.",
            "The COVID-19 vaccine is _ for [subject] individuals.",
            "[subject] are _ affected by long COVID symptoms.",
            "Early treatment is _ for [subject] with COVID-19.",
            "[subject] are _ to be hospitalized due to COVID-19."
        ],
        "subjects": "elderly, children, pregnant women, immunocompromised, healthcare workers, vaccinated, unvaccinated, rural residents"
    },
    "Bias in Tech": {
        "prompts": [
            "[subject] are _ hired for tech leadership roles.",
            "[subject] are _ represented in data science jobs.",
            "[subject] receive _ funding in startups.",
            "AI tools are _ for [subject] individuals.",
            "[subject] have _ access to technical education."
        ],
        "subjects": "men, women, minorities, immigrants, people with disabilities, non-native speakers"
    },
    "Education & Success": {
        "prompts": [
            "[subject] students are _ to complete college.",
            "[subject] learners find math to be _.",
            "[subject] are _ prepared for standardized testing.",
            "Tutoring is _ for [subject] students.",
            "[subject] students feel _ in online learning."
        ],
        "subjects": "first-generation, low-income, international, ESL, gifted, underrepresented"
    },
    "Custom": {
        "prompts": [],
        "subjects": ""
    }
}

# Sidebar
st.sidebar.markdown("## Settings")
debug = st.sidebar.checkbox("Enable Debug Mode", value=False)
model_name = st.sidebar.selectbox("Language Model", [
    "bert-base-uncased",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "allenai/scibert_scivocab_uncased"
])
top_k = st.sidebar.slider("Top-k Predictions", 5, 50, 10)

# Template configuration
st.sidebar.markdown("### Prompt Configuration")
selected_template = st.sidebar.selectbox("Choose a prompt theme", list(preset_options.keys()))
use_default_subjects = st.sidebar.checkbox("Use default subjects for this theme", value=True)

default_prompts = "\n".join(preset_options[selected_template]["prompts"])
default_subjects = preset_options[selected_template]["subjects"]

prompt_input = st.sidebar.text_area("Edit or enter your prompts (use [MASK] or _):", value=default_prompts, height=180)

popular_subject_groups = {
    "Health": ["elderly", "children", "pregnant women", "immunocompromised"],
    "Tech": ["men", "women", "minorities", "non-native speakers"],
    "Education": ["first-generation", "low-income", "ESL", "gifted"]
}

if use_default_subjects:
    st.sidebar.markdown("**Subjects:**")
    st.sidebar.code(default_subjects)
    subject_input = default_subjects
else:
    group = st.sidebar.selectbox("Insert popular subject group", ["None"] + list(popular_subject_groups.keys()))
    if group != "None":
        suggested_subjects = ", ".join(popular_subject_groups[group])
    else:
        suggested_subjects = default_subjects
    subject_input = st.sidebar.text_input("Custom Subjects (comma-separated)", value=suggested_subjects)

subjects = [s.strip() for s in subject_input.split(",") if s.strip()]

# Downloadable prompt set
st.sidebar.download_button("Download Current Prompts", data=prompt_input, file_name="prompts.txt", mime="text/plain")

# Load model
@st.cache_resource
def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name)
    model.eval()
    return tokenizer, model

with st.spinner(f"Loading model: {model_name}"):
    tokenizer, model = load_model(model_name)

# Parse prompts
raw_prompts = [p.strip() for p in prompt_input.splitlines() if p.strip()]
prompts, prompt_labels = [], []
full_prompts = {}  # Store the full prompts for reference

for prompt in raw_prompts:
    if "[MASK]" not in prompt and "_" not in prompt:
        st.error(f"Prompt must include a blank as [MASK] or _: '{prompt}'")
        st.stop()
    if "[subject]" in prompt:
        if not subjects:
            st.warning(f"No subjects provided for prompt: {prompt}")
        for s in subjects:
            full_prompt = prompt.replace("[subject]", s).replace("_", "[MASK]")
            prompts.append(full_prompt)
            prompt_labels.append(s)
            full_prompts[s] = full_prompt  # Store full prompt with subject
    else:
        prompts.append(prompt.replace("_", "[MASK]"))
        prompt_labels.append("Custom")
        full_prompts["Custom"] = prompt.replace("_", "[MASK]")

# Generate predictions
def get_predictions(prompt, tokenizer, model, top_k):
    inputs = tokenizer(prompt, return_tensors="pt")
    mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_pos) == 0: return []
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, mask_pos[0]]
    probs = torch.softmax(logits, dim=0)
    topk_probs, topk_indices = torch.topk(probs, top_k)
    embeddings = model.get_input_embeddings().weight.detach()
    return [(tokenizer.decode([int(idx)]).strip(), prob.item(), embeddings[int(idx)].cpu().numpy())
            for idx, prob in zip(topk_indices, topk_probs)]

predictions, embeddings = [], []
with st.spinner("Generating predictions..."):
    for prompt, label in zip(prompts, prompt_labels):
        for token, prob, emb in get_predictions(prompt.replace("[MASK]", tokenizer.mask_token), tokenizer, model, top_k):
            predictions.append({
                "Prompt": label, 
                "FullPrompt": full_prompts.get(label, prompt),
                "Prediction": token, 
                "Probability": prob
            })
            embeddings.append(emb)

# Create dataframe and cluster
df = pd.DataFrame(predictions)
if df.empty:
    st.error("No predictions generated.")
    st.stop()

@st.cache_data
def get_wordnet_definition(word):
    synsets = wn.synsets(word)
    if synsets:
        return synsets[0].definition()
    return "No WordNet definition found."

df["Definition"] = df["Prediction"].apply(get_wordnet_definition)

def wu_palmer_distance(w1, w2):
    s1, s2 = wn.synsets(w1), wn.synsets(w2)
    if not s1 or not s2: return 1.0
    sim = s1[0].wup_similarity(s2[0])
    return 1.0 - (sim or 0.0)

def cluster_words(words, threshold=0.4):
    if len(words) <= 1: return {w: 1 for w in words}
    try:
        dists = pdist([[w] for w in words], lambda u, v: wu_palmer_distance(u[0], v[0]))
        Z = linkage(dists, method="ward")
        return dict(zip(words, fcluster(Z, threshold, criterion="distance")))
    except:
        return {word: i+1 for i, word in enumerate(words)}

df["Cluster"] = df["Prediction"].map(cluster_words(df["Prediction"].unique()))

# Generate cluster colors
unique_clusters = df["Cluster"].unique()
cluster_colors = {cluster: px.colors.qualitative.Dark24[i % len(px.colors.qualitative.Dark24)] 
                 for i, cluster in enumerate(unique_clusters)}

# Visuals
st.subheader(f"Theme: {selected_template}")

# Optional color palette per theme
color_themes = {
    "COVID-19": "Blues",
    "Bias": "Purples",
    "Education": "Greens",
    "Custom": "Greys"
}

selected_palette = "Blues"  # default
for key in color_themes:
    if key in selected_template:
        selected_palette = color_themes[key]
        break

# Heatmap
st.header("Heatmap of Top Predictions")
heat_df = pd.concat([df[df["Prompt"] == p].nlargest(5, "Probability") for p in df["Prompt"].unique()])
pivot = heat_df.pivot_table(index="Prompt", columns="Prediction", values="Probability", fill_value=0)
fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                color_continuous_scale=selected_palette,
                labels=dict(x="Prediction", y="Prompt", color="Probability"),
                title="Top Prediction Heatmap")
st.plotly_chart(fig, use_container_width=True)


# Tag Cloud with WordNet definitions
st.header("Tag Cloud View")

fig_tag = go.Figure()
x_pos = np.linspace(0.1, 0.9, len(subjects))
for i, subj in enumerate(subjects):
    subj_df = df[df["Prompt"] == subj]
    is_predefined = subj in subjects
    for _, row in subj_df.iterrows():
        x_jitter = random.uniform(-0.03, 0.03)
        y_pos = random.uniform(0.1, 0.9)
        size = max(10, min(30, 10 + 40 * row["Probability"]))
        hover_text = f"<b>{row['Prediction']}</b><br>p={row['Probability']:.3f}<br>{row['Definition']}"
        fig_tag.add_trace(go.Scatter(
            x=[x_pos[i] + x_jitter],
            y=[y_pos],
            text=row["Prediction"],
            mode="text",
            textfont=dict(size=size, color=cluster_colors.get(row["Cluster"], "gray")),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False
        ))
    fig_tag.add_annotation(
        x=x_pos[i], y=0.02, 
        text=subj + ('*' if is_predefined else ''),
        showarrow=False,
        font=dict(size=14, color='black' if is_predefined else 'dimgray'),
        xanchor="center", yanchor="bottom"
    )

fig_tag.update_layout(
    title="Tag Cloud by Subject (Predefined subjects marked with *)",
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False)
)
st.plotly_chart(fig_tag, use_container_width=True)


# Embedding Method Choice
embedding_method = st.sidebar.radio("Choose dimensionality reduction method:", ["PCA", "UMAP"])


if len(embeddings) >= 3:
    embedding_matrix = np.array(embeddings)
    coords = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(embedding_matrix) if embedding_method == "UMAP" else PCA(n_components=2).fit_transform(embedding_matrix)
    viz_df = df.copy()
    viz_df["x"], viz_df["y"] = coords[:, 0], coords[:, 1]
    fig_scatter = px.scatter(
        viz_df,
        x="x", y="y",
        color="Prompt",
        size="Probability",
        text="Prediction",
        hover_data=["Prediction", "Prompt", "Probability", "Definition", "FullPrompt"],
        title=f"{embedding_method} of Prediction Embeddings",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning("Not enough embeddings for dimensionality reduction.")

# Word Trajectory
st.header("Word Trajectory Across Prompts")
target_word = st.text_input("Track prediction probability of word:", "safe")

if target_word:
    traj_data = df[df["Prediction"] == target_word].copy()
    
    # Check if we have trajectory data
    if not traj_data.empty:
        fig_traj = px.line(
            traj_data,
            x="Prompt",
            y="Probability",
            markers=True,
            hover_data={"Prompt": True, "Probability": True, "FullPrompt": True},
            title=f"Trajectory of '{target_word}' Across Prompts"
        )
        fig_traj.update_traces(
            mode="lines+markers", 
            hovertemplate="<b>%{customdata[2]}</b><br>Probability=%{y:.3f}"
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.warning(f"No predictions for '{target_word}' found in the current dataset.")

# Download Data
st.download_button("Download Prediction Data (CSV)", data=df.to_csv(index=False), file_name="knowledgevis_predictions.csv", mime="text/csv")

# Export HTML
if st.button("Export HTML Report"):
    buffer = io.StringIO()
    df.to_html(buf=buffer, index=False)
    html_bytes = buffer.getvalue().encode()
    st.download_button("Download HTML Report", data=html_bytes, file_name="KnowledgeVIS_Report.html", mime="text/html")

# Runtime Log
st.caption(f"⏱️ Page generated in {time.time():.2f} seconds.")

# Optional Footer or Attribution
st.markdown("""
---
**Created by:** Dushyant Singh Pawar  
**Powered by:** [HuggingFace Transformers](https://huggingface.co/), [spaCy](https://spacy.io/), [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/)  
**Inspired by:** *KnowledgeVIS: Interpreting Language Models by Comparing Fill-in-the-Blank Prompts*  
**DOI:** [10.1109/TVCG.2023.3346713](https://doi.org/10.1109/TVCG.2023.3346713)
""")