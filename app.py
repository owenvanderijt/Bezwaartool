
import os
import PyPDF2
import tempfile
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Functie om tekst uit een PDF te halen
def pdf_naar_tekst(pad):
    with open(pad, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        tekst = ''
        for pagina in reader.pages:
            tekst += pagina.extract_text()
        return tekst

# Functie om database-PDF's in te laden
def laad_database(pad_naar_map):
    database = {}
    for bestandsnaam in os.listdir(pad_naar_map):
        if bestandsnaam.endswith('.pdf'):
            volledige_pad = os.path.join(pad_naar_map, bestandsnaam)
            tekst = pdf_naar_tekst(vollede_pad)
            database[bestandsnaam] = tekst
    return database

# Vergelijk nieuwe bezwaargrond met database
def vergelijk_nieuwe_met_database(nieuwe_tekst, database):
    if not database:
        return [], "⚠️ De databank is leeg. Voeg eerst PDF's toe via de linkerbovenknop."

    teksten = [nieuwe_tekst] + list(database.values())
    vectorizer = TfidfVectorizer().fit_transform(teksten)
    vectors = vectorizer.toarray()

    overeenkomsten = cosine_similarity([vectors[0]], vectors[1:])[0]
    scores = list(zip(database.keys(), overeenkomsten))
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores, None

# Trek conclusie op basis van vergelijkingsscores
def analyseer_conclusie(scores, drempel=0.75):
    if not scores:
        return "⚠️ Geen vergelijkbare documenten gevonden in de databank."

    naam, hoogste_score = scores[0]
    if hoogste_score > drempel:
        return f"✅ Deze bezwaargrond lijkt sterk op '{naam}' met een overeenkomst van {round(hoogste_score * 100, 2)}%. Mogelijk een standaardgrond."
    else:
        return f"❗ Geen sterke overeenkomst gevonden. Mogelijk een inhoudelijk nieuwe grond."

# Streamlit interface
st.set_page_config(page_title="Bezwaargrond Vergelijker", layout="centered")
st.title("📝 Bezwaargrond Vergelijker")

# Knoppen en upload
st.subheader("➕ Voeg toe aan databank")
pdf_databank = st.file_uploader("Upload hier een PDF met bezwaargrond om toe te voegen aan de databank", type="pdf", key="upload1")
if pdf_databank:
    save_path = os.path.join("data", pdf_databank.name)
    with open(save_path, "wb") as f:
        f.write(pdf_databank.read())
    st.success(f"✅ Bestand '{pdf_databank.name}' is toegevoegd aan de databank.")

st.markdown("---")

st.subheader("🔍 Vergelijk nieuwe bezwaargrond met databank")
pdf_vergelijk = st.file_uploader("Upload hier een nieuwe bezwaargrond (PDF)", type="pdf", key="upload2")
if pdf_vergelijk:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_vergelijk.read())
        nieuw_pad = tmp.name

    nieuwe_tekst = pdf_naar_tekst(nieuw_pad)
    database = laad_database("data/")
    scores, foutmelding = vergelijk_nieuwe_met_database(nieuwe_tekst, database)

    st.subheader("🧠 Analyse")
    if foutmelding:
        st.warning(foutmelding)
    else:
        conclusie = analyseer_conclusie(scores)
        st.write(conclusie)

        st.subheader("📊 Top 3 overeenkomsten:")
        for naam, score in scores[:3]:
            st.write(f"- {naam}: {round(score * 100, 2)}% overeenkomst")
