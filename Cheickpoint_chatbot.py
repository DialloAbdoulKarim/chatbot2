import os
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Configuration du chemin pour le fichier texte
file_path = r"questions_reponses_data_science.txt"

# Téléchargement des stopwords français
nltk.download('stopwords')
stopwords_fr = set(stopwords.words('french'))

# Charger le modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

# Prétraitement du texte avec spaCy
def preprocess(text):
    doc = nlp(text)
    words = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stopwords_fr]
    return " ".join(words)

# Fonction pour charger et prétraiter le fichier texte
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
    
    questions, answers, processed_questions = [], [], []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
                questions.append(question)
                processed_questions.append(preprocess(question))
            elif line.startswith("Réponse:"):
                answer = line.replace("Réponse:", "").strip()
                answers.append(answer)
    return questions, processed_questions, answers

# Fonction pour trouver la réponse la plus pertinente
def find_best_response(user_query, questions, processed_questions, answers):
    user_query_processed = preprocess(user_query)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    query_tfidf = vectorizer.transform([user_query_processed])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    best_match_index = similarities.argmax()
    
    if similarities[best_match_index] > 0:
        return answers[best_match_index]
    return "Je ne sais pas répondre à cette question."

# Interface utilisateur Streamlit
def chatbot():
    st.title("Chatbot Data Science")
    st.write("Posez une question et obtenez une réponse pertinente !")

    try:
        questions, processed_questions, answers = load_data(file_path)
        st.success("Fichier chargé avec succès ! Posez votre question ci-dessous.")
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")
        return

    # Interface pour poser des questions
    user_query = st.text_input("Votre question :")
    if user_query:
        response = find_best_response(user_query, questions, processed_questions, answers)
        st.write(f"**Réponse :** {response}")

if __name__ == "__main__":
    chatbot()

           
