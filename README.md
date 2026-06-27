# 🤖 RAG Document Assistant — Chatbot sur vos Documents PDF

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-00599C?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)

> Application de **Retrieval-Augmented Generation (RAG)** : uploadez un PDF et interrogez-le en langage naturel. Construit avec LangChain, FAISS et GPT-3.5.

---

## 🎯 Fonctionnalités

- 📄 Upload de documents PDF
- 🔍 Indexation vectorielle automatique (FAISS)
- 💬 Interface conversationnelle avec historique
- 📌 Citation des sources (numéros de page)
- 🔒 Clé API gérée localement (`.env`)

---

## ⚙️ Installation & lancement

```bash
pip install -r requirements.txt

# Créer un fichier .env
echo "OPENAI_API_KEY=sk-votre-cle-ici" > .env

# Lancer l'application
streamlit run app.py
```

---

## 🏗️ Architecture RAG

```
PDF → Découpage en chunks → Embeddings OpenAI → FAISS Index
                                                      ↓
Question utilisateur → Embeddings → Recherche similarité → Contexte
                                                              ↓
                                              GPT-3.5 → Réponse citée
```

---

## 💡 Cas d'usage

- Analyser des rapports financiers / rapports annuels
- Interroger des contrats ou documents légaux
- Extraire des insights de publications de recherche

---

## 👤 Auteur

**Harry TEGUE** — Data Analyst | GenAI Engineering
Certifié IBM Generative AI Engineer (Coursera, 2026)
