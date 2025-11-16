import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MedicalChatbot:
    def __init__(self):
        self.dataset = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.nn_model = None
        self.medical_entities = {
            'symptoms': [],
            'diseases': [],
            'treatments': [],
            'medications': []
        }
        
    def load_medquad_data(self):
        """Load MedQuAD dataset from GitHub or local files"""
        try:
            # Try to load from local file first
            if os.path.exists('medquad_sample.csv'):
                self.dataset = pd.read_csv('medquad_sample.csv')
                st.success("Loaded MedQuAD dataset from local file")
                return True
                
            # If local file doesn't exist, create a sample dataset
            st.info("Creating sample medical Q&A dataset...")
            self.create_sample_dataset()
            return True
            
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False
    
    def create_sample_dataset(self):
        """Create a sample medical Q&A dataset"""
        sample_data = {
            'question': [
                "What are the symptoms of diabetes?",
                "How is hypertension treated?",
                "What causes asthma?",
                "What are the early signs of cancer?",
                "How to prevent heart disease?",
                "What is the treatment for pneumonia?",
                "What are the symptoms of COVID-19?",
                "How is arthritis diagnosed?",
                "What medications are used for depression?",
                "What are the risk factors for stroke?",
                "How to manage migraine headaches?",
                "What is the prognosis for Alzheimer's disease?",
                "What are the complications of diabetes?",
                "How is bronchitis treated?",
                "What are the symptoms of thyroid problems?"
            ],
            'answer': [
                "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, and slow healing of wounds. Type 1 and type 2 diabetes may present with similar symptoms.",
                "Hypertension treatment typically includes lifestyle changes such as reducing salt intake, regular exercise, weight management, and medications like ACE inhibitors, beta-blockers, or diuretics as prescribed by a doctor.",
                "Asthma is caused by a combination of genetic and environmental factors. Common triggers include allergens, respiratory infections, physical activity, cold air, stress, and certain medications.",
                "Early signs of cancer may include unexplained weight loss, persistent fatigue, lumps or thickening, changes in bowel or bladder habits, persistent cough, or unusual bleeding. Regular screenings are important.",
                "Heart disease prevention involves maintaining a healthy diet low in saturated fats, regular physical activity, not smoking, managing stress, controlling blood pressure and cholesterol levels, and maintaining a healthy weight.",
                "Pneumonia treatment depends on the cause. Bacterial pneumonia is treated with antibiotics, viral pneumonia may require antiviral medications, and fungal pneumonia needs antifungal drugs. Rest and fluids are also important.",
                "COVID-19 symptoms include fever, cough, shortness of breath, fatigue, muscle aches, loss of taste or smell, sore throat, and headache. Severe cases may lead to pneumonia and respiratory distress.",
                "Arthritis diagnosis involves physical examination, review of symptoms, blood tests for inflammatory markers, imaging tests like X-rays or MRI, and sometimes joint fluid analysis.",
                "Depression medications include SSRIs (like fluoxetine, sertraline), SNRIs (like venlafaxine, duloxetine), atypical antidepressants, and in severe cases, MAOIs or tricyclic antidepressants. Therapy is often combined with medication.",
                "Stroke risk factors include high blood pressure, smoking, diabetes, high cholesterol, obesity, physical inactivity, family history, age, and certain heart conditions like atrial fibrillation.",
                "Migraine management includes identifying triggers, medications like triptans or NSAIDs for acute attacks, preventive medications, lifestyle modifications, stress management, and adequate sleep.",
                "Alzheimer's disease prognosis varies but is generally progressive. Early diagnosis and treatment can help manage symptoms. Average life expectancy after diagnosis is 4-8 years, but some live 20 years or more.",
                "Diabetes complications include cardiovascular disease, nerve damage, kidney damage, eye damage, foot problems, skin conditions, hearing impairment, and increased susceptibility to infections.",
                "Bronchitis treatment focuses on symptom relief: rest, fluids, humidifiers, cough suppressants for dry cough, and sometimes bronchodilators or corticosteroids for inflammation. Antibiotics are only for bacterial infections.",
                "Thyroid problem symptoms vary: hyperthyroidism may cause weight loss, rapid heartbeat, anxiety; hypothyroidism may cause weight gain, fatigue, depression; both can cause goiter and changes in menstrual patterns."
            ]
        }
        
        self.dataset = pd.DataFrame(sample_data)
        self.dataset.to_csv('medquad_sample.csv', index=False)
        st.success("Created sample medical Q&A dataset")
    
    def preprocess_text(self, text):
        """Preprocess text for similarity matching"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def train_model(self):
        """Train the retrieval model"""
        if self.dataset is None:
            st.error("No dataset loaded")
            return False
        
        try:
            # Preprocess questions
            processed_questions = self.dataset['question'].apply(self.preprocess_text)
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)
            
            # Train Nearest Neighbors model
            self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.nn_model.fit(self.tfidf_matrix)
            
            # Extract medical entities
            self.extract_medical_entities()
            
            st.success("Model trained successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False
    
    def extract_medical_entities(self):
        """Extract basic medical entities from the dataset"""
        medical_keywords = {
            'symptoms': ['symptom', 'sign', 'pain', 'fever', 'cough', 'fatigue', 'headache', 'nausea'],
            'diseases': ['diabetes', 'hypertension', 'asthma', 'cancer', 'pneumonia', 'arthritis', 'depression'],
            'treatments': ['treatment', 'therapy', 'medication', 'surgery', 'exercise', 'diet'],
            'medications': ['antibiotic', 'antiviral', 'antidepressant', 'inhibitor', 'blocker']
        }
        
        for category, keywords in medical_keywords.items():
            entities = set()
            for text in self.dataset['question'] + self.dataset['answer']:
                if isinstance(text, str):
                    for keyword in keywords:
                        if keyword in text.lower():
                            entities.add(keyword)
            self.medical_entities[category] = list(entities)
    
    def find_similar_questions(self, query, k=3):
        """Find similar questions to the user query"""
        if self.nn_model is None or self.vectorizer is None:
            return []
        
        try:
            # Preprocess query
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            
            # Find similar questions
            distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                similarity = 1 - dist  # Convert distance to similarity
                results.append({
                    'question': self.dataset.iloc[idx]['question'],
                    'answer': self.dataset.iloc[idx]['answer'],
                    'similarity': similarity
                })
            
            return results
            
        except Exception as e:
            st.error(f"Error finding similar questions: {e}")
            return []
    
    def recognize_medical_entities(self, text):
        """Recognize medical entities in the text"""
        entities_found = {}
        
        for category, entities in self.medical_entities.items():
            found = []
            for entity in entities:
                if entity in text.lower():
                    found.append(entity)
            if found:
                entities_found[category] = found
        
        return entities_found

def main():
    st.title("🏥 Medical Q&A Chatbot")
    st.markdown("Ask medical questions and get informed answers based on the MedQuAD dataset.")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalChatbot()
        st.session_state.chatbot.load_medquad_data()
        st.session_state.chatbot.train_model()
        st.session_state.messages = []
        st.session_state.show_entities = True
    
    # Sidebar
    with st.sidebar:
        st.header("Medical Chatbot Settings")
        
        st.subheader("Dataset Info")
        if st.session_state.chatbot.dataset is not None:
            st.write(f"Total Q&A pairs: {len(st.session_state.chatbot.dataset)}")
            st.write("Sample questions:")
            for i, q in enumerate(st.session_state.chatbot.dataset['question'].head(5)):
                st.write(f"{i+1}. {q}")
        
        st.subheader("Medical Entities")
        entities_expander = st.expander("View recognized entities")
        with entities_expander:
            for category, entities in st.session_state.chatbot.medical_entities.items():
                st.write(f"**{category.title()}:**")
                st.write(", ".join(entities))
        
        st.subheader("Options")
        st.session_state.show_entities = st.checkbox("Show medical entities", value=True)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if message["role"] == "assistant" and st.session_state.show_entities:
                    entities = st.session_state.chatbot.recognize_medical_entities(message["content"])
                    if entities:
                        st.markdown("**Recognized Medical Entities:**")
                        for category, entity_list in entities.items():
                            st.write(f"• **{category.title()}:** {', '.join(entity_list)}")
        
        # User input
        user_query = st.chat_input("Ask a medical question...")
        
        if user_query:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Get bot response
            with st.spinner("Searching for answers..."):
                similar_questions = st.session_state.chatbot.find_similar_questions(user_query)
                
                if similar_questions:
                    best_match = similar_questions[0]
                    response = f"**Answer:** {best_match['answer']}\n\n"
                    response += f"*Similarity: {best_match['similarity']:.2%}*"
                    
                    # Add related questions
                    if len(similar_questions) > 1:
                        response += "\n\n**Related Questions:**\n"
                        for i, similar in enumerate(similar_questions[1:4], 1):
                            response += f"{i}. {similar['question']}\n"
                else:
                    response = "I'm sorry, I couldn't find a specific answer to your question in my knowledge base. Please consult a healthcare professional for medical advice."
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        st.subheader("Quick Questions")
        st.markdown("Try these sample questions:")
        
        sample_questions = [
            "What are diabetes symptoms?",
            "How to treat hypertension?",
            "What causes asthma?",
            "Early cancer signs",
            "Heart disease prevention"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"btn_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                
                with st.spinner("Searching..."):
                    similar_questions = st.session_state.chatbot.find_similar_questions(question)
                    
                    if similar_questions:
                        best_match = similar_questions[0]
                        response = f"**Answer:** {best_match['answer']}\n\n"
                        response += f"*Similarity: {best_match['similarity']:.2%}*"
                    else:
                        response = "I couldn't find a specific answer. Please consult a healthcare professional."
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.subheader("Medical Disclaimer")
        st.warning("""
        This chatbot provides general health information and is not a substitute for professional medical advice. 
        Always consult healthcare professionals for medical concerns.
        """)

if __name__ == "__main__":
    main()
