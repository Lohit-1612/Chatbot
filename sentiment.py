import streamlit as st
import re
import random
from textblob import TextBlob
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentimentChatbot:
    def __init__(self):
        # Response templates for different sentiments
        self.response_templates = {
            'positive': {
                'greeting': [
                    "I'm glad to hear you're feeling positive! How can I assist you today?",
                    "Great to see your positive energy! What can I help you with?",
                    "Wonderful! I'm here to help. What do you need assistance with?"
                ],
                'general': [
                    "That's fantastic! How can I make your day even better?",
                    "I love your positive attitude! What can I do for you?",
                    "Your positivity is contagious! How may I assist you?"
                ],
                'problem_solution': [
                    "I'm thrilled I could help! Is there anything else you'd like to know?",
                    "Great! I'm happy that solved your problem. What else can I assist with?",
                    "Perfect! So glad that worked out. How else can I be of service?"
                ]
            },
            'negative': {
                'greeting': [
                    "I understand you might be feeling frustrated. I'm here to help resolve any issues.",
                    "I'm sorry to hear you're having a tough time. Let me help you with that.",
                    "I can see this is concerning you. Let's work together to find a solution."
                ],
                'apology': [
                    "I apologize for the inconvenience. Let me help fix this for you.",
                    "I'm sorry you're experiencing this issue. I'll do my best to resolve it.",
                    "My apologies for the trouble. Let me assist you right away."
                ],
                'escalation': [
                    "I understand this is serious. Let me connect you with a specialist who can help.",
                    "This sounds important. I'll escalate this to our support team immediately.",
                    "I want to make sure you get the best help. Let me transfer you to an expert."
                ]
            },
            'neutral': {
                'greeting': [
                    "Hello! How can I assist you today?",
                    "Hi there! What can I help you with?",
                    "Welcome! How may I be of service?"
                ],
                'general': [
                    "How can I help you with that?",
                    "What would you like to know about this?",
                    "I'm here to assist. Could you tell me more?"
                ]
            }
        }
        
        # Product/service knowledge base
        self.knowledge_base = {
            'shipping': {
                'questions': ['shipping', 'delivery', 'ship', 'deliver', 'arrive', 'when will it come'],
                'answers': {
                    'positive': "Our shipping is fast and reliable! Standard delivery takes 3-5 business days.",
                    'negative': "I understand shipping delays can be frustrating. We offer expedited shipping options and can check your order status.",
                    'neutral': "Standard shipping takes 3-5 business days. Express options are available for faster delivery."
                }
            },
            'returns': {
                'questions': ['return', 'refund', 'exchange', 'send back', 'send back'],
                'answers': {
                    'positive': "Our return process is customer-friendly! You have 30 days to return items in original condition.",
                    'negative': "I'm sorry about the return issue. We'll make the process smooth for you. Let me guide you through the steps.",
                    'neutral': "Returns are accepted within 30 days with original packaging. The process usually takes 5-7 business days for refunds."
                }
            },
            'pricing': {
                'questions': ['price', 'cost', 'expensive', 'cheap', 'discount', 'how much'],
                'answers': {
                    'positive': "We offer great value! Check our current promotions for the best deals.",
                    'negative': "I understand pricing concerns. Let me check if there are any available discounts or payment plans.",
                    'neutral': "Our pricing is competitive. Would you like information about current promotions or payment options?"
                }
            },
            'quality': {
                'questions': ['quality', 'broken', 'defective', 'work', 'function', 'not working'],
                'answers': {
                    'positive': "We're proud of our product quality! All items come with a 1-year warranty.",
                    'negative': "I'm sorry to hear about the quality issue. We'll replace it immediately under our warranty.",
                    'neutral': "Our products come with a 1-year warranty and quality guarantee. Let me know if you have specific concerns."
                }
            },
            'account': {
                'questions': ['account', 'login', 'password', 'sign in', 'profile'],
                'answers': {
                    'positive': "Great! I can help with your account. What specific account feature would you like to know about?",
                    'negative': "I'm sorry you're having account issues. Let's get that sorted out for you right away.",
                    'neutral': "I can help with account-related questions. What do you need assistance with?"
                }
            }
        }

    def analyze_sentiment(self, text):
        """Analyze sentiment of the input text"""
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity, analysis.sentiment.subjectivity
        elif polarity < -0.1:
            return 'negative', polarity, analysis.sentiment.subjectivity
        else:
            return 'neutral', polarity, analysis.sentiment.subjectivity

    def extract_intent(self, text):
        """Extract the intent from user message"""
        text_lower = text.lower()
        
        # Check for greetings
        greeting_words = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']
        if any(word in text_lower for word in greeting_words):
            return 'greeting'
        
        # Check for thanks
        if any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'thanks'
        
        # Check for specific topics
        for topic, data in self.knowledge_base.items():
            if any(word in text_lower for word in data['questions']):
                return topic
        
        return 'general'

    def generate_response(self, user_input, sentiment, intent, sentiment_score):
        """Generate appropriate response based on sentiment and intent"""
        
        # Special handling for very negative sentiments
        if sentiment == 'negative' and sentiment_score < -0.5:
            response = random.choice(self.response_templates['negative']['escalation'])
            return response + " Could you please describe the issue in more detail?"
        
        # Handle greetings
        if intent == 'greeting':
            if sentiment == 'positive':
                return random.choice(self.response_templates['positive']['greeting'])
            elif sentiment == 'negative':
                return random.choice(self.response_templates['negative']['greeting'])
            else:
                return random.choice(self.response_templates['neutral']['greeting'])
        
        # Handle thanks
        if intent == 'thanks':
            return "You're very welcome! I'm glad I could help. Is there anything else you need assistance with today?"
        
        # Handle specific topics
        for topic, data in self.knowledge_base.items():
            if intent == topic:
                base_response = data['answers'][sentiment]
                
                # Add sentiment-appropriate follow-up
                if sentiment == 'positive':
                    follow_up = " Is there anything else I can help with to maintain this great experience?"
                elif sentiment == 'negative':
                    follow_up = " I want to make sure we resolve this completely for you."
                else:
                    follow_up = " Would you like more specific information about this?"
                
                return base_response + follow_up
        
        # General responses based on sentiment
        if sentiment == 'positive':
            return random.choice(self.response_templates['positive']['general'])
        elif sentiment == 'negative':
            apology = random.choice(self.response_templates['negative']['apology'])
            return apology + " Please tell me more about what's concerning you."
        else:
            return random.choice(self.response_templates['neutral']['general'])

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = SentimentChatbot()
    if 'satisfaction_score' not in st.session_state:
        st.session_state.satisfaction_score = 0
    if 'sentiment_history' not in st.session_state:
        st.session_state.sentiment_history = []

def display_chat_message(role, message, sentiment=None):
    """Display a chat message with appropriate styling"""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message)
            if sentiment:
                st.caption(f"Sentiment: {sentiment}")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message)

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    emoji_map = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emoji_map.get(sentiment, '😐')

def plot_sentiment_analysis(conversation_history):
    """Create sentiment analysis visualization"""
    if len(conversation_history) < 2:
        return None
    
    # Prepare data for plotting
    df = pd.DataFrame(conversation_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['sentiment_score'] = df['sentiment_score'].astype(float)
    
    # Create line chart for sentiment over time
    fig = px.line(df, x='timestamp', y='sentiment_score', 
                  title='Sentiment Trend During Conversation',
                  labels={'sentiment_score': 'Sentiment Score', 'timestamp': 'Time'})
    
    # Add sentiment zones
    fig.add_hrect(y0=0.1, y1=1, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Positive")
    fig.add_hrect(y0=-0.1, y1=0.1, line_width=0, fillcolor="yellow", opacity=0.1, annotation_text="Neutral")
    fig.add_hrect(y0=-1, y1=-0.1, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Negative")
    
    return fig

def plot_sentiment_distribution(conversation_history):
    """Create sentiment distribution pie chart"""
    if not conversation_history:
        return None
    
    sentiments = [msg['sentiment'] for msg in conversation_history if msg['role'] == 'user']
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    fig = px.pie(values=sentiment_counts.values, 
                 names=sentiment_counts.index,
                 title='Sentiment Distribution in Conversation',
                 color=sentiment_counts.index,
                 color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'yellow'})
    
    return fig

def calculate_satisfaction_score(conversation_history):
    """Calculate customer satisfaction score"""
    if not conversation_history:
        return 50  # Default neutral score
    
    user_messages = [msg for msg in conversation_history if msg['role'] == 'user']
    if not user_messages:
        return 50
    
    positive_count = sum(1 for msg in user_messages if msg['sentiment'] == 'positive')
    negative_count = sum(1 for msg in user_messages if msg['sentiment'] == 'negative')
    total_messages = len(user_messages)
    
    # Simple scoring: more positive = higher score
    base_score = (positive_count / total_messages) * 100
    # Penalize for negative sentiments
    penalty = (negative_count / total_messages) * 30
    final_score = max(0, min(100, base_score - penalty))
    
    return final_score

def main():
    st.set_page_config(
        page_title="Sentiment-Aware Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Sentiment Analysis")
        st.markdown("---")
        
        if st.session_state.conversation:
            satisfaction_score = calculate_satisfaction_score(st.session_state.conversation)
            st.session_state.satisfaction_score = satisfaction_score
            
            # Satisfaction gauge
            st.subheader("Customer Satisfaction")
            gauge_color = "green" if satisfaction_score > 70 else "red" if satisfaction_score < 40 else "orange"
            st.markdown(f"""
            <div style="text-align: center;">
                <h1 style="color: {gauge_color}; font-size: 48px;">{satisfaction_score:.0f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Conversation stats
            st.subheader("Conversation Stats")
            user_messages = [msg for msg in st.session_state.conversation if msg['role'] == 'user']
            if user_messages:
                sentiment_counts = pd.Series([msg['sentiment'] for msg in user_messages]).value_counts()
                for sentiment, count in sentiment_counts.items():
                    emoji = get_sentiment_emoji(sentiment)
                    st.write(f"{emoji} {sentiment.title()}: {count}")
            
            st.markdown("---")
        
        # Clear conversation button
        if st.button("🔄 Clear Conversation", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.sentiment_history = []
            st.rerun()
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot uses sentiment analysis to:
        - Detect customer emotions 😊😐😞
        - Respond appropriately to each sentiment
        - Track conversation quality
        - Improve customer satisfaction
        """)
    
    # Main content area
    st.title("🤖 Sentiment-Aware Chatbot")
    st.markdown("I'm here to help! I can understand your emotions and respond appropriately.")
    
    # Create two columns for the main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat container
        st.subheader("💬 Chat")
        
        # Display conversation history
        for message in st.session_state.conversation:
            display_chat_message(
                message['role'], 
                message['content'],
                sentiment=f"{message['sentiment'].title()} ({message['sentiment_score']:.2f})" if message['role'] == 'user' else None
            )
        
        # User input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to conversation
            sentiment, score, subjectivity = st.session_state.chatbot.analyze_sentiment(user_input)
            user_message = {
                'role': 'user',
                'content': user_input,
                'sentiment': sentiment,
                'sentiment_score': score,
                'subjectivity': subjectivity,
                'timestamp': datetime.now()
            }
            st.session_state.conversation.append(user_message)
            
            # Generate bot response
            intent = st.session_state.chatbot.extract_intent(user_input)
            response = st.session_state.chatbot.generate_response(user_input, sentiment, intent, score)
            
            # Add bot response to conversation
            bot_message = {
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            }
            st.session_state.conversation.append(bot_message)
            
            # Update sentiment history
            st.session_state.sentiment_history.append({
                'timestamp': datetime.now(),
                'sentiment': sentiment,
                'score': score,
                'message': user_input[:50] + '...' if len(user_input) > 50 else user_input
            })
            
            st.rerun()
    
    with col2:
        st.subheader("📊 Analytics")
        
        if st.session_state.conversation:
            # Sentiment trend chart
            fig_trend = plot_sentiment_analysis([
                {**msg, 'timestamp': msg['timestamp']} 
                for msg in st.session_state.conversation 
                if msg['role'] == 'user'
            ])
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Sentiment distribution
            fig_dist = plot_sentiment_distribution(st.session_state.conversation)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Recent sentiments
            st.subheader("Recent Sentiments")
            recent_sentiments = st.session_state.sentiment_history[-5:]  # Last 5 sentiments
            for sentiment_data in reversed(recent_sentiments):
                emoji = get_sentiment_emoji(sentiment_data['sentiment'])
                st.write(f"{emoji} {sentiment_data['sentiment'].title()} ({sentiment_data['score']:.2f})")
                st.caption(sentiment_data['message'])
                st.markdown("---")
        else:
            st.info("Start a conversation to see analytics here!")
            
            # Demo sentiments
            st.subheader("Try these examples:")
            demo_examples = [
                ("I love your service! 😊", "positive"),
                ("I'm having issues with my order 😞", "negative"),
                ("What's the delivery time?", "neutral"),
                ("This product is amazing!", "positive"),
                ("I want a refund now!", "negative")
            ]
            
            for example, sentiment in demo_examples:
                emoji = get_sentiment_emoji(sentiment)
                if st.button(f"{emoji} {example}", use_container_width=True):
                    st.session_state.demo_input = example
                    st.rerun()

# Run the app
if __name__ == "__main__":
    main()
