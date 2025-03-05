# Sentiment Analysis Model with Weightless Neural Networks
import streamlit as st
import numpy as np
from src.arch_sentiment import SentimentWNN
from src.data_loader import load_sentiment140

def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = SentimentWNN()
        st.session_state.trained = False

def train_model():
    """Train the model on Sentiment140 dataset"""
    try:
        # Load data
        train_data, _ = load_sentiment140(split_ratio=0.8)
        
        if train_data is None:
            return False, "Please upload the Sentiment140 dataset to proceed."
        
        # Take a small subset for quick training
        train_subset = train_data.sample(n=100, random_state=42)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train in small batches
        batch_size = 5
        num_batches = len(train_subset) // batch_size
        accuracies = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            batch_texts = train_subset['text'].values[start_idx:end_idx]
            batch_labels = train_subset['target'].values[start_idx:end_idx]
            
            # Train and track accuracy
            acc = st.session_state.model.train(batch_texts, batch_labels)
            accuracies.append(acc)
            
            # Update progress
            progress = (i + 1) / num_batches
            progress_bar.progress(progress)
            status_text.text(f"Training batch {i+1}/{num_batches} (Accuracy: {np.mean(accuracies):.2f})")
        
        st.session_state.trained = True
        return True, np.mean(accuracies)
    except Exception as e:
        return False, str(e)

def main():
    st.title("Sentiment Analysis Agent")
    st.write("A Weightless Neural Network Model for Twitter Sentiment Analysis")
    
    init_session_state()
    
    # Sidebar for model controls
    with st.sidebar:
        st.header("Model Controls")
        if st.button("Reset Model"):
            st.session_state.model = SentimentWNN()
            st.session_state.trained = False
            st.success("Model reset successfully!")
        
        st.markdown("---")
        st.markdown("""
        ### About This Model
        An implementation of AO Labs' weightless neural network architecture:
        - Pure binary operations throughout
        - Pattern-based sentiment recognition
        - Continuous learning capability
        - Transparent decision making
        
        ### Implementation Details
        - Binary state channels (256-dimensional)
        - Local connection patterns
        - Dynamic pattern adaptation
        - Word-level sentiment analysis
        
        ### Learning Approach
        - Small initial training set (100 samples)
        - Interactive feedback loop
        - Real-time pattern updates
        - Continuous accuracy improvement
        """)
    
    # Model training section
    if not st.session_state.trained:
        st.warning("Model needs initial training to learn basic sentiment patterns")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                success, result = train_model()
                if success:
                    st.success(f"Initial training successful! Accuracy: {result:.2f}")
                else:
                    st.error(f"Training failed: {result}")
    else:
        st.success("Model is trained and ready for analysis!")
    
    # Main interface
    st.header("Try it out!")
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your text here or choose from examples below..."
    )
    
    # Example selector
    example_type = st.selectbox(
        "Or choose an example:",
        ["Custom", "Positive Examples", "Negative Examples"]
    )
    
    if example_type == "Positive Examples":
        text_input = st.selectbox("Select positive example:", [
            "This is amazing! Really love it!",
            "Great experience, highly recommend!",
            "Excellent service, very satisfied!",
            "Perfect solution to my problem!",
            "Best decision I've made!"
        ])
    elif example_type == "Negative Examples":
        text_input = st.selectbox("Select negative example:", [
            "This is terrible, very disappointed!",
            "Worst experience ever, avoid!",
            "Complete waste of time!",
            "Absolutely horrible service!"
        ])
    
    # Analyze button
    if text_input and st.button("Analyze Sentiment"):
        if not st.session_state.trained:
            st.error("Please train the model first!")
            return
        
        try:
            # Get prediction and confidence
            prediction = st.session_state.model.predict(text_input)
            confidence = st.session_state.model.get_confidence()
            
            # Display results
            st.header("Results")
            
            # Sentiment result
            st.subheader("Sentiment")
            if prediction == 1:
                st.write("Positive 😊")
            else:
                st.write("Negative ☹️")
            
            # Confidence score
            st.subheader("Confidence")
            st.write(f"{confidence*100:.2f}%")
            
            # Word Analysis
            st.subheader("Word Analysis")
            st.write("Word patterns detected:")
            
            words = text_input.lower().split()
            word_sentiments = {}
            for word in set(words):
                sentiment = st.session_state.model.get_word_sentiment(word)
                if abs(sentiment - 0.5) > 0.1:  # Only show words with meaningful sentiment
                    word_sentiments[word] = sentiment
            
            # Sort words by sentiment strength and display
            sorted_words = sorted(word_sentiments.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
            for word, sentiment in sorted_words:
                sentiment_str = "positive" if sentiment > 0.5 else "negative"
                strength = abs(sentiment - 0.5) * 2 * 100
                st.write(f"'{word}': {sentiment_str} ({strength:.2f}% strength)")
            
            # Feedback section
            st.subheader("Help Improve the Model")
            st.write("Was this prediction correct?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("👍 Yes"):
                    st.session_state.model.update(text_input, prediction)
                    st.success("Thank you! This helps the model learn.")
            
            with col2:
                if st.button("👎 No"):
                    correct_label = 0 if prediction == 1 else 1
                    st.session_state.model.update(text_input, correct_label)
                    st.success("Thank you! The model will learn from this correction.")
                    
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main() 