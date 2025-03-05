import numpy as np
from collections import Counter
import re

class SentimentWNN:
    """
    Simplified Weightless Neural Network for Sentiment Analysis
    Following AO Labs' pure binary channel architecture for pattern-based learning
    """
    def __init__(self):
        # Architecture sizes
        self.vocab_size = 5000  # Maximum vocabulary size
        self.pattern_size = 256  # Pattern channels
        self.output_size = 1    # Binary sentiment
        
        # Binary state layers
        self.input_state = np.zeros(self.pattern_size, dtype=np.int8)
        self.hidden_state = np.zeros(self.pattern_size, dtype=np.int8)
        self.prev_hidden_state = np.zeros(self.pattern_size, dtype=np.int8)
        self.output_state = np.zeros(self.output_size, dtype=np.int8)
        
        # Initialize pattern connections with more structure
        self.pattern_connections = np.zeros((self.pattern_size, self.pattern_size), dtype=np.int8)
        for i in range(self.pattern_size):
            # Create structured local connections with stronger negative bias
            neighbors = np.random.choice(self.pattern_size, size=12, replace=False)
            self.pattern_connections[i, neighbors] = 1
        
        # Initialize output connections with balanced weights
        self.output_connections = np.zeros((self.output_size, self.pattern_size), dtype=np.int8)
        active_outputs = np.random.choice(self.pattern_size, size=self.pattern_size//2, replace=False)
        self.output_connections[0, active_outputs] = 1
        
        # Common words (for initialization)
        self.word_to_pattern = {}
        self.initialize_common_words()
        
        # Thresholds for activation
        self.hidden_threshold = 0.35  # Slightly higher threshold for hidden layer
        self.output_threshold = 0.55  # Higher threshold for positive sentiment
    
    def initialize_common_words(self):
        """Initialize patterns for common sentiment words with stronger biases"""
        positive_words = {
            'good': 0.9, 'great': 0.9, 'excellent': 0.95, 'amazing': 0.95,
            'love': 0.95, 'best': 0.9, 'perfect': 0.95, 'wonderful': 0.9,
            'fantastic': 0.9, 'awesome': 0.9, 'happy': 0.85, 'beautiful': 0.85
        }
        negative_words = {
            'bad': 0.1, 'terrible': 0.05, 'worst': 0.05, 'hate': 0.05,
            'poor': 0.1, 'awful': 0.05, 'horrible': 0.05, 'avoid': 0.1,
            'disappointed': 0.1, 'waste': 0.1, 'dont': 0.15, 'not': 0.15,
            'never': 0.1, 'worthless': 0.05, 'garbage': 0.05, 'useless': 0.1
        }
        neutral_words = {
            'the': 0.5, 'and': 0.5, 'is': 0.5, 'at': 0.5, 'in': 0.5,
            'to': 0.5, 'a': 0.5, 'of': 0.5, 'for': 0.5, 'with': 0.5,
            'buy': 0.5, 'price': 0.5, 'quality': 0.5, 'product': 0.5,
            'it': 0.5, 'this': 0.5, 'that': 0.5, 'they': 0.5
        }
        
        # Initialize patterns with stronger biases
        for word, bias in {**positive_words, **negative_words, **neutral_words}.items():
            if bias < 0.5:  # Negative words
                pattern = np.random.binomial(1, 0.05, self.pattern_size)  # Very strong negative bias
            elif bias > 0.5:  # Positive words
                pattern = np.random.binomial(1, 0.95, self.pattern_size)  # Very strong positive bias
            else:  # Neutral words
                pattern = np.random.binomial(1, 0.5, self.pattern_size)  # Balanced
            self.word_to_pattern[word] = pattern
    
    def ensure_pattern_size(self, pattern):
        """Ensure pattern has correct size through padding or truncation"""
        pattern = np.asarray(pattern)
        if len(pattern) > self.pattern_size:
            return pattern[:self.pattern_size]
        elif len(pattern) < self.pattern_size:
            padded = np.zeros(self.pattern_size, dtype=np.int8)
            padded[:len(pattern)] = pattern
            return padded
        return pattern
    
    def text_to_binary(self, text):
        """Convert text to binary pattern using word-level features"""
        # Preprocess text
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        # Get or create patterns for words
        patterns = []
        for word in words:
            if word not in self.word_to_pattern:
                # Create balanced random pattern for unknown word
                pattern = np.random.binomial(1, 0.5, self.pattern_size)
                self.word_to_pattern[word] = pattern
            patterns.append(self.word_to_pattern[word])
        
        if not patterns:
            return np.zeros(self.pattern_size, dtype=np.int8)
        
        # Combine patterns using weighted average
        combined = np.zeros(self.pattern_size, dtype=np.float32)
        for pattern in patterns:
            pattern = self.ensure_pattern_size(pattern)
            combined += pattern
        
        # Normalize and threshold
        combined /= max(1, len(patterns))
        return (combined > 0.5).astype(np.int8)
    
    def update_states(self, input_pattern):
        """Update network states using binary operations"""
        # Ensure input pattern has correct size
        input_pattern = self.ensure_pattern_size(input_pattern)
        
        # Save previous state
        self.prev_hidden_state = self.hidden_state.copy()
        
        # Update input state
        self.input_state = input_pattern
        
        # Update hidden state through pattern connections
        new_hidden_state = np.zeros(self.pattern_size, dtype=np.int8)
        for i in range(self.pattern_size):
            connected = self.pattern_connections[i] * self.input_state
            new_hidden_state[i] = int(np.sum(connected) > np.sum(self.pattern_connections[i]) * self.hidden_threshold)
        self.hidden_state = new_hidden_state
        
        # Update output state with balanced threshold
        connected = self.output_connections * self.hidden_state
        self.output_state[0] = int(np.sum(connected) > np.sum(self.output_connections) * self.output_threshold)
    
    def predict(self, text):
        """Make binary prediction from text with improved negative detection"""
        input_pattern = self.text_to_binary(text)
        self.update_states(input_pattern)
        
        # Calculate sentiment score based on pattern activation
        sentiment_score = np.mean(self.hidden_state)
        
        # Process text for negation and strong indicators
        text = text.lower()
        words = set(re.findall(r'\w+', text))
        
        # Strong negative indicators (immediate negative)
        strong_negative = {'worst', 'terrible', 'horrible', 'hate', 'awful', 'garbage', 'worthless'}
        if any(word in strong_negative for word in words):
            return 0
        
        # Negation handling
        negations = {'not', 'dont', 'never', 'no', 'cant', 'wont', 'shouldnt', 'wouldnt'}
        has_negation = any(word in negations for word in words)
        
        # Adjust sentiment based on negation
        if has_negation:
            sentiment_score = 1 - sentiment_score
        
        # Use stricter threshold for positive sentiment
        return int(sentiment_score > self.output_threshold)
    
    def update(self, text, label):
        """Learn from a single example"""
        try:
            # Get input pattern with correct size
            input_pattern = self.text_to_binary(text)
            input_pattern = self.ensure_pattern_size(input_pattern)
            self.update_states(input_pattern)
            
            if self.output_state[0] != label:
                # Update output connections
                if label == 1:
                    # Strengthen connections to active hidden units
                    self.output_connections[0, self.hidden_state == 1] = 1
                else:
                    # Weaken connections to active hidden units
                    self.output_connections[0, self.hidden_state == 1] = 0
                
                # Update pattern connections based on input
                active_inputs = (input_pattern == 1)
                active_hidden = (self.hidden_state == 1)
                
                # Ensure shapes match before updating
                if len(active_inputs) == self.pattern_size and len(active_hidden) == self.pattern_size:
                    if label == 1:
                        # Strengthen connections for positive examples
                        self.pattern_connections[active_hidden, :] = np.logical_or(
                            self.pattern_connections[active_hidden, :],
                            np.broadcast_to(active_inputs, (np.sum(active_hidden), self.pattern_size))
                        )
                    else:
                        # Weaken connections for negative examples
                        self.pattern_connections[active_hidden, :] = np.logical_and(
                            self.pattern_connections[active_hidden, :],
                            ~np.broadcast_to(active_inputs, (np.sum(active_hidden), self.pattern_size))
                        )
        except Exception as e:
            print(f"Warning: Update failed: {str(e)}")
    
    def train(self, texts, labels):
        """Train on a batch of examples"""
        correct = 0
        total = 0
        for text, label in zip(texts, labels):
            try:
                # Ensure input has correct pattern size
                pred = self.predict(text)
                if pred != label:
                    self.update(text, label)
                correct += (pred == label)
                total += 1
            except Exception as e:
                print(f"Warning: Failed to process example: {str(e)}")
                continue
        return correct / max(1, total)
    
    def get_confidence(self):
        """Calculate confidence based on state stability and pattern strength"""
        state_stability = np.mean(self.hidden_state == self.prev_hidden_state)
        pattern_strength = np.abs(np.mean(self.hidden_state) - 0.5) * 2
        return (state_stability + pattern_strength) / 2
    
    def get_word_sentiment(self, word):
        """Get sentiment strength for a word"""
        if word not in self.word_to_pattern:
            return 0.5  # Neutral for unknown words
        pattern = self.word_to_pattern[word]
        sentiment = np.mean(pattern)
        return sentiment 