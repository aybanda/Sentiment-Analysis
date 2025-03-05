# Sentiment Analysis with Weightless Neural Networks

A sentiment analysis model using pure binary/weightless neural networks, following AO Labs' architecture principles.

## Dataset

This project uses the Sentiment140 dataset. You can:

1. Place the dataset file (`sentiment140.csv`) in the `data/` directory locally
2. Or upload it through the web interface when running the application

## Setup and Running

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
streamlit run app.py
```

## Features

- Pure binary/weightless neural networks
- Pattern-based sentiment recognition
- Continuous learning capability
- Word-level analysis
- Real-time feedback integration

## Architecture

Built following AO Labs' architecture principles:

- Binary state channels (256-dimensional)
- Local connection patterns
- Dynamic pattern adaptation
- Real-time learning

## Usage

1. Start the application
2. Upload or use local Sentiment140 dataset
3. Train the model (uses a small subset for demonstration)
4. Enter text or choose examples to analyze
5. Provide feedback to help the model learn

## Overview

This Model demonstrates AO Labs' architecture through:

- Pure binary/weightless neural network for sentiment analysis
- Pattern-based learning with binary channel operations
- Real-time continuous learning capabilities
- Interactive feedback and adaptation

### Architecture

- Input Layer: Binary patterns from text processing
- Hidden Layer: Pattern-based connections with binary states
- Output Layer: Binary decision channels for sentiment prediction

## Quick Start

1. First, download the Sentiment140 dataset:

   ```bash
   # Create data directory
   mkdir -p data

   # Download dataset (1.6M tweets)
   wget -O data/sentiment140.csv http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
   unzip -p trainingandtestdata.zip training.1600000.processed.noemoticon.csv > data/sentiment140.csv
   ```

2. Install and run:

   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Run the agent
   streamlit run app.py
   ```

## Implementation Details

### 1. Agent Architecture

- Pure binary operations throughout
- Pattern-based processing and storage
- Local connection patterns
- Dynamic threshold adaptation

### 2. Continuous Learning

- Starts with minimal initial training (100 samples)
- Learns from user feedback in real-time
- Updates patterns based on interaction
- Adapts connection strengths through use

### 3. Interactive Interface

- Real-time sentiment analysis
- Word-level pattern analysis
- Confidence scoring
- User feedback integration

## Code Structure

```
├── data/
│   └── sentiment140.csv     # Training dataset
├── src/
│   ├── arch_sentiment.py    # WNN Agent implementation
│   └── data_loader.py       # Sentiment140 dataset handler
├── app.py                   # Streamlit interface
└── requirements.txt         # Dependencies
```

## Usage Example

```python
from src.arch_sentiment import SentimentWNN

# Initialize agent
agent = SentimentWNN()

# Initial training
agent.train(texts, labels)  # Train on 100 samples

# Make prediction
text = "This is amazing!"
prediction = agent.predict(text)
confidence = agent.get_confidence()

# Learn from feedback
agent.update(text, label=1)  # 1 for positive sentiment
```

## Features

- Binary pattern operations
- Real-time learning and adaptation
- Pattern-based sentiment analysis
- Interactive learning interface
- Word-level sentiment detection

## Dependencies

Required packages (with minimum versions):

- streamlit>=1.24.0 (for interactive interface)
- pandas>=1.5.0 (for dataset handling)
- numpy>=1.21.0 (for binary operations)

All dependencies focus on maintaining a lightweight implementation aligned with AO Labs' binary/weightless approach.

## Acknowledgments

- Built on AO Labs' weightless neural network architecture
- Uses Sentiment140 dataset for initial training
- Implements pure binary pattern-based learning

## License

MIT License
