# Sentiment Analysis with Weightless Neural Networks

A sentiment analysis implementation using pure binary/weightless neural networks, created for [AO Labs Issue #11](https://github.com/aolabsai/ao_arch/issues/11). This project demonstrates how to build an agent that can learn from new datasets while maintaining the core principles of weightless neural networks.

## Key Features

- **Pure Binary Operations**: Implements AO Labs' weightless neural network architecture
- **Pattern-Based Learning**: Uses binary patterns for sentiment recognition
- **Continuous Learning**: Adapts through user feedback and interaction
- **Minimal Training**: Starts with small dataset (100 samples) to demonstrate quick learning
- **Interactive Interface**: Real-time analysis and feedback collection

## Dataset

This project uses the Sentiment140 dataset. You have several options:

1. Use the full dataset:

   - Download the full Sentiment140 dataset
   - Place `sentiment140.csv` in the `data/` directory
   - The application will automatically create a balanced sample (100k tweets)

2. Use the sample dataset:

   - A balanced sample of 100k tweets will be created automatically
   - The sample is created once and reused for subsequent runs

3. Upload through web interface:
   - Upload either the full dataset or a sample
   - If you upload the full dataset, it will be automatically sampled
   - Maximum upload size: 200MB

## AO Labs Architecture Implementation

1. **Weightless Neural Networks**:

2. **Continuous Learning**:

3. **Pattern Recognition**:

## Setup and Running

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Start the application
2. Upload or use local Sentiment140 dataset
3. Train the model (uses a small subset for demonstration)
4. Enter text or choose examples to analyze
5. Provide feedback to help the model learn

## Technical Details

### Architecture

- Input Layer: Binary patterns from text processing
- Hidden Layer: Pattern-based connections with binary states
- Output Layer: Binary decision channels for sentiment prediction

### Implementation

- Pure binary operations throughout
- Pattern-based processing and storage
- Local connection patterns
- Dynamic threshold adaptation

### Learning Process

- Initial training on small dataset (100 samples)
- Continuous learning through user feedback
- Pattern reinforcement based on correct predictions
- Real-time adaptation of connection patterns

## Code Structure

```
├── data/
│   └── sentiment140_sample.csv  # Sample dataset for training
├── src/
│   ├── arch_sentiment.py        # WNN implementation
│   └── data_loader.py           # Dataset handling
├── app.py                       # Streamlit interface
└── requirements.txt             # Dependencies
```

## Dependencies

- streamlit>=1.28.0 (for interactive interface)
- pandas>=2.0.0 (for dataset handling)
- numpy>=1.24.0 (for binary operations)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built following AO Labs' weightless neural network architecture principles
- Uses Sentiment140 dataset for demonstration
- Created in response to [AO Labs Issue #11](https://github.com/aolabsai/ao_arch/issues/11)
- Dataset: Sentiment140 (https://www.kaggle.com/datasets/kazanova/sentiment140)
