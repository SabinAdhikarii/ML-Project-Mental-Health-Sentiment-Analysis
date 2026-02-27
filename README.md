# Mental Health Condition Classification

A comprehensive text classification project that implements and compares multiple machine learning approaches to classify mental health conditions from textual data.

## Project Overview

This project develops and evaluates three different machine learning models for automated mental health condition classification:
- Logistic Regression (baseline model)
- Random Forest (traditional ML)
- BERT (deep learning transformer)

The system classifies text into seven mental health categories: Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal.

## Dataset

**Source:** Mental Health Condition Classification Dataset  
**Platform:** HuggingFace Datasets  
**Dataset ID:** sai1908/Mental_Health_Condition_Classification

### Dataset Statistics
- Total samples: 103,488
- Number of classes: 7
- Average text length: 82 words
- Class distribution: Relatively balanced (10.67% - 17.03%)

### Classes
1. Anxiety (17,620 samples - 17.03%)
2. Normal (16,068 samples - 15.53%)
3. Depression (15,901 samples - 15.37%)
4. Stress (15,230 samples - 14.72%)
5. Personality Disorder (13,915 samples - 13.45%)
6. Bipolar (13,708 samples - 13.25%)
7. Suicidal (11,046 samples - 10.67%)

## Project Structure

```
mental-health-classification/
├── Mental_Health_Classification.ipynb    # Main project notebook
├── WordCloud_Guide.ipynb                 # Word cloud generation guide
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── results/                              # Model outputs and results
```
## Models

### 1. Logistic Regression
- **Features:** TF-IDF vectorization (max 5000 features)
- **Advantages:** Fast training, interpretable, good baseline
- **Use case:** Quick preliminary analysis

### 2. Random Forest
- **Features:** TF-IDF vectorization (same as Logistic Regression)
- **Configuration:** 100 estimators, max depth 50
- **Advantages:** Handles non-linear relationships, provides feature importance
- **Use case:** More robust predictions than simple linear models

### 3. BERT (Bidirectional Encoder Representations from Transformers)
- **Model:** bert-base-uncased
- **Alternative:** distilbert-base-uncased (recommended for faster training)
- **Configuration:** 3 epochs, batch size 16, max length 128 tokens
- **Advantages:** State-of-the-art contextual understanding
- **Use case:** Highest accuracy predictions

## Implementation Details

### Text Preprocessing
- Lowercase conversion
- Special character and digit removal
- Tokenization
- Stopword removal
- Lemmatization

### Feature Extraction
**For Traditional ML Models:**
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Maximum features: 5000
- N-gram range: (1, 2) - unigrams and bigrams

**For BERT:**
- WordPiece tokenization
- Maximum sequence length: 128 tokens
- Padding and truncation applied

### Training Configuration
- Train-test split: 80-20
- Stratified sampling to maintain class distribution
- Random seed: 42 (for reproducibility)

## Evaluation Metrics

The models are evaluated using:
- **Accuracy:** Overall correctness
- **Precision:** Positive prediction accuracy
- **Recall:** True positive detection rate
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed class-wise performance
- **ROC-AUC:** Area under ROC curve for each class

## Results

### Performance Comparison
Models are compared across multiple metrics with visualizations including:
- Overall performance bar charts
- Per-class F1-score comparison
- Confusion matrices for each model
- ROC curves (multi-class)

### Expected Performance Hierarchy
Logistic Regression < Random Forest < BERT

## Visualizations

The project includes comprehensive visualizations:
- Class distribution plots
- Text length distribution analysis
- Word clouds for each mental health condition
- Top words frequency charts
- Confusion matrices
- ROC curves
- Feature importance plots (Random Forest)

## Ethical Considerations

### Critical Issues Addressed

**1. Bias and Fairness**
- Potential biases in training data
- Performance variation across demographics
- Mitigation through diverse data collection

**2. Privacy and Confidentiality**
- Sensitive mental health information
- Data anonymization requirements
- HIPAA/GDPR compliance considerations

**3. Misclassification Consequences**
- False negatives for serious conditions (especially suicidal)
- Life-threatening implications
- Need for human oversight

**4. Stigmatization**
- Risk of labeling individuals
- Person-first language usage
- Educational approach to mental health

**5. Over-reliance on AI**
- Not a replacement for professional diagnosis
- Clear disclaimers required
- Encouragement to seek professional help

### Recommendations

**For Deployment:**
- Never use as standalone diagnostic tool
- Always include mental health professional oversight
- Implement multi-stage screening process
- Provide immediate crisis resources
- Regular auditing for bias and fairness

**For Users:**
- This is a screening tool, not a diagnosis
- Always consult qualified mental health professionals
- Results should be interpreted with caution
- Seek immediate help if experiencing crisis

## Limitations

1. **Model Limitations:**
   - Training data may not represent all populations
   - Cultural variations in mental health expression
   - Limited to text-based analysis only

2. **Technical Limitations:**
   - BERT requires significant computational resources
   - Processing time increases with dataset size
   - GPU recommended for BERT training

3. **Scope Limitations:**
   - Cannot replace professional mental health assessment
   - No consideration of individual history or context
   - Text-only analysis (no voice, facial expression, etc.)

## Future Work

### Model Improvements
- Implement ensemble methods combining all three models
- Use Mental-BERT (domain-specific pretrained model)
- Hyperparameter optimization
- Train on full dataset with more epochs
- Multi-task learning for severity prediction

### Data Enhancements
- Collect more diverse demographic data
- Include temporal tracking of mental health changes
- Multi-modal data integration (voice, facial expressions)
- Fine-grained severity classifications

### Deployment
- Develop REST API for model serving
- Create web interface for user interaction
- Implement real-time monitoring
- Add human-in-the-loop review system
- Crisis detection and intervention protocols

## Crisis Resources

If you or someone you know is experiencing a mental health crisis:

- **National Suicide Prevention Lifeline (USA):** 988
- **Crisis Text Line:** Text HOME to 741741
- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

## Academic Context

**Course:** CC6057NI - Applied Machine Learning  
**Institution:** London Metropolitan University  
**Assessment:** Text Classification Coursework

## Requirements Met

This project fulfills all coursework requirements:
- Implementation of multiple ML models (3 models: LR, RF, BERT)
- Text classification on selected dataset
- Model evaluation and comparison
- Comprehensive visualizations
- Ethical considerations discussion
- Well-documented code and explanations


## Acknowledgments

- London Metropolitan University for course structure and guidance
- HuggingFace for dataset hosting and transformer libraries
- Open-source community for ML tools and libraries

## 👤 Author

**SabinAdhikari**  
GitHub: [SabinAdhikarii](https://github.com/SabinAdhikarii)

---

## 📞 Support

For questions or issues, please contact me at sabinofficial99@gmail.com).

---

**Last Updated:** January 2026

## License

This project is for educational and demonstration purposes. The Sanima bank dataset has been used. You can use the dataset directltly. The original source of the dataset is scrapped data from [NEPSE](https://nepsealpha.com/stocks/SANIMA/info?utm_source=copilot.com)  



**Happy Analyzing! 📈**



**Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Active Development
## Disclaimer

**IMPORTANT:** This project is for educational and research purposes only. The models developed should NOT be used for clinical diagnosis or as a substitute for professional mental health care. If you are experiencing mental health issues, please consult with a qualified mental health professional immediately.
