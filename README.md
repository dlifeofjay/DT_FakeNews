Fake News Detection Using Decision Tree Machine Learning
Overview

This project aims to develop a machine learning model capable of identifying fake news and real news articles. The model uses pattern recognition in textual data to differentiate between real news and fake news based on several linguistic features such as word usage, sentence structure, and other textual patterns. The classifier has been trained using a Decision Tree model, a supervised machine learning algorithm that recursively splits the data based on feature values to make predictions.

Project Details
Classification Task

The goal is to classify news articles into two categories:

Real News (1)

Fake News (0)

Model Used: Decision Tree Classifier

Decision Tree is a machine learning algorithm that splits data into subsets based on feature values. Each split corresponds to a decision rule, and the model's structure resembles a tree where leaves represent classifications (Real or Fake News). This model is simple, interpretable, and effective for classification tasks.

Why Decision Tree?

Interpretability: Decision trees provide a clear and visual representation of decision-making processes.

Efficiency: Decision trees perform well even on smaller datasets, and they are fast to train compared to more complex models like neural networks.

Input Data

The model takes textual news articles as input and extracts relevant features based on:

Word frequency

Sentence structure

N-grams (sequences of n words)

Sentiment and lexical diversity

These features help the model make an informed decision about whether an article is real or fake.

Output

The model outputs either a 1 (Real News) or 0 (Fake News) based on the classification result.

Key Features

Pattern Recognition: The model identifies word patterns, n-grams, and syntactic structures that distinguish fake news from real news.

Decision Tree Classifier: The decision tree splits the data into simpler decisions based on word patterns, making it easy to interpret and debug.

Feature Engineering: Features such as word counts, n-grams, sentiment analysis, and lexical diversity are extracted from the news articles to enhance model performance.

Installation

To run the project locally, follow these steps:

Clone the repository:

git clone https://github.com/dlifeofjay/DT_FakeNews
cd DT_FakeNews


Install required dependencies:

pip install -r requirements.txt


Run the app:

streamlit run DT_app.py

Model Performance

The model’s performance is evaluated using common classification metrics such as:

Accuracy: How often the model's predictions match the actual labels.

Precision: The proportion of true positive results among the predicted positives.

Recall: The proportion of true positive results among the actual positives.

F1-Score: A weighted average of precision and recall, providing a balance between the two.

For more detailed information on the model’s evaluation, performance metrics, and results, please refer to the PDF documentation provided in the repository.

File Structure

DT_app.py: Main script for running the model using Streamlit, which provides an interactive web interface for prediction.

requirements.txt: List of Python libraries required to run the project.

DTFN.ipynb: Jupyter notebook where the Decision Tree model is trained, tested, and evaluated.

DTFN_model.joblib: Saved trained Decision Tree model.

DTFN_vectorizer.joblib: Saved text vectorizer (e.g., CountVectorizer or TfidfVectorizer) for transforming news articles into numerical features.

Additional Information

For further insights into the structure of the model, its performance evaluation, and the methodology behind its development, please refer to the PDF documentation file included in the repository. The document provides a comprehensive explanation of:

How the model works

The decision tree structure

Evaluation metrics and results

Possible improvements and future work
