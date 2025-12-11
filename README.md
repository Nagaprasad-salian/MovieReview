# Sentiment Analysis on IMDB Movie Reviews
Machine Learning Project – Logistic Regression + TF-IDF

## 1. Project Overview
This project performs sentiment analysis on the IMDB Movie Reviews dataset. 
The objective is to classify each review as Positive or Negative using machine learning techniques.

The project includes:
- Data loading  
- Text preprocessing  
- TF-IDF feature extraction  
- Model training (Logistic Regression)  
- Performance evaluation  
- Saving the trained model  
- Exporting predictions  
- Generating a final project report  

This is a complete end-to-end NLP workflow.

---

## 2. Dataset
**IMDB 50K Movie Reviews Dataset**  
Download from Kaggle:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset contains:
- 50,000 reviews  
- Labels: `positive` and `negative`  
- Balanced dataset (25k each)

> Note: The dataset is NOT uploaded to GitHub because GitHub’s file upload limit is 25MB.
> Please download it manually and place it in the project folder as:
> `IMDB Dataset.csv`

---

## 3. Data Preprocessing
The following preprocessing steps were applied:
- Removal of HTML tags  
- Lowercasing of text  
- Removal of URLs  
- Removal of punctuation  
- Removal of numbers  
- Collapsing multiple spaces  

A new column `cleaned_review` is created and used for model training.

---

## 4. Feature Extraction – TF-IDF
TF-IDF converts the cleaned text into numerical features.

**Parameters used:**
- `max_features = 20000`
- `ngram_range = (1, 2)` → uses unigrams + bigrams  
- English stopword removal enabled  

TF-IDF helps highlight important sentiment words such as “amazing”, “boring”, “excellent”, etc.

---

## 5. Machine Learning Model
The chosen classifier is:

### Logistic Regression

**Why Logistic Regression?**
- Excellent for binary classification  
- Works very well with TF-IDF vectors  
- Fast to train  
- Handles high-dimensional sparse data  
- Easy to interpret  

Pipeline used in the project:


---

## 6. Evaluation Metrics
The model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Classification Report  
- Confusion Matrix  
- ROC-AUC Curve  

These metrics ensure the model performs reliably on unseen test reviews.

---

## 7. Saved Files
During execution, the following files are generated:

- `imdb_sentiment_pipeline.pkl` → Saved trained model  
- `imdb_test_predictions.csv` → Predictions on the test set  
- Confusion matrix visualization  
- ROC curve visualization  
- Auto-generated project report  

---
## 8. Summary

This project builds a complete sentiment analysis system using the IMDB dataset.
With TF-IDF and Logistic Regression, the model achieves strong performance with high accuracy and balanced precision and recall.

The pipeline demonstrates real-world machine learning skills:

Cleaning text

Engineering features

Training models

Evaluating results

Saving models

Reporting outcomes

## 9. Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Jupyter Notebook

## 10. License

This project is for educational and academic purposes only.
## 11. How to Run the Project

### Step 1 — Clone the repository
```bash
git clone <your-repo-url>
cd <project-folder>


pip install -r requirements.txt

Download the dataset from Kaggle and save it as:

IMDB Dataset.csv

Run the Jupyter Notebook
jupyter notebook


---
