# Binary text classification
Assignment 2 for language analytics (S2023). This repository holds the code for training binary text classification models using the `scikit-learn` library. More specifically, a logistic regression classifier and a multi-layer perceptron (MLP) classifier were trained using tf-idf vectorizers.

## Description of the data
The data used for this assignment is the [Fake or real news dataset](https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset) from Kaggle. The data is stored in the `in` directory.

## Usage and reproduciblity
To produce the results for this assignment, run the following commands from the root directory of the repository. This creates and activates a virtual environment and runs the script with the desired arguments.
1. Create virtual environment and install dependencies
```
bash setup.sh
```
2. Run the text classification script
```
bash run
```

**Alternatively**, you can run the following commands manually.

1. Create the virtual environment (if not already created):
```
bash setup.sh
```

2. Run the script with the desired arguments
```
python src/binary_text_classification.py --clf_type <model_type> --vec_type <vectorizer_type> --save_models <True/False>
```

### Arguments
- `clf_type`: The type of classifier to use. Can be either `mlp` or `logistic`.
- `vec_type`: The type of vectorizer to use. Can be either `count` or `tfidf`.
- `save_models`: Whether to save the trained models. Can be either `True` or `False`.


## Repository overview
```
├── in
│   └── fake_or_real_news.csv               <- Training data
├── models                                 
│   ├── classifier_logistic.pkl             <- Logistic regression classifier
│   └── ..
├── out
│   ├── count_logistic_report.txt           <- Classification report for logistic regression  
│   └── ..
├── src
│   ├── binary_text_classification.py       <- Main script training the models and saving the results
│   └── text_clf.py                         <- Holds the `TextClassifier` class for training and evaluating models
└── README.md                               <- The top-level README for this project
```

## Results
The models (vectorizers and classifiers) are saved in the `models` folder. The classification reports are saved in the `out` directory.

Two models were trained, one being a logistic regression classifier and the other a multi-layer perceptron (MLP) classifier with three 20-unit hidden layers. The models were trained using both count and tf-idf vectorizers.

| Model   | Fake/Real | Precision | Recall | F1-Score | Support |
|---------|-----------|-----------|--------|----------|---------|
| Logistic |   FAKE    |   0.88    |  0.88  |   0.88   |   974   |
| Logistic |   REAL    |   0.88    |  0.88  |   0.88   |   927   |
| MLP |   FAKE    |   0.88    |  0.87  |   0.88   |   974   |
| MLP |   REAL    |   0.86    |  0.88  |   0.87   |   927   |


The two models performed similarly, with the MLP classifier having a slightly worse performance (lower scores for precision of real news and recall of fake news, as well as a slightly lower accuracy).

It might be that the more simple model performs better because the data is not very complex. The MLP classifier might be overfitting the data, which could explain the slightly worse performance. However, the difference in performance is very small, so it is hard to say for sure.