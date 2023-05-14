# Binary text classification
Assignment 2 for language analytics (S2023). This repository holds the code for training simple binary text classification models using the `scikit-learn` library.


## Usage
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

## Output
The models (vectorizers and classifiers) are saved in the `models` folder. The classification reports are saved in the `out` directory.

## Data
The data used for this assignment is the [Fake or real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle. The data is stored in the `in` directory. 

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
Two models were trained, one being a logistic regression classifier and the other a multi-layer perceptron (MLP) classifier with three 20-unit hidden layers. The models were trained using both count and tf-idf vectorizers.

The two models performed similarly, with the MLP classifier having a slightly worse performance (lower scores for precision of real news and recall of fake news, as well as a slightly lower accuracy).

See the classification reports in the `out` directory for more details.