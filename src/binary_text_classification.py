"""
Assignment 2 for language analytics (S2023)

This script runs the text classification pipeline, which includes:
    - loading the data
    - vectorizing the data
    - splitting the data into train and test
    - training the model
    - predicting the labels for the test data
    - saving the metrics
    - saving the vectorizer and classifier (optional)

Author: Laura Bock Paulsen (202005791)
"""

from pathlib import Path
import pandas as pd
import argparse
from joblib import dump
from time import perf_counter
from text_clf import TextClassification

# importing logging module to send messages to the console
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vec_type', type=str, default='tfidf', help='vectorizer type (can be tfidf or count)')
    parser.add_argument('--clf_type', type=str, default='mlp', help='classifier type (can be logistic or mlp)')
    parser.add_argument('--save_models', type=bool, default=False, help='save the vectorizer and classifier (default: False)')

    # check that the input is valid
    args = parser.parse_args()
    if args.vec_type not in ['tfidf', 'count']:
        raise ValueError('vec_type must be either tfidf or count')
    
    if args.clf_type not in ['logistic', 'mlp']:
        raise ValueError('clf_type must be either logistic or mlp')
    
    return args

def plot_history(history, save_path=None):
    """
    Plots the training and validation accuracy and loss.
    """
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    ax[0].plot(history.history['accuracy'], label='train')
    ax[0].plot(history.history['val_accuracy'], label='validation')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    ax[1].plot(history.history['loss'], label='train')
    ax[1].plot(history.history['val_loss'], label='validation')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    

def main():
    start = perf_counter() # for timing the script execution
    path = Path(__file__)
    # parse the input
    args = input_parse()
    
    # logging setup
    logger = logging.getLogger()
    logger.info(f'Binary text decoding with vec_type = {args.vec_type}, clf_type = {args.clf_type}')

    # load the data
    data_path = path.parents[1] / 'in' / 'fake_or_real_news.csv'
    df = pd.read_csv(data_path)

    # splitting the data into X (the text) and y (the label)
    X, y = df['text'], df['label']

    # initialize the TextClassification class 
    tclf = TextClassification(vec_type = args.vec_type, clf_type = args.clf_type)

    # run the model (splits the data into train and test)
    y_pred, y_test, history = tclf.train_test_predict(X, y)

    # plot the training and validation accuracy and loss
    plot_path = path.parents[1] / 'out' / f'{args.vec_type}_{args.clf_type}_history.png'
    plot_history(history, save_path = plot_path)

    # get the classification report
    report = tclf.get_metrics(y_pred, y_test)

    # save report as txt file
    report_path = path.parents[1] / 'out' / f'{args.vec_type}_{args.clf_type}_report.txt'
    
    with open(report_path, 'w') as f:
        f.write(report)

    # save models if specified
    if args.save_models:
        model_path = path.parents[1] / 'models'
        dump(tclf.vec, model_path / f'vectorizer_{args.vec_type}.pkl')
        dump(tclf.clf, model_path / f'classifier_{args.clf_type}.pkl')
        
    end = perf_counter()
    logger.info(f'Finished in {end-start} seconds')

if __name__ == '__main__':
    main()