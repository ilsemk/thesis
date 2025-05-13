import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from svm import SVMClassifier

def read_data(dataset):
    '''Reads CSV file and extracts the fragments and corresponding authors'''
    print("Read dataset...")

    inputdf = pd.read_csv(dataset, sep=",", encoding="utf-8", header=0)

    fragments = inputdf['fragment'].to_list()
    authors = inputdf['author_label'].to_list()

    assert len(fragments) == len(authors), 'Error: there should be an equal number of authors and fragments'
    print(f'Number of samples: {len(fragments)}')

    return fragments, authors

def evaluate(true_labels, predicted_labels, class_labels=None):
    '''Evaluates the performance of the classification'''
    print("Evaluating...")
    confusion_matrix = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    print("Final evaluation:")
    print(confusion_matrix)

    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy}")

    precision = metrics.precision_score(true_labels, predicted_labels, average='macro', zero_division=0.0)
    print(f"Precision score: {precision}")

    recall = metrics.recall_score(true_labels, predicted_labels, average='macro', zero_division=0.0)
    print(f"Recall score: {recall}")

    f1_score = metrics.f1_score(true_labels, predicted_labels, average='macro')
    print(f"F1-score: {f1_score}")

def main():
    fragments, authors = read_data("data_thesis.csv")

    X_train, X_test, y_train, y_test = train_test_split(fragments, authors, test_size=0.2, random_state=42)

    cls = SVMClassifier()
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)

    evaluate(y_test, y_pred)

    cls.extract_ast(fragments[0])

if __name__ == "__main__":
    main()