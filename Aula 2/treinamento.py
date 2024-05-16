import os 
import argparse 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import make_pipeline 
import mlflow 
import mlflow.sklearn

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", type=str, help="Conjunto de dados a ser utilizado.") 
    args = parser.parse_args()

    mlflow.start_run() 
    mlflow.sklearn.autolog()

    df = pd.read_csv(args.dataset)

    train_df, test_df = train_test_split( 
        df, 
        test_size=0.1, 
        random_state=2 
    ) 
    
    clf = make_pipeline( 
        CountVectorizer( 
            strip_accents="unicode", 
            stop_words="english", 
            ngram_range=(1, 2), 
        ), 
        RandomForestClassifier(n_estimators=5, random_state=2, n_jobs=None, verbose=0) 
    ) 
    
    clf.fit(train_df.text, train_df.sentiment)

    y_true = test_df.sentiment 
    y_pred = clf.predict(test_df.text) 
    
    mlflow.log_metric("test_accuracy_score", accuracy_score(y_true, y_pred)) 
    mlflow.log_metric("test_precision_score", precision_score(y_true, y_pred, average="macro")) 
    mlflow.log_metric("test_recall_score", recall_score(y_true, y_pred, average="macro")) 
    mlflow.log_metric("test_f1_score", f1_score(y_true, y_pred, average="macro")) 
    
    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_, normalize='true') 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    mlflow.log_figure(disp.plot(cmap="Greens").figure_, 'test_confusion_matrix.png')

    mlflow.sklearn.log_model( 
        sk_model=clf, 
        registered_model_name="analise_sentimentos", 
        artifact_path="analise_sentimentos", 
    ) 
    
    mlflow.sklearn.save_model( 
        sk_model=clf, 
        path=os.path.join("analise_sentimentos", "trained_model"), 
    )

    mlflow.end_run()

if __name__ == "__main__":
    main()
