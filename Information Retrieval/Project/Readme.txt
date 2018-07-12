


DATASET 2 # SENTENCE CLASSIFICATION
Setup:
Split the labeled_articles folder into training_set and test_set folder.
Place 72 files in training_set folder and 18 files in test_set folder.
Place the notebook file in the same folder as the two folders are placed in.
"stopwords" for this problem are given so place stopwords folder in the current folder.
"unlabelled" folder is not used in this exercise as we want to measure the performance of the system with labelled data.

Execution:
1) First all filenames are read from training_set folder and training_set folder.
2) Each file has multiple lines. Each line starts with the "label" and than "sentence" seperated by "tab".
3) Files are read and preprocessing is applied. 
	- stopwords removed
	- label and sentence seperated
3) Labels and sentences are saved in seperate files after preprocessing
4) Sentences are converted into vectors using Sklearn function "TfidfVectorizer()".
5) Labels are read as list and used as labels for the classifiers.
6) 3 builtin SKlearn classifiers are used for this exercise.
	- MultinomialNB (Naive Bayes)
	- NearestCentroid
	- KNeighborsClassifier
7) Files from "training_set" folder are used for training
8) Files from "test_set" folder are used for testing.
9) Evaluation measures used are:
	- Accuracy
	- F1 score - micro
	- F1 score - macro
	- Confusion Matrix
10) It is observed that precision and recall values are same.
11) Results can be seen in the Python notebook.