# Hate-Speech-Detection-for-Arabic-text

## Overview:
This repository houses the comprehensive codebase for a rigorous research endeavor targeting hate speech detection within Arabic text on various social media platforms. Employing a multitude of sophisticated models and state-of-the-art techniques, this project aims to advance the domain of hate speech detection for the Arabic language.

Each code file is meticulously annotated with references, shedding light on the underlying methodologies and principles. For traditional ML models such as NB, RF, and SVM, references are embedded directly within the source code. Meanwhile, for deep learning paradigms like CNN, LSTM, and BERT variants, they are conveniently located within the first code cell and are additionally listed at the end of this document.



### Prerequisities:
To successfully run the code, ensure the following:

Python Version: Python 3.x
Execution Environment: A platform supporting Jupyter notebooks (e.g., Jupyter Lab, Jupyter Notebook, Google Colab)
Packages & Libraries: Make sure to install all required packages and libraries. Notably, for the BERT models, both the Accelerate and Transformers libraries are mandatory.

## Data Preparation:
The dataset undergoes a rigorous cleaning process facilitated by the Preprocess_1_.ipynb script. The polished dataset, aptly named 'UseThisClean.csv', serves as the foundation for all subsequent modeling. Ensure this dataset is readily accessible in the execution environment before launching any of the modeling notebooks.


## Model Execution Guide:
Files are systematically named following a convention that reflects their respective models and feature extraction techniques. For example:

NB stands for Naive Bayes
Uni stands for Unigrams
BD signifies a Balanced Dataset
For instance, a file titled UniNBBD employs Naive Bayes on Unigrams extracted from a Balanced Dataset.

To execute any model:

Upload the corresponding .ipynb file to your preferred environment.
Run each cell sequentially, ensuring data paths align as necessary.

## Comprehensive References:

The backbone of this project is a myriad of scholarly articles, tutorials, and resources that offer in-depth perspectives on the techniques employed. A curated list is available below, serving as a testament to the rigorous academic and practical foundations of this research.


Ahamed, S. (2018) Text classification using CNN, LSTM and visualize word embeddings: Part-2, Medium. Available at: https://sabber.medium.com/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d (Accessed: 17 September 2023).
Alshammari, W.T. (2022) How to use Arabic word2vec word embedding with LSTM for sentiment analysis task, Medium. Available at: https://medium.com/@WaadTSS/how-to-use-arabic-word2vec-word-embedding-with-lstm-af93858b2ce (Accessed: 17 September 2023).
Arabic language understanding with Bert - Wissam Antoun (2020) YouTube. Available at: https://www.youtube.com/watch?v=N9pQ3CZ0v6U (Accessed: 17 September 2023).
Awan, A.A. and Navlani, A. (2023) Naive Bayes classifier tutorial: With Python Scikit-Learn, DataCamp. Available at: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn (Accessed: 17 September 2023).
Classify text with Bert  :   text  :   tensorflow (no date) TensorFlow. Available at: https://www.tensorflow.org/text/tutorials/classify_text_with_bert (Accessed: 17 September 2023).
Edureka (2021) Support Vector Machine | SVM Tutorial | Machine learning training | Edureka | ML rewind - 1, YouTube. Available at: https://www.youtube.com/watch?v=6PtblAwRzVg (Accessed: 17 September 2023).
Edureka (2022) Random Forest algorithm | random forest complete explanation | data science training | edureka, YouTube. Available at: https://www.youtube.com/watch?v=3LQI-w7-FuE (Accessed: 17 September 2023).
Khalid, S. (2020) Bert explained: A Complete Guide with Theory and tutorial, Medium. Available at: https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c (Accessed: 17 September 2023).
Naive Bayes classifier in Python (from scratch!) (2021) YouTube. Available at: https://www.youtube.com/watch?v=3I8oX3OUL6I (Accessed: 17 September 2023).
Sentdex (no date) Beginning SVM from Scratch in Python, Python programming tutorials. Available at: https://pythonprogramming.net/svm-in-python-machine-learning-tutorial/ (Accessed: 17 September 2023).
Text classification using Bert & Tensorflow | Deep Learning Tutorial 47 (Tensorflow, Keras & Python) (2021) YouTube. Available at: https://www.youtube.com/watch?v=hOCDJyZ6quA (Accessed: 17 September 2023).
Text classification using CNN with tensorflow 2.1 in python #NLP #tutorial (2020) YouTube. Available at: https://www.youtube.com/watch?v=MsL79ZIqWpg (Accessed: 17 September 2023).
Text classification using neural network | google colab (2020) YouTube. Available at: https://www.youtube.com/watch?v=QE6-FMiRajQ (Accessed: 17 September 2023).
Text pre-processing using NLTK and python | datahour by dr. Nitin Shelke (2022) YouTube. Available at: https://www.youtube.com/watch?v=aF7NvdjSqfo&amp;t=1362s (Accessed: 17 September 2023).
Text preprocessing in NLP | Python (2022) YouTube. Available at: https://www.youtube.com/watch?v=Br5dmsa49wo (Accessed: 17 September 2023).
Analytics Vidhya. (2021). LSTM for Text Classification | Beginners Guide to Text Classification. [online] Available at: https://www.analyticsvidhya.com/blog/2021/06/lstm-for-text-classification/.
‌www.youtube.com. (n.d.). Text Classification using LSTM on Amazon Review Dataset with TensorFlow 2.1 #nlp #tutorial. [online] Available at: https://www.youtube.com/watch?v=DUZn8hMLnbI [Accessed 17 Sep. 2023].
@article{joulin2016bag,
title={Bag of Tricks for Efficient Text Classification},
author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
journal={arXiv preprint arXiv:1607.01759},
year={2016}
}
Ahamed, S. (2018) Text classification using CNN, LSTM and visualize word embeddings: Part-2, Medium. Available at: https://sabber.medium.com/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d (Accessed: 17 September 2023).
Mikolov, T., Chen, K., Corrado, G. and Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. [online] arXiv.org. Available at: https://arxiv.org/abs/1301.3781.
Alshammari, W.T. (2022) How to use Arabic word2vec word embedding with LSTM for sentiment analysis task, Medium. Available at: https://medium.com/@WaadTSS/how-to-use-arabic-word2vec-word-embedding-with-lstm-af93858b2ce (Accessed: 17 September 2023).
@inproceedings{antoun2020arabert,
title={AraBERT: Transformer-based Model for Arabic Language Understanding},
author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
booktitle={LREC 2020 Workshop Language Resources and Evaluation Conference 11--16 May 2020},
pages={9}
}Arabic language understanding with Bert - Wissam Antoun (2020) YouTube. Available at: https://www.youtube.com/watch?v=N9pQ3CZ0v6U (Accessed: 17 September 2023). Khalid, S. (2020) Bert explained: A Complete Guide with Theory and tutorial, Medium. Available at: https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c (Accessed: 17 September 2023). Classify text with Bert  :   text  :   tensorflow (no date) TensorFlow. Available at: https://www.tensorflow.org/text/tutorials/classify_text_with_bert (Accessed: 17 September 2023). Text classification using Bert & Tensorflow | Deep Learning Tutorial 47 (Tensorflow, Keras & Python) (2021) YouTube. Available at: https://www.youtube.com/watch?v=hOCDJyZ6quA (Accessed: 17 September 2023).
@inproceedings{inoue-etal-2021-interplay,
title = "The Interplay of Variant, Size, and Task Type in {A}rabic Pre-trained Language Models",
author = "Inoue, Go  and
Alhafni, Bashar  and
Baimukan, Nurpeiis  and
Bouamor, Houda  and
Habash, Nizar",
booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
month = apr,
year = "2021",
address = "Kyiv, Ukraine (Online)",
publisher = "Association for Computational Linguistics",
abstract = "In this paper, we explore the effects of language variants, data sizes, and fine-tuning task types in Arabic pre-trained language models. To do so, we build three pre-trained language models across three variants of Arabic: Modern Standard Arabic (MSA), dialectal Arabic, and classical Arabic, in addition to a fourth language model which is pre-trained on a mix of the three. We also examine the importance of pre-training data size by building additional models that are pre-trained on a scaled-down set of the MSA variant. We compare our different models to each other, as well as to eight publicly available models by fine-tuning them on five NLP tasks spanning 12 datasets. Our results suggest that the variant proximity of pre-training data to fine-tuning data is more important than the pre-training data size. We exploit this insight in defining an optimized system selection model for the studied tasks.",
}
