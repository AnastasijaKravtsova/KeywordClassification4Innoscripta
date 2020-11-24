# Keywords classification

### Solution

I regarded keyword classification as a sequence labeling task. In this way, it becomes similar to the Named Entity Recognition task.

Each token in the source sequence must be provided with a tag from a predefined tag set (`B-KEY, I-KEY, O`).

The way I solved the task was the following:
* I took [the DeepPavlov NER config based on multilingual BERT](https://github.com/deepmipt/DeepPavlov/blob/0.13.0/deeppavlov/configs/ner/ner_ontonotes_bert_mult.json) and adapted it for keywords labeling,
* prepared the data (preprocessed it and made train/valid/test split in 0.8/0.1/0.1 fraction, see [data_preparation.ipynb](data_preparation.ipynb)),
* trained the model (you can download it from [here](http://lnsigo.mipt.ru/export/models/keyword_model.tar.gz)).

### Results
```bash
{"test": {"eval_examples_count": 647, "metrics": {"ner_f1": 88.2937, "ner_token_f1": 90.2774}, "time_spent": "0:00:08"}}
```

More details about DeepPavlov's multilingual BERT and its application to sequence labeling tasks can be found in [TWS post](https://towardsdatascience.com/19-entities-for-104-languages-a-new-era-of-ner-with-the-deeppavlov-multilingual-bert-1bfa6d413ea6).


### Steps to reproduce the model training:

##### Create and activate a virtual environment for python 3
```bash
python3 -m venv venv
. venv/bin/activate
```

##### Install requirements in your virtual environment
```bash
pip install -r requirements.txt
```

##### Launch model training and evaluation
```bash
python -m deeppavlov train kws_multi_bert.json
```
To run it on a CUDA-compatible GPU one can install `tensorflow-gpu`:
```bash
pip install tensorflow-gpu==1.15.2
```
