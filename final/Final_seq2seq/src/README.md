# Video Caption via Seq2seq Model
### Requirements
1. numpy == 1.14.3
2. pandas == 0.23.4
3. keras == 2.1.6
4. nltk == 3.3
5. pyenchant == 2.0.0
### Prerequisite
Please download the following files of the datasets (https://www.kaggle.com/c/2018-ml-final-video-caption/data).
> P.S. Specific names of all the directories and the files are as same as the datasets mentioned above.
> Assume the dataset is downloaded to the directory ".../"
### Directories and files
```
$ tree
.
├── src
│   ├── savings
│   │   ├── opt2idx.npy
│   │   └── test_caption.npy
│   ├── text_preprocessing_savings
│   │   ├── captions.npy
│   │   └── optioons_v2.npy
│   ├── final_project_test.sh
│   ├── final_project_train.sh
│   ├── final_seq.py
│   ├── final_seq_test_02.py
│   └── text_preprocessing_v2.py
├── README.md
└── Report.pdf

3 directories, 11 files
```
### Data Preprocessing
1. Tokenization
2. Spelling correction
3. Removement of stop words
4. Removement of inexisted words
5. Stemming
> Check the saved file 'options_v2.npy' under the directory 'text_preprocessing_savings/', after running:
```Shell
$ python3 text_preprocessing_v2.py <.../training_label.json> <.../testing_options.csv>
```
### Training
```Shell
$ bash final_project_train.sh <.../training_label.json> <.../training_data/feat>
```
### Testing
> Check the saved file 'seq2seq_output.csv' under the current directory after running:
```Shell
$ bash final_project_test.sh <.../testing_options.csv> <.../testing_data/feat>
```
### Evaluation
|Metrics|Training|Validation|Testing (Kaggle Public Score)
|---|---|---|---
|Cross Entropy Loss|1.7561|1.8020|-
|Accuracy|67.87%|68.45%|45.40%
> Kaggle competition link: https://www.kaggle.com/c/2018-ml-final-video-caption
### Reference
- Rafael A. Rivera-Soto, Juanita Ordo´nez. Sequence to Sequence Models for
Generating Video Captions. 2017. Stanford. 
http://cs231n.stanford.edu/reports/2017/pdfs/31.pdf
- Y Guo, B Yao, Y Liu. Sequence to Sequence Model for Video Captioning. 2017.
Stanford.
https://pdfs.semanticscholar.org/eecf/af49500434d91970b24831081d5d2c68697e.pdf
- Quanzeng You, Hailin Jin, Zhaowen Wang, Chen Fang, and Jiebo Luo. Image
Captioning with Semantic Attention. 2016 CVPR.
https://arxiv.org/pdf/1603.03925.pdf

