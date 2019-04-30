wget 'https://www.dropbox.com/s/c26servnb27mc8s/ensemble_lstm_4_layer_v2.h5?dl=1' -O 'ensemble_lstm_4_layer_v2.h5'
wget 'https://www.dropbox.com/s/myjjk3cpy66av9r/token_test_x_v3.npy?dl=1' -O 'token_test_x_v3.npy'

wget 'https://www.dropbox.com/s/d45citgvug9rxw3/word2vec_model_v3.model?dl=1' -O 'word2vec_model_v3.model'
wget 'https://www.dropbox.com/s/hudhtrok2akz95y/word2vec_model_v3.model.trainables.syn1neg.npy?dl=1' -O 'word2vec_model_v3.model.trainables.syn1neg.npy'
wget 'https://www.dropbox.com/s/fjj6a2vwi54dzhx/word2vec_model_v3.model.wv.vectors.npy?dl=1' -O 'word2vec_model_v3.model.wv.vectors.npy'

python3 test.py $1 $2 $3
