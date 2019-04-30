wget 'https://www.dropbox.com/s/mip9g3yq7j5kdzy/token_train_x_v3.npy?dl=1' -O 'token_train_x_v3.npy'

python3 train.py $1 $2 $3 $4
