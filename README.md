# how to use
download the code and follow the next steps:

1.prepare data

Download tiny-imagenet-200 and unzip it. Change the data path in utils.py. Notice that we only use train data here. Then run the processdata.py and you will get two folder 'owndataset1' and 'owndataset2', and there just one class in them with 100000 64*64 and 50000 32*32 images.

2.train 

Just run training_vqgan.py. Then you should change the args in training_transformer.py. The 'checkpoint_path' should be changed. Ten run training_transformer.py.

3 hyperparameters

In training_vqgan.py:

batch_size
epochs
disc_start:it means when we start use the discriminator. If it is too small the decoder is too poor and the training will be useless. If it is too large, the training may not get good results.

In training_transformer:

batch_size
epochs
temperature:it means the random degree when sampling
top_k:it means the model's freedom when sampling, so it can't be too small or too big.


