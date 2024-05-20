# AI System for Primer Design   

## Autoencoder
1. Train encder and decoder.
```
train.py [-h] [-k K_FOLDS]
        [-e NUM_EPOCHS]
        [-b BATCH_SIZE]
        [-l LEARNING_RATE]
        [--embedding_dim EMBEDDING_DIM]
        [--hidden_dim HIDDEN_DIM]
        [--plot_every PLOT_EVERY]
        [--data_path DATA_PATH]
        [--word_dict WORD_DICT]
        [--debug DEBUG
```

2. Regress CT values.
```
regress.py [-h] [-k K_FOLDS]
          [-e NUM_EPOCHS]
          [-b BATCH_SIZE]
          [-l LEARNING_RATE]
          [--embedding_dim EMBEDDING_DIM]
          [--hidden_dim HIDDEN_DIM]
          [--plot_every PLOT_EVERY]
          [--data_path DATA_PATH]
          [--word_dict WORD_DICT]
          [--debug DEBUG]
```

3. Inference CT values.
```
inference.py [-h] [-k K_FOLDS]
            [-b BATCH_SIZE]
            [--embedding_dim EMBEDDING_DIM]
            [--hidden_dim HIDDEN_DIM]
            [--data_path DATA_PATH]
            [--word_dict WORD_DICT]
            [--debug DEBUG]
```

## Multi-input CNN

### Two-step classifier-regressor
0. Generate binary 'label' for data with NaN
1. Train classifier
    ```
    python train_cnn_multi.py \
        --data_path='./data/train_df_with_label.csv' \
        --loss_function='bce_loss' \
        -e 1000 \
        --target_name='label' \
        --patience 10
    ```
2. Inference with classifier (you may use '--model_path') -> produce 'train/test_df_wtih_label_no_nan.csv'
    ```
    python inference_cnn_classifier.py \
        --data_path='./data/train_df_with_label.csv' \
        --target_name='label'    
    python inference_cnn_classifier.py \
        --data_path='./data/test_df_with_label.csv' \
        --target_name='label'    
    ```
3. Train regressor without NaN (predicted) data
    ```
    python train_cnn_multi.py --data_path='./data/train_df_with_label_no_nan.csv' \
        -e 1000 \
        --patience 20
    ```
4. Inference with regressor on train set for qualitative analysis (you may use '--model_path')
    ```
    python inference_cnn_multi.py --data_path='./data/train_df_with_label_no_nan.csv'
    ```
5. Inference with regressor on test set for predicting ct value (you may use '--model_path')
    ```
    python inference_cnn_multi.py --data_path='./data/test_df_with_label_no_nan.csv'
    ```