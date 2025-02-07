# WWW2025 challenge- Brick By Brick 2024


## Team onfo

1. **Team Name**: yddm. 

## Requirements

1. **Python Version**: This project requires Python 3.7.6. 
2. **Dependencies**: All third-party packages are listed in `requirements.txt`.

## Data Preparation

1. **Original Feature Data**: Please save the official raw feature data in the following directories:
   - Training data: `./data/train_X`
   - Testing data: `./data/test_X`
   
   Additionally, save the `train_y_v0.1.0.csv` file in the `./data` directory (this file no need to be copied).

## Preprocessing

1. Navigate to the `./code` directory.
2. Execute the following commands:
   ```bash
   python 1-preprocess_test.py
   python 1-preprocess_train.py
   ```
   This will generate intermediate files in the `./data_tmp/test` and `./data_tmp/train` directories, facilitating parallel processing.

## Feature Engineering
   
1. Execute the following commands to generate feature files:
   - For the first version:
     ```bash
     python 2-dataprocess_test_v13.py
     python 2-dataprocess_train_v13.py
     ```
     This will generate feature files in the `./data_fea_sub/v13` directory.
     
   - For the second version:
     ```bash
     python 2-dataprocess_test_v13_2.py
     python 2-dataprocess_train_v13_2.py
     ```
     This will generate feature files in the `./data_fea_sub/v13_2` directory.
     
   - For the third version:
     ```bash
     python 2-dataprocess_test_v13_3.py
     python 2-dataprocess_train_v13_3.py
     ```
     This will generate feature files in the `./data_fea_sub/v13_3` directory.
     
   - For the fourth version:
     ```bash
     python 2-dataprocess_test_v13_4.py
     python 2-dataprocess_train_v13_4.py
     ```
     This will generate feature files in the `./data_fea_sub/v13_4` directory.
     
     And if you want to quick run, you can download data from https://huggingface.co/datasets/Mike19/www2_0_2_4/tree/main which named `data_fea_sub.zip`, you can unzip it into th path `./data_fea_sub`

## Model Training and Inference

1. Navigate to the `./code` directory.
2. Execute the code in the notebook `main.ipynb`. This will generate the following files:
   - In the `./res` folder: `v51_sd1999110_feacut_10drop_merge_v13_cab_5fold_5mean5max_42.csv`
   - In the `./tar` folder: `v51_sd1999110_feacut_10drop_merge_v13_cab_5fold_5mean5max_42.tar.gz`
3. Execute the code in the notebook `infer_stack.ipynb`. This will generate the following files, and this is best score online:
   - In the `./res` folder: `v51_sd1999110_feacut_10drop_merge_v13_cab_5fold_5mean5max_42_stack1bigger.csv`
   - In the `./tar` folder: `v51_sd1999110_feacut_10drop_merge_v13_cab_5fold_5mean5max_42_stack1bigger.tar.gz`

   The `.tar.gz` file is the submission file for online evaluation.
   Different hardware devices, Python versions, and third-party package versions can lead to deviations in training results. I have provided a pre-trained model that you can download and extract to the model root directory. Of course, you can also retrain the model if you prefer.  you can download data from https://huggingface.co/datasets/Mike19/www2_0_2_4/tree/main which named `model.zip`

## Conclusion
1. Follow these steps to prepare your data, process features, and train your model. If you encounter any issues, please refer to the documentation or seek assistance.
