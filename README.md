# Different Speech Translation Models Encode and Translate Speaker Gender Differently

This repository contains the code associated with the ACL2025 paper  
[_**Different Speech Translation Models Encode and Translate Speaker Gender Differently**_](link_to_be_added).

## 📦 Getting Started

These instructions will help you set up the environment and run the core experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/dennisfcc/speech-translation-gender.git
cd speech-translation-gender
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Example Usage

The following code examples demonstrate how to replicate the results presented in the paper. 
The process includes data preprocessing, training probe classifiers, and evaluating translations.

### Data Preprocessing

We use the [MuST-C corpus](https://www.sciencedirect.com/science/article/abs/pii/S0885230820300887) 
as our primary dataset. The preprocessing steps are as follows:

1. **Filter Training Data**
Sentences labeled with speaker gender (based on the MuST-Speaker resource) are sampled from the raw training data 
using the `cli/filter_train_data.py` script to create the training and validation datasets.

2. **Filter Test Data**  
From the [MuST-SHE](https://aclanthology.org/2020.acl-main.619/) dataset, we retain only Category 1 sentences spoken 
by speakers labeled as either Male or Female. This is done using the `cli/filter_tst_dev_data.py` script.

3. **Preprocessing**  
All filtered data is preprocessed using the `cli/preprocess_data.py` script, which generates `${*_data_tsv}` 
files containing all the necessary information.


### Extract Hidden States

The first step is to extract hidden states from the ST models.
For the `facebook/seamless-m4t-v2-large` and `johntsi/ZeroSwot-Large_asr-cv_en-to-200` models, 
the `${layer}` parameter can be set to either:
- `post_adapter`: hidden states are extracted **after** the adapter layers (if present) on top of the encoder.
- `pre_adapter`: hidden states are extracted **before** the adapter layers.

For the model `facebook/s2t-medium-mustc-multilingual-st`, only the `post_adapter` setting is supported.
The `--max-seq-len` parameter is always set to `60`, meaning input audio is trimmed to a maximum duration of 60 seconds.

Other useful arguments include:
- `${*_embeddings_h5}` is the path to the HDF5 file where the extracted hidden states will be stored.
- `${lang}` specifies the language code of the input data (`es`, `fr`, `it`).
- `${model}` is the name of the model hosted on the Hugging Face Hub (`facebook/seamless-m4t-v2-large`, 
`johntsi/ZeroSwot-Large_asr-cv_en-to-200`, `facebook/s2t-medium-mustc-multilingual-st`).

```bash
python /path/to/speech-translation-gender/cli/extract_embeddings.py \
  --tsv-path ${*_data_tsv} --output-file ${*_embeddings_h5} \
  --lang ${lang} --model-name ${model_name} \
  --layer ${layer} --num-workers 0 --max-seq-len 60
```

### Train Probe

As a probe, we use an attention-based classifier (see the paper for more details), 
which produces an output for the entire input sequence. 
The training hyperparameters are the same as those used to obtain the final results.

The argument `${saved_probe}` defines the path where the trained probe will be saved during training, and from which 
it will be loaded during evaluation.


```bash
python /path/to/speech-translation-gender/cli/train_probe.py \
  --dataframe-train ${train_data_tsv} --embeddings-train ${train_embeddings_h5} \
  --dataframe-val ${validation_data_tsv} --embeddings-val ${validation_embeddings_h5} \
  --probe attention --level sequence --attention-type scaled_dot \
  --save-probe ${saved_probe} \
  --batch-size 32 --update-frequency 1 --tol 0.00001 --max-iter 40000 --early-stopping 20 \
  --num-layers 1 --num-heads 1 --dropout 0.0 --dropout-att 0.0 --learning-rate 0.0001
```

### Evaluate Probe

The trained probe can be evaluated using the following code.

```
python /path/to/speech-translation-gender/cli/evaluate_probe.py \
  --dataframe ${data_tsv_file} --embeddings ${embeddings_h5_file} \
  --output-tsv ${output_tsv_file} \
  --probe attention --attention-type scaled_dot --num-layers 1 \
  --level sequence --pretrained-probe ${saved_probe} --batch-size 32
```


### Evaluate Translations

To generate and evaluate translations, use the following code.

Translations are saved to the `${output_tsv}` file in TSV format.
- **Gender accuracy** is evaluated using the official MuST-SHE script, 
which requires the `mosesdecoder` for tokenization.
- **Translation quality** is assessed using the `cli/evaluate_translation.py` script.

```bash
# Generate translations
python /path/to/speech-translation-gender/cli/transcribe_data.py \
  --tsv-path ${test_data_tsv} --lang ${lang} --model-name ${model_name} --output-file ${output_tsv} \
  --num-workers 0 --batch-size 1 --max-seq-len 60

# Evaluate gender accuracy
cut -f4 ${output_tsv} | tail -n+2 > __h__
python /path/to/mustshe-directory/MuST-SHE-v1.1-eval-script/mustshe_acc_v1.1.py \
  --input <(perl /path/to/mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${lang} -q -no-escape < __h__) \
  --tsv-definition ${output_tsv}
rm __h__

# Evaluate translation quality
python /path/to/speech-translation-gender/cli/evaluate_translation.py --file-path ${output_tsv} 
```


## 📄 Citing the Paper

```
@inproceedings{fucci-et-al-2025-different,
title = "Different Speech Translation Models Encode and Translate Speaker Gender Differently",
author = {Fucci, Dennis and Gaido, Marco and Negri, Matteo and Bentivogli, Luisa and 
Martins, André F. T. and Attanasio, Giuseppe},
booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics",
year = "2025",
address = "Vienna, Austria",
publisher = "Association for Computational Linguistics"
}
```
