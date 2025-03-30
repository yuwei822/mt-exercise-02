# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marpng/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

## Language Modeling with Pride and Prejudice (Exercise 2)

This repository contains my modified version of the language modeling exercise for Machine Translation (FS 2024, University of Zurich).

### ✅ Dataset
- Used *Pride and Prejudice* by Jane Austen, downloaded from [Project Gutenberg](https://www.gutenberg.org/ebooks/1342).
- Preprocessed using scripts provided (`preprocess_raw.py` and `preprocess.py`).

###  Model Training
I trained 5 different language models with varying hyperparameters:

| Model    | Embedding Size | Hidden Size | Dropout | Epochs | Valid Perplexity |
|----------|----------------|-------------|---------|--------|------------------|
| Model01  | 250            | 250         | 0.5     | 20     | 56.40            |
| Model02  | 250            | 250         | 0.5     | 30     | **56.08**        |
| Model03  | 300            | 300         | 0.5     | 20     | 57.93            |
| Model04  | 250            | 250         | 0.2     | 20     | 61.74            |
| Model05  | 250            | 250         | 0.8     | 20     | 80.38            |

###  Sample Generation
Generated sample text using the best-performing model (`Model_B`) can be found in the `samples` folder.

Example:
impossible
to be among no one opposed on it Addison of vain ; but The meanness of which I have got
it away on either side of my sister than mercenary , or circumstances than the carriage meet in matrimony ,
but she attracted their daughter . <eos> “ Can be scarcely , ” said Mrs. Gardiner , “ supposed you
left yourself . ” “ My dear , ” replied Elizabeth ; “ to be easily advisable to think how
, because we were to be engaged you have got no more about herself . ” “ I have judged
the contents of inclination to assure you , ” said Miss Bingley , “ I believe very different the girl
### Reproduce
**Training Example:**
```bash
python tools/pytorch-examples/word_language_model/main.py \
  --data data/pride --dropout 0.5 --epochs 30 --save models/model02.pt
