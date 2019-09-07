## Innoplexus Hackathon: Sentiment Analysis

### Introduction

The goal of the challenge was to classify the sentiment of a piece of text as `positive`, `negative` or `neutral`.  However, the problem statement came with a small twist. Each piece of text was a review related to pharmaceutical drug and the sentiment had to be predicted in the context of that drug. It might happen that there are more two or more drugs being referred to in the text, but the sentiment is different in the context of each.

Below is a sample of the training data (text has been truncated to save space):

| unique_hash                              | text                                                         | drug        | sentiment |
| ---------------------------------------- | ------------------------------------------------------------ | ----------- | --------- |
| 16c0970609b9bea8bbb533adf2c61c897424031b | Of course, shortly after I posted my "everything is good" update, I began to develop the Gilenya cough... | gilenya     | 1         |
| 906db737204ead11986d1492af6f004e16a006e3 | Rituxan -this is my main MS drug. Rituxan is NOT technically approved by the FDA (Food and Drug Administration) for multiple sclerosis but itΓÇÖs sister Ocrelizumab is... | ocrelizumab | 2         |
| 603c2f1612eeabcaac016b6da0df4117b6a8ccd8 | No problem. I know how hard and lonely this journey can be, and I'm happy to help. So, his doctors feel the Tecentriq is... | tecentriq   | 0         |

### Approach

#### Finding Optimal Hyperparamters

I started off with finetuning a pretrained BERT base model. I used [PyTorch Transformers Repository](https://github.com/huggingface/pytorch-transformers) by Huggingface. In order to get it working on the current dataset, all I had to do was write a DataProcessor, which could load data for this challenge and make a couple of minor modifications to the code. Given below is the code for the same:

```python
class InnoplexProcessor(DataProcessor):
    """Processor for the Innoplexus data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"), quotechar='"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv"), quotechar='"'), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0] # "%s-%s" % (set_type, i)
            text_a = line[1]
            # text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

```

In the beginning I finetuned a BERT base model using the defaults provided in the repository. I trained the model for 3 epochs with a batch size of 16. This resulted in a public score of 0.525 and a private score of 0.549. I continued to experiment with the base and large models and tried out various hyperparameters like the number of epochs, learning rate, batch size and sequence length. From my experiments, I concluded that finetuning the model for large number of epochs (> 5) and using large batch sizes (> 16) hurt performance. Also, large model didn't give any significant improvements over the base model. 

#### Preventing Greedy Tokenization

After the initial experiments with model and hyperparameter selection, I decided to look at the data and also read up on and understand how BERT's tokenizer parsed the names of the drugs (since they weren't a part of its vocabulary). I came across this brilliant [tutorial by Chris McCormick](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/). Based on this I came to know that BERT's tokenizer was greedily parsing the drug names and breaking them up into smaller chunks that are in it's vocabulary. I wanted to see if results would improve if I would prevent this from happening. For this I used two different approaches. First, I added the names of all the drugs in the training set to the vocabulary. This resulted in a public score of 0.521 and a private score of 0.527. 

#### Downsampling

Before experimenting with the second approach, I looked at the class distribution in the data. What I found was that there was a severe class imbalance. 

The distribution of classes was as given below:

| Class ID | Number of Samples | Percentage |
| -------- | ----------------- | ---------- |
| 2        | 3825              | 72.46      |
| 1        | 837               | 15.86      |
| 0        | 617               | 11.69      |

As it can be seen above, there a too many instances of class 2 (which was for neutral sentiment) compared to the other 2 classes. To make the class distribution balanced, I downsampled the dataset by selecting 837 rows of class 2 at random. 

Now, we come to the second approach for preventing greedy tokenization. The trick was to substitute the name of the drug with a word that exists in BERT's vocabulary but not in the text. So, I substituted the names of the drugs by the word "mango". However, so that the context is maintained (as discussed in the beginning), for each text, only the drug, in whose context the sentiment was to predicted, was substituted by "mango". After this is how the training data looked like:



| unique_hash                              | text                                                         | drug  | sentiment |
| ---------------------------------------- | ------------------------------------------------------------ | ----- | --------- |
| 16c0970609b9bea8bbb533adf2c61c897424031b | Of course, shortly after I posted my "everything is good" update, I began to develop the mango cough... | mango | 1         |
| 906db737204ead11986d1492af6f004e16a006e3 | Rituxan -this is my main MS drug. Rituxan is NOT technically approved by the FDA (Food and Drug Administration) for multiple sclerosis but itΓÇÖs sister mango is... | mango | 2         |
| 603c2f1612eeabcaac016b6da0df4117b6a8ccd8 | No problem. I know how hard and lonely this journey can be, and I'm happy to help. So, his doctors feel the mango is... | mango | 0         |

With this, and downsampling the dataset, the public score improved to 0.529 and private score remained more or less the same.

#### Plus Ultra

To go beyond, I downloaded [Rotten Tomatoes](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) sentiment analysis data from Kaggle. Training a BERT base model with the additional data gave a public score of 0.533 and a private score of 0.553. This led to a rank of 19 on the [public leaderboard](https://datahack.analyticsvidhya.com/contest/innoplexus-online-hiring-hackathon/lb) and a rank of 11 on the [private leaderboard](https://datahack.analyticsvidhya.com/contest/innoplexus-online-hiring-hackathon/pvt_lb).



