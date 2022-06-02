from json import encoder
import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig

import utils
from torch_rnn_classifier import TorchRNNModel
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

transformers.utils.logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

### Config ###
HIDDEN_DIM = 768
HIDDEN_ACTIVATION = nn.ReLU

# TODO move this into init of roberta classifier class
roberta_config = AutoConfig.from_pretrained(
    "roberta-base",
    output_hidden_states=True,
    num_labels=3,
    finetuning_task='sst3')
roberta_tokenizer = AutoTokenizer.from_pretrained(
    'roberta-base',
    truncation=True,
    max_length=128,
    padding='max_length')

# Modular sentiment classifier definition


class SentimentClassifierModel(nn.Module):
    def __init__(self, n_classes, encoder_module, pooling_module):
        super().__init__()
        self.n_classes = n_classes
        self.encoder_module = encoder_module
        self.pooling_module = pooling_module
        self.classifier_module = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            HIDDEN_ACTIVATION(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM))

    def forward(self, *args):
        reps = self.encoder_module(*args)
        pooled = self.pooling_module(reps)
        return self.classifier_module(pooled)

# Classifiers (follow the Scikit-Learn model API).


class SentimentClassifierBase(TorchShallowNeuralClassifier):
    def __init__(self, encoder_module=None, pooling_module=None, name=None, *args, **kwargs):
        self.encoder_module = encoder_module
        encoder_module.train()
        self.pooling_module = pooling_module
        self.name = name
        super().__init__(*args, **kwargs)

    def build_graph(self):
        return SentimentClassifierModel(self.n_classes_, self.encoder_module, self.pooling_module).to(device)

    def build_dataset(self, X, y=None):
        """Dataset processing, e.g. tokenizing text."""
        raise NotImplementedError

    def __repr__(self):
        return self.name


class SentimentClassifierRNN(SentimentClassifierBase):
    def build_dataset(self, X, y=None):
        # TODO see build_dataset and _prepare_sequences in TorchRNNClassifier
        X = [x.split() for x in X]
        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
        return dataset


class SentimentClassifierRoberta(SentimentClassifierBase):
    def build_dataset(self, X, y=None):
        data = roberta_tokenizer.batch_encode_plus(
            X,
            max_length=None,
            add_special_tokens=True,
            padding='longest',
            return_attention_mask=True)
        indices = torch.tensor(data['input_ids'])
        mask = torch.tensor(data['attention_mask'])
        if y is None:
            dataset = torch.utils.data.TensorDataset(indices, mask)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(indices, mask, y)
        return dataset

# Pooling layer definitions


class PoolingModuleBase(nn.Module):
    def forward(self, reps):
        raise NotImplementedError


class PoolingModuleRNNLast(PoolingModuleBase):
    def forward(self, reps):
        """Takes the final hidden output as the pooled output."""
        return reps[-1]


class PoolingModuleTransformerCLS(PoolingModuleBase):
    def forward(self, reps):
        """Takes the last-layer CLS rep as the pooled output."""
        return reps.last_hidden_state[:, 0, :]


class PoolingModuleRNNAAN(PoolingModuleBase):
    def forward(self, reps):
        """Uses a concept-based abstraction-aggregation network over all RNN output reps."""
        # TODO(atharva): Implement AAN layers
        raise NotImplementedError


class PoolingModuleTransformerAAN(PoolingModuleBase):
    def forward(self, reps):
        """Uses a concept-based abstraction-aggregation network over all transformer output reps."""
        # TODO(atharva): Implement AAN layers
        # raise NotImplementedError
        return reps.last_hidden_state[:, 0, :]


def build_untrained_classifier_models(X_train, transformer_hyperparams, lstm_hyperparams):
    """Instantiate our different models."""
    vocab = utils.get_vocab(X_train, mincount=2)

    encoder_lstm = None  # TODO define LSTM
    # TorchRNNModel(
    #     vocab_size=len(vocab),
    #     use_embedding=self.use_embedding,
    #     embed_dim=self.embed_dim,
    #     rnn_cell_class=self.rnn_cell_class,
    #     hidden_dim=self.hidden_dim,
    #     bidirectional=self.bidirectional,
    #     freeze_embedding=self.freeze_embedding)
    encoder_roberta = AutoModel.from_pretrained(
        'roberta-base', config=roberta_config)
    encoder_dynasent = AutoModel.from_pretrained(os.path.join(
        'models', 'dynasent_model1.bin'), config=roberta_config)

    pooler_lstm_last = PoolingModuleRNNLast()
    pooler_transformer_cls = PoolingModuleTransformerCLS()
    pooler_lstm_aan = PoolingModuleRNNAAN()
    pooler_transformer_aan = PoolingModuleTransformerAAN()


    sentiment_classifier_lstm_base = None # SentimentClassifierRNN(
        # encoder_lstm.detach().clone(), pooler_lstm_last, kwargs=lstm_hyperparams)
    sentiment_classifier_roberta_base = SentimentClassifierRoberta(
        AutoModel.from_pretrained('roberta-base', config=roberta_config),
        pooler_transformer_cls, 'sentiment_classifier_roberta_base', **transformer_hyperparams)
    sentiment_classifier_dynasent_base = SentimentClassifierRoberta(
        AutoModel.from_pretrained(os.path.join('models', 'dynasent_model1.bin'), config=roberta_config),
        pooler_transformer_cls, 'sentiment_classifier_dynasent_base', **transformer_hyperparams)
    sentiment_classifier_lstm_aan = None # SentimentClassifierRNN(
        # encoder_lstm.detach().clone(), pooler_lstm_aan, kwargs=lstm_hyperparams)
    sentiment_classifier_roberta_aan = SentimentClassifierRoberta(
        AutoModel.from_pretrained('roberta-base', config=roberta_config),
        pooler_transformer_aan, 'sentiment_classifier_roberta_aan', **transformer_hyperparams)
    sentiment_classifier_dynasent_aan = SentimentClassifierRoberta(
        AutoModel.from_pretrained(os.path.join('models', 'dynasent_model1.bin'), config=roberta_config),
        pooler_transformer_aan, 'sentiment_classifier_dynasent_aan', **transformer_hyperparams)
        
    return (sentiment_classifier_lstm_base,
        sentiment_classifier_roberta_base,
        sentiment_classifier_dynasent_base,
        sentiment_classifier_lstm_aan,
        sentiment_classifier_roberta_aan,
        sentiment_classifier_dynasent_aan)
