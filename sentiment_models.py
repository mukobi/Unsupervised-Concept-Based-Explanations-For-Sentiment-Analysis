from json import encoder
import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from abc import abstractmethod, abstractstaticmethod


import utils
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


### Modular sentiment classifier definition ###

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


### Pooling layer definitions ###

class PoolingModuleBase(nn.Module):
    @abstractmethod
    def forward(self, reps):
        """
        Given an input-sequence-length array of hidden representations, pool
        them into a fixed-length input for the classification layer.
        """
        pass


class PoolingModuleRNNLast(PoolingModuleBase):
    def forward(self, reps):
        """Takes the final hidden output as the pooled output."""
        return reps[-1]


class PoolingModuleTransformerCLS(PoolingModuleBase):
    def forward(self, reps):
        """Takes the last-layer CLS rep as the pooled output."""
        return reps.last_hidden_state[:, 0, :]


class PoolingModuleAAN(PoolingModuleBase):
    def forward(self, reps):
        """Uses a concept-based abstraction-aggregation network over all transformer output reps."""
        # TODO(atharva): Implement AAN layers
        K = 10  # Hard code number of concepts for now based on whatever the paper uses
        # raise NotImplementedError
        return reps.last_hidden_state[:, 0, :]  # CLS token, will change


### Build Dataset function definitions ###
def _build_dataset_lstm(self, X, y):
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


def _build_dataset_roberta(self, X, y):
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


### Classifiers (follow the Scikit-Learn model API) ###

class SentimentClassifierBase(TorchShallowNeuralClassifier):
    def __init__(self, *args, **kwargs):
        self.encoder_module = self.build_encoder_module()
        self.pooling_module = self.build_pooling_module()
        self.encoder_module.train()
        self.pooling_module.train()
        super().__init__(*args, **kwargs)

    def build_graph(self):
        return SentimentClassifierModel(self.n_classes_, self.encoder_module, self.pooling_module).to(device)

    @abstractmethod
    def build_dataset(self, X, y=None):
        """Dataset processing, e.g. tokenizing text."""
        pass

    @abstractmethod
    def build_encoder_module(self):
        """Instantiate the encoder module once."""
        pass

    @abstractmethod
    def build_pooling_module(self):
        """Instantiate the pooling module once."""
        pass


class SentimentClassifierLSTM(SentimentClassifierBase):
    def __repr__(self):
        return "LSTM (Baseline)"

    def build_dataset(self, X, y=None):
        return _build_dataset_lstm(self, X, y)

    def build_encoder_module(self):
        return None  # TODO build LSTM

        # vocab = utils.get_vocab(X_train, mincount=2)
        # TorchRNNModel(
        #     vocab_size=len(vocab),
        #     use_embedding=self.use_embedding,
        #     embed_dim=self.embed_dim,
        #     rnn_cell_class=self.rnn_cell_class,
        #     hidden_dim=self.hidden_dim,
        #     bidirectional=self.bidirectional,
        #     freeze_embedding=self.freeze_embedding)

    def build_pooling_module(self):
        return PoolingModuleRNNLast()


class SentimentClassifierLSTMAAN(SentimentClassifierLSTM):
    def __repr__(self):
        return "LSTM (AAN)"

    def build_pooling_module(self):
        return PoolingModuleAAN()


class SentimentClassifierRoberta(SentimentClassifierBase):
    def __repr__(self):
        return "RoBERTa-Base (Baseline)"

    def build_dataset(self, X, y=None):
        return _build_dataset_roberta(self, X, y)

    def build_encoder_module(self):
        return AutoModel.from_pretrained(
            'roberta-base', config=roberta_config)

    def build_pooling_module(self):
        return PoolingModuleTransformerCLS()


class SentimentClassifierRobertaAAN(SentimentClassifierRoberta):
    def __repr__(self):
        return "RoBERTa-Base (AAN)"

    def build_pooling_module(self):
        return PoolingModuleAAN()


class SentimentClassifierDynasent(SentimentClassifierRoberta):
    def __repr__(self):
        return "DynaSent-M1 (Baseline)"

    def build_encoder_module(self):
        return AutoModel.from_pretrained(os.path.join(
            'models', 'dynasent_model1.bin'), config=roberta_config)


class SentimentClassifierDynasentAAN(SentimentClassifierDynasent):
    def __repr__(self):
        return "DynaSent-M1 (AAN)"

    def build_pooling_module(self):
        return PoolingModuleAAN()
