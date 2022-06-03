from json import encoder
import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from abc import abstractmethod, abstractstaticmethod

import utils
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNDataset

transformers.utils.logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

### Config ###
HIDDEN_DIM = 768
HIDDEN_ACTIVATION = nn.ReLU

# TODO move this into init of roberta classifier class
roberta_config = AutoConfig.from_pretrained(
    "roberta-base", output_hidden_states=True, num_labels=3, finetuning_task="sst3"
)
roberta_tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base", truncation=True, max_length=128, padding="max_length"
)


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


class Attention_Concepts(torch.nn.Module):
    def __init__(self, input_size, n_concepts, device=torch.device("cpu")):
        """
        implementation of self-attention.
        """
        super().__init__()
        self.n_concepts = n_concepts

        self.ff = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, 1, bias=False) for k in range(n_concepts)]
        ).to(device)

    def forward(self, input_, mask=None):
        """
        input vector: input_
        output:
            attn_weights: attention weights
            attn_ctx_vec: context vector
        """
        batch_size = input_.size(0)

        attn_weight = []
        attn_ctx_vec = []
        for k in range(self.n_concepts):
            attn_ = self.ff[k](input_).squeeze(2)
            if mask is not None:
                attn_ = attn_.masked_fill(mask == 0, -1e9)
            attn_ = torch.softmax(attn_, dim=1)
            ctx_vec = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)
            attn_weight.append(attn_)
            attn_ctx_vec.append(ctx_vec)

        attn_weight = torch.cat(attn_weight, 0).view(self.n_concepts, batch_size, -1)
        attn_weight = attn_weight.transpose(0, 1)
        attn_ctx_vec = torch.cat(attn_ctx_vec, 0).view(self.n_concepts, batch_size, -1)
        attn_ctx_vec = attn_ctx_vec.transpose(0, 1)

        return attn_weight, attn_ctx_vec


class AttentionSelf(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, dropout_rate=None, device=torch.device("cpu")
    ):
        """
        implementation of self-attention.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.device = device

        self.ff1 = torch.nn.Linear(input_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, 1, bias=False)
        if dropout_rate is not None:
            self.model_drop = torch.nn.Dropout(dropout_rate)

    def forward(self, input_, mask=None):
        """
        input vector: input_
        output:
            attn_: attention weights
            ctx_vec: context vector
        """
        attn_ = torch.tanh(self.ff1(input_))
        attn_ = self.ff2(attn_).squeeze(2)
        if mask is not None:
            attn_ = attn_.masked_fill(mask == 0, -1e9)
        # dropout method 1.
        # if self.dropout_rate is not None:
        #     drop_mask = Variable(torch.ones(attn_.size())).to(self.device)
        #     drop_mask = self.model_drop(drop_mask)
        #     attn_ = attn_.masked_fill(drop_mask == 0, -1e9)

        attn_ = torch.softmax(attn_, dim=1)
        # dropout method 2.
        if self.dropout_rate is not None:
            attn_ = self.model_drop(attn_)
        ctx_vec = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)

        return attn_, ctx_vec


class PoolingModuleAAN(PoolingModuleBase):
    def __init__(self, n_concepts=10):
        super().__init__()
        self.n_concepts = n_concepts
        self.abs = Attention_Concepts(
            input_size=768, n_concepts=self.n_concepts, device=device
        ).to(device)

        self.agg = AttentionSelf(input_size=768, hidden_size=768, device=device).to(
            device
        )

    def forward(self, reps):
        """Uses a concept-based abstraction-aggregation network over all transformer output reps."""
        # TODO(atharva): Implement AAN layers

        attn_cpt, ctx_cpt = self.abs(reps)
        attn, ctx = self.agg(ctx_cpt)
        output = {
            "attn": attn,
            "ctx": ctx,
            "attn_concept": attn_cpt,
            "ctx_concept": ctx_cpt,
        }

        return output


### build_dataset function definitions ###


def _build_dataset_lstm(self, X, y):
    # Split each example sentence by whitespace for simplicity
    X = [sentence.split() for sentence in X]

    # Prepare sequences by
    new_X = []
    seq_lengths = []
    index = dict(zip(self.vocab, range(len(self.vocab))))
    unk_index = index["$UNK"]
    for ex in X:
        seq = [index.get(w, unk_index) for w in ex]
        seq = torch.tensor(seq)
        new_X.append(seq)
        seq_lengths.append(len(seq))
    seq_lengths = torch.tensor(seq_lengths)

    if y is None:
        return TorchRNNDataset(new_X, seq_lengths)
    else:
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        y = [class2index[label] for label in y]
        return TorchRNNDataset(new_X, seq_lengths, y)


def _build_dataset_roberta(self, X, y):
    data = roberta_tokenizer.batch_encode_plus(
        X,
        max_length=None,
        add_special_tokens=True,
        padding="longest",
        return_attention_mask=True,
    )
    indices = torch.tensor(data["input_ids"])
    mask = torch.tensor(data["attention_mask"])
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


### Concept interpretability function definitions ###
def __get_concept_explanation(self, input):
    # TODO Implement the probabilistic equations to get the scores and keywords
    raise NotImplementedError


### Modular sentiment classifier module definition ###


class SentimentClassifierModel(nn.Module):
    def __init__(self, n_classes, encoder_module, pooling_module):
        super().__init__()
        self.n_classes = n_classes
        self.encoder_module = encoder_module
        self.pooling_module = pooling_module
        self.classifier_module = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            HIDDEN_ACTIVATION(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, *args):
        reps = self.encoder_module(*args)
        pooled = self.pooling_module(reps)
        return self.classifier_module(pooled)


### Classifiers (follow the Scikit-Learn model API) ###

## Base ##
class SentimentClassifierBase(TorchShallowNeuralClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_module = self.build_encoder_module()
        self.pooling_module = self.build_pooling_module()
        self.encoder_module.train()
        self.pooling_module.train()

    def build_graph(self):
        return SentimentClassifierModel(
            self.n_classes_, self.encoder_module, self.pooling_module
        ).to(device)

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


## LSTM ##
class EncoderModuleLSTM(nn.Module):
    """Use an LSTM as an encoder by returning the hidden states."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(HIDDEN_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM)

    def forward(self, X, seq_lengths):
        X = self.embedding(X)
        embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            X,
            batch_first=True,
            lengths=seq_lengths.cpu(),  # TODO maybe try no cpu
            enforce_sorted=False,
        )
        outputs, _ = self.lstm(embeddings)
        return outputs


class SentimentClassifierLSTM(SentimentClassifierBase):
    def __init__(self, *args, X_train=[], **kwargs):
        self.vocab = utils.get_vocab(X_train, mincount=2)
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "LSTM (Baseline)"

    def build_dataset(self, X, y=None):
        return _build_dataset_lstm(self, X, y)

    def build_encoder_module(self):
        return EncoderModuleLSTM()

    def build_pooling_module(self):
        return PoolingModuleRNNLast()


class SentimentClassifierLSTMAAN(SentimentClassifierLSTM):
    def __repr__(self):
        return "LSTM (AAN)"

    def build_pooling_module(self):
        return PoolingModuleAAN()


## RoBERTa-Base ##
class SentimentClassifierRoberta(SentimentClassifierBase):
    def __repr__(self):
        return "RoBERTa-Base (Baseline)"

    def build_dataset(self, X, y=None):
        return _build_dataset_roberta(self, X, y)

    def build_encoder_module(self):
        return AutoModel.from_pretrained("roberta-base", config=roberta_config)

    def build_pooling_module(self):
        return PoolingModuleTransformerCLS()


class SentimentClassifierRobertaAAN(SentimentClassifierRoberta):
    def __repr__(self):
        return "RoBERTa-Base (AAN)"

    def build_pooling_module(self):
        return PoolingModuleAAN()


## DynaSent Model 1 ##
class SentimentClassifierDynasent(SentimentClassifierRoberta):
    def __repr__(self):
        return "DynaSent-M1 (Baseline)"

    def build_encoder_module(self):
        return AutoModel.from_pretrained(
            os.path.join("models", "dynasent_model1.bin"), config=roberta_config
        )


class SentimentClassifierDynasentAAN(SentimentClassifierDynasent):
    def __repr__(self):
        return "DynaSent-M1 (AAN)"

    def build_pooling_module(self):
        return PoolingModuleAAN()
