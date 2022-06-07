import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from abc import abstractmethod

from lib.torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from lib.aan_attention import AttentionConcepts, AttentionSelf

transformers.utils.logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

### Config ###
HIDDEN_DIM = 768
HIDDEN_ACTIVATION = nn.ReLU
NUM_CONCEPTS = 10
BATCH_SIZE = 16
DIVERSITY_PENALTY_BETA = 1

def phi(text):
    return text


### Pooling layer definitions ###


class PoolingModuleBase(nn.Module):
    @abstractmethod
    def forward(self, reps):
        """
        Given an input-sequence-length array of hidden representations, pool
        them into a fixed-length input for the classification layer.
        """
        pass


class PoolingModuleTransformerCLS(PoolingModuleBase):
    def forward(self, reps):
        """Takes the last-layer CLS rep as the pooled output."""
        return reps.last_hidden_state[:, 0, :]


class PoolingModuleAAN(PoolingModuleBase):
    def __init__(self, n_concepts=NUM_CONCEPTS):
        super().__init__()
        self.n_concepts = n_concepts
        self.abs = AttentionConcepts(
            input_size=HIDDEN_DIM, n_concepts=self.n_concepts).to(device)
        self.agg = AttentionSelf(
            input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM, dropout_rate=0.02).to(device)

    def forward(self, reps):
        """Uses a concept-based abstraction-aggregation network over all transformer output reps."""
        self.attn_abs, self.ctx_abs = self.abs(reps)  # Abstraction
        self.attn_agg, self.ctx_agg = self.agg(self.ctx_abs)  # Aggregation

        return self.ctx_agg


### Loss function ###

class AANLoss(nn.Module):
    """Cross-entropy loss + an abstraction diversity penalty."""

    def __init__(self, pooling_module_aan):
        super().__init__()
        self.pooling_module_aan = pooling_module_aan
        self.reduction = 'mean'
        self.cross_entropy = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, batch_preds, y_batch):
        # Read the concept attention weights from the last batch
        attn_abs = self.pooling_module_aan.attn_abs

        batch_size = attn_abs.size(0)
        cpt_cross = torch.bmm(attn_abs, attn_abs.transpose(1, 2))
        diag = torch.eye(cpt_cross.size(1), cpt_cross.size(2)).to(device)
        diag = diag.unsqueeze(0).repeat(batch_size, 1, 1)
        cpt_cross = cpt_cross - diag

        # diversity_penalty = torch.sqrt(torch.mean(cpt_cross*cpt_cross))  # From the repo
        diversity_penalty = torch.div(torch.norm(cpt_cross, p='fro'), NUM_CONCEPTS)  # Mine

        # Total loss = x_entropy + beta * diversity_penalty (as in beta-VAE)
        return self.cross_entropy(batch_preds, y_batch) + torch.mul(diversity_penalty, DIVERSITY_PENALTY_BETA)


### build_dataset function definitions ###


def _build_dataset_roberta(self, X, y, tokenizer):
    data = tokenizer.batch_encode_plus(
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
            nn.Dropout(0.01),  # TODO this is prob too high
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM))

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
        self.model = SentimentClassifierModel(
            self.n_classes_, self.encoder_module, self.pooling_module
        ).to(device)
        return self.model

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


class SentimentClassifierAANBase(SentimentClassifierBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = AANLoss(self.pooling_module)

    def build_pooling_module(self):
        return PoolingModuleAAN()

## RoBERTa-Base ##


class SentimentClassifierRoberta(SentimentClassifierBase):
    def __init__(self, *args, **kwargs):
        self.roberta_config = AutoConfig.from_pretrained(
            "roberta-base", output_hidden_states=True, num_labels=3, finetuning_task="sst3")
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", truncation=True, max_length=128, padding="max_length")
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "RoBERTa-Base (Baseline)"

    def build_dataset(self, X, y=None):
        return _build_dataset_roberta(self, X, y, self.roberta_tokenizer)

    def build_encoder_module(self):
        return AutoModel.from_pretrained("roberta-base", config=self.roberta_config)

    def build_pooling_module(self):
        return PoolingModuleTransformerCLS()


class SentimentClassifierRobertaAAN(SentimentClassifierAANBase, SentimentClassifierRoberta):
    def __repr__(self):
        return "RoBERTa-Base (AAN)"


## DynaSent Model 1 ##
class SentimentClassifierDynasent(SentimentClassifierRoberta):
    def __repr__(self):
        return "DynaSent-M1 (Baseline)"

    def build_encoder_module(self):
        return AutoModel.from_pretrained(os.path.join("models", "dynasent_model1.bin"), config=self.roberta_config)


class SentimentClassifierDynasentAAN(SentimentClassifierAANBase, SentimentClassifierDynasent):
    def __repr__(self):
        return "DynaSent-M1 (AAN)"
