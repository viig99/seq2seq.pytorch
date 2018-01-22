import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from .seq2seq_base import Seq2Seq
from .modules.recurrent import Recurrent, RecurrentAttention, StackedRecurrent
from .modules.state import State
from .modules.weight_norm import weight_norm as wn
from seq2seq.tools.config import PAD


def bridge_bidirectional_hidden(hidden):
    #  the bidirectional hidden is  (layers*directions) x batch x dim
    #  we need to convert it to layers x batch x (directions*dim)
    num_layers = hidden.size(0) // 2
    batch_size, hidden_size = hidden.size(1), hidden.size(2)
    return hidden.view(num_layers, 2, batch_size, hidden_size) \
        .transpose(1, 2).contiguous() \
        .view(num_layers, batch_size, hidden_size * 2)


class RecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, embedding_size=None, num_layers=1,
                 bias=True, batch_first=False, dropout=0, embedding_dropout=0,
                 forget_bias=None, context_transform=None, bidirectional=True,
                 num_bidirectional=None, mode='LSTM', residual=False, weight_norm=False):
        super(RecurrentEncoder, self).__init__()
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        embedding_size = embedding_size or hidden_size
        num_bidirectional = num_bidirectional or num_layers
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     sparse=False,
                                     padding_idx=PAD)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        if context_transform is not None:  # additional transform on context before output
            self.context_transform = nn.Linear(hidden_size, context_transform)
            if weight_norm:
                self.context_transform = wn(self.context_transform)

        if bidirectional and num_bidirectional > 0:
            assert hidden_size % 2 == 0
            hidden_size = hidden_size // 2
        if num_bidirectional is not None and num_bidirectional < num_layers:
            self.rnn = StackedRecurrent()
            self.rnn.add_module('bidirectional', Recurrent(mode, embedding_size, hidden_size,
                                                           num_layers=num_bidirectional, bias=bias,
                                                           batch_first=batch_first, residual=residual,
                                                           weight_norm=weight_norm, forget_bias=forget_bias,
                                                           dropout=dropout, bidirectional=True))
            self.rnn.add_module('unidirectional', Recurrent(mode, hidden_size * 2, hidden_size * 2,
                                                            num_layers=num_layers - num_bidirectional,
                                                            batch_first=batch_first, residual=residual,
                                                            weight_norm=weight_norm, forget_bias=forget_bias,
                                                            bias=bias, dropout=dropout, bidirectional=False))
        else:
            self.rnn = Recurrent(mode, embedding_size, hidden_size,
                                 num_layers=num_layers, bias=bias,
                                 batch_first=batch_first, residual=residual,
                                 weight_norm=weight_norm, dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, PackedSequence):
            # Lengths data is wrapped inside a Variable.
            bsizes = inputs[1]
            emb = PackedSequence(self.embedding_dropout(
                self.embedder(inputs[0])), bsizes)
            # Get padding mask
            time_dim = 1 if self.batch_first else 0
            bsizes = torch.Tensor(bsizes).type_as(inputs[0].data)
            range_batch = torch.arange(0, bsizes[0]).type_as(inputs[0].data)
            range_batch = range_batch.unsqueeze(time_dim)
            bsizes = bsizes.unsqueeze(1 - time_dim)
            padding_mask = Variable(
                (bsizes - range_batch).le(0), requires_grad=False)
        else:
            padding_mask = inputs.eq(PAD)
            emb = self.embedding_dropout(self.embedder(inputs))
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, PackedSequence):
            outputs = unpack(outputs)[0]
        if hasattr(self, 'context_transform'):
            context = self.context_transform(outputs)
        else:
            context = None

        state = State(outputs=outputs, hidden=hidden_t, context=context,
                      mask=padding_mask, batch_first=self.batch_first)
        return state


class RecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, num_layers=1, bias=True,
                 batch_first=False, forget_bias=None, dropout=0, embedding_dropout=0,
                 mode='LSTM', residual=False, weight_norm=False, tie_embedding=True):
        super(RecurrentDecoder, self).__init__()
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        embedding_size = hidden_size
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     sparse=False,
                                     padding_idx=PAD)
        self.rnn = Recurrent(mode, embedding_size, self.hidden_size,
                             num_layers=num_layers, bias=bias, forget_bias=forget_bias,
                             batch_first=batch_first, residual=residual, weight_norm=weight_norm,
                             dropout=dropout, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

    def forward(self, inputs, state):
        hidden = state.hidden
        if isinstance(inputs, PackedSequence):
            # Lengths data is wrapped inside a Variable.
            emb = PackedSequence(self.embedding_dropout(
                self.embedder(inputs[0])), inputs[1])
        else:
            emb = self.embedding_dropout(self.embedder(inputs))
        x, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, PackedSequence):
            x = unpack(x)[0]

        x = self.classifier(x)
        return x, State(hidden=hidden_t, batch_first=self.batch_first)


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128, embedding_size=None,
                 num_layers=1, bias=True, forget_bias=None, batch_first=False,
                 dropout=0, embedding_dropout=0, tie_embedding=False, residual=False,
                 weight_norm=False, attention=None, concat_attention=True, num_pre_attention_layers=None, mode='LSTM'):
        super(RecurrentAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        attention = attention or {}

        self.layers = num_layers
        self.batch_first = batch_first
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     sparse=False,
                                     padding_idx=PAD)
        self.rnn = RecurrentAttention(embedding_size, context_size, hidden_size, num_layers=num_layers,
                                      bias=bias, batch_first=batch_first, dropout=dropout,
                                      forget_bias=forget_bias, residual=residual, weight_norm=weight_norm,
                                      attention=attention, concat_attention=concat_attention,
                                      num_pre_attention_layers=num_pre_attention_layers, mode=mode)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

        self.hidden_size = hidden_size

    def forward(self, inputs, state, get_attention=False):
        context, hidden = state.context, state.hidden
        if context.context is not None:
            attn_input = (context.context, context.outputs)
        else:
            attn_input = context.outputs
        emb = self.embedding_dropout(self.embedder(inputs))
        if get_attention:
            x, hidden, attentions = self.rnn(emb, attn_input, state.hidden,
                                             mask_attention=context.mask,
                                             get_attention=get_attention)
        else:
            x, hidden = self.rnn(emb, attn_input, state.hidden,
                                 mask_attention=context.mask)
        x = self.classifier(x)

        new_state = State(hidden=hidden, context=context,
                          batch_first=self.batch_first)
        if get_attention:
            new_state.attention_score = attentions
        return x, new_state


class RecurrentAttentionSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256, num_layers=2,
                 embedding_size=None, bias=True, dropout=0, embedding_dropout=0,
                 tie_embedding=False, transfer_hidden=False, forget_bias=None,
                 residual=False, weight_norm=False, encoder=None, decoder=None, batch_first=False):
        super(RecurrentAttentionSeq2Seq, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('bidirectional', True)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('bias', bias)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('embedding_dropout', embedding_dropout)
        encoder.setdefault('residual', residual)
        encoder.setdefault('weight_norm', weight_norm)
        encoder.setdefault('batch_first', batch_first)
        encoder.setdefault('forget_bias', forget_bias)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('bias', bias)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('embedding_dropout', embedding_dropout)
        decoder.setdefault('residual', residual)
        decoder.setdefault('weight_norm', weight_norm)
        decoder.setdefault('batch_first', batch_first)
        decoder.setdefault('forget_bias', forget_bias)
        decoder['context_size'] = encoder['hidden_size']

        self.encoder = RecurrentEncoder(**encoder)
        self.decoder = RecurrentAttentionDecoder(**decoder)
        self.transfer_hidden = transfer_hidden

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def bridge(self, context):
        state = State(context=context, batch_first=self.decoder.batch_first)
        if not self.transfer_hidden:
            state.hidden = None
        else:
            hidden = state.context.hidden
            new_hidden = []
            for h in hidden:
                if self.encoder.bidirectional:
                    new_h = bridge_bidirectional_hidden(h)
                else:
                    new_h = h
                new_hidden.append(new_h)
            state.hidden = new_hidden
        return state


class RecurrentLanguageModel(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256,
                 num_layers=2, bias=True, dropout=0, tie_embedding=False):
        super(RecurrentLanguageModel, self).__init__()
        self.decoder = RecurrentDecoder(vocab_size, hidden_size=hidden_size,
                                        tie_embedding=tie_embedding,
                                        num_layers=num_layers, bias=bias, dropout=dropout)

    def encode(self, *kargs, **kwargs):
        return None
