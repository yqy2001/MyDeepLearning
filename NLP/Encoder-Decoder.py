"""
@Author:        禹棋赢
@StartTime:     2021/8/21 17:36
@Filename:      Encoder-Decoder.py
"""
import torch
import torch.nn as nn
from d2l import torch as d2l

from NLP.attention import DotProductAttention


class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

    def init_state(self, enc_outputs, *args):
        """extract required things for encoder's output to initialize decoder"""
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)  # get required things to initialize decoder
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)  # [bsz, seq_len] --> [bsz, seq_len, embed_size]
        X = X.permute(1, 0, 2)  # [seq_len, bsz, embed_size], nn.RNN's requirement, first axis is seq_len (time steps)
        # output: [seq_len, bsz, num_directions * hidden_size], the output of the last layer of GRU
        # state: [num_layers * num_directions, bsz, hidden_size], the output of the last time step of each layer
        output, state = self.rnn(X)  # the state defaults to zeros
        return output, state


class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # the input is embedding + encoder's output, here encoder and decoder share same hidden_size
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers=num_layers, dropout=0)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        """

        :param X:
        :param state: the last step hidden state of encoder
            tensor, [num_layers * num_directions, bsz, hidden_size]
        :return:
        """
        X = self.embedding(X).permute(1, 0, 2)  # [seq_len, bsz, embed_size]
        context = state[-1].repeat(X.shape[0], 1, 1)  # use the last layer's hidden state
        X = torch.cat((X, context), 2)  # [seq_len, bsz, embed_size + hidden_size]
        # output: [seq_len, bsz, hidden_size]
        output, state = self.rnn(X, state)
        output = self.linear(output).permute(1, 0, 2)
        return output, state


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface"""
    def __init__(self):
        super(AttentionDecoder, self).__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super(Seq2SeqAttentionDecoder, self).__init__()
        self.attention = DotProductAttention(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs  # hidden_state: [num_layers*num_directions, bsz, hidden_size]
        outputs = outputs.permute(1, 0, 2)  # [seq_len, bsz, hidden_size] --> [bsz, seq_len, hidden_size]
        return outputs, hidden_state, enc_valid_lens

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)  # [bsz, seq_len, embed_size] --> [seq_len, bsz, embed_size]
        outputs, self._attention_weights = [], []

        for x in X:  # x: [bsz, embed_size]
            query = torch.unsqueeze(hidden_state[-1], dim=1)  # [bsz, 1, hidden_size]
            kv = enc_outputs  # [bsz, seq_len, hidden_size]
            context = self.attention(query, kv, kv, enc_valid_lens)  # [bsz, 1, hidden_size]

            input = torch.cat((context, torch.unsqueeze(x, 1)), dim=2).permute(1, 0, 2)  # [1, bsz, hidden+embed]
            # output:[1, bsz, hidden], hidden_state:[num_layers, bsz, hidden]
            output, hidden_state = self.rnn(input, hidden_state)
            outputs.append(output)
            self._attention_weights.append(self.attention.attention_weights)

        outputs = torch.cat(outputs, dim=0)  # [seq_len, bsz, hidden]
        pred = self.linear(outputs).permute(1, 0, 2)  # [bsz, seq_len, vocab_size]
        return pred, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = d2l.MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def train_seq2seqatten():
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens,
                                 num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens,
                                      num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


if __name__ == '__main__':
    # embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    # batch_size, num_steps = 64, 10
    # lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
    #
    # train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    # encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    # decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    # net = EncoderDecoder(encoder, decoder)
    # train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, hidden_size=16,
                                 num_layers=2)
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, hidden_size=16,
                                      num_layers=2)
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)  # (`batch_size`, `num_steps`)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
