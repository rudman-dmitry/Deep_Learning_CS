import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator
from tqdm.notebook import tqdm


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    char_list = list({text[idx]: idx for idx in range(len(text))}.keys())
    char_list.sort()
    char_to_idx = {char_list[idx]: idx for idx in range(len(char_list))}
    idx_to_char = {idx: char_list[idx] for idx in range(len(char_list))}                
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = '%s' % text 
    for char in chars_to_remove:
        text_clean = text_clean.replace(char, '') 
    n_removed = len(text) - len(text_clean) 
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    indices = []
    for char in text:
        indices.append(char_to_idx[char])
    result = torch.zeros((len(indices), len(list(char_to_idx.keys()))), dtype=torch.int8)
    result[torch.arange(len(indices)), indices] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    indices = torch.argmax(embedded_text, dim=1).tolist()
    chars =[]
    for idx in indices:
        chars.append(idx_to_char[idx])
    result = ''.join(chars) 
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    text_size = (len(text) - 1) // seq_len
    text_size *= seq_len
    embeded = chars_to_onehot(text[:text_size], char_to_idx)
    samples = embeded.reshape((embeded.size(0)//seq_len), seq_len, embeded.size(1))
    indeces = list(map(lambda x: char_to_idx[x], text[1:text_size + 1]))
    labels = torch.tensor(indeces)
    labels = labels.reshape((int(labels.size(0) / seq_len), seq_len))
    samples = samples.to(device)
    labels = labels.to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    ymax, _ = torch.max(y, dim=dim, keepdim=True)
    y = y - ymax
    if temperature != 0:
        y /= temperature
    result = torch.exp(y) / torch.sum(torch.exp(y), dim=dim, keepdim=True)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        hidden_state = None
        chars_to_gen = n_chars - len(start_sequence)
        for char in tqdm(range(chars_to_gen)):
            if  hidden_state is not None:
                hidden_state = hidden_state.to(dtype=torch.float)
                hidden_state = hidden_state.to(device)
            x = chars_to_onehot(start_sequence, char_to_idx)
            x = x.reshape(1, x.size(0), x.size(1)).to(device)
            layer_output, hidden_state = model.forward(x.to(dtype=torch.float), hidden_state)
            model_out = layer_output[0, layer_output.size(1)-1, :]  
            distr = hot_softmax(model_out[0:], temperature=T)
            best_pred = torch.multinomial(distr, 1).item()
            start_sequence += idx_to_char[best_pred]
        out_text = start_sequence
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        idx = []
        total_len = self.__len__()
        batch_size = self.batch_size
        for i in range(total_len // batch_size):
            for j in range(batch_size):
                idx.append(j * total_len // batch_size + i)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        input_size = in_dim
        for layer in range(self.n_layers):
            w_xz = nn.Linear(input_size, h_dim, bias = False)
            self.add_module("w_xz layer_" + str(layer), w_xz)
            w_hz = nn.Linear(h_dim, h_dim, bias = True)
            self.add_module("w_hz layer_" + str(layer), w_hz)
            w_xr = nn.Linear(input_size, h_dim, bias = False)
            self.add_module("w_xr layer_" + str(layer), w_xr)
            w_hr = nn.Linear(h_dim, h_dim, bias = True)
            self.add_module("w_hr layer_" + str(layer), w_hr)
            w_xg = nn.Linear(input_size, h_dim, bias = False)
            self.add_module("w_xg layer_" + str(layer), w_xg)
            w_hg = nn.Linear(h_dim, h_dim, bias = True)
            self.add_module("w_hg layer_" + str(layer), w_hg)
            if dropout != 0:
                drop_layer = nn.Dropout2d(dropout)
                self.add_module("dropout", drop_layer)
            else:
                drop_layer = 0
            input_size = h_dim
            layers = [w_xz, w_hz, w_xr, w_hr, w_xg, w_hg, drop_layer]
            self.layer_params.append(layers)
        
        w_hy = nn.Linear(h_dim, out_dim, bias = True)
        self. add_module("w_hy", w_hy)
        self.layer_params.append(w_hy)
        
        self.nl_z = torch.sigmoid
        self.nl_r = torch.sigmoid
        self.nl_g = torch.tanh   
            
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        final_hidden_state = []
        for layer in range(self.n_layers):
            w_xz, w_hz, w_xr, w_hr, w_xg, w_hg, drop_layer = self.layer_params[layer]
            h = layer_states[layer]
            layer_memory = []
            for char in range(seq_len):
                x = layer_input[:, char, :]
                z = self.nl_z(w_xz(x)+w_hz(h))
                r = self.nl_r(w_xr(x)+w_hr(h))
                g = self.nl_g(w_xg(x)+w_hg(h*r))
                h = z*h+(1-z)*g
                layer_memory.append(h)
            layer_input = torch.stack(layer_memory, dim = 1)
            if drop_layer != 0:
                layer_input = drop_layer(layer_input)
            final_hidden_state.append(h)
        w_hy = self.layer_params[self.n_layers]
        layer_output = w_hy(layer_input)
        hidden_state = torch.stack(final_hidden_state, dim=1)
        # ========================
        return layer_output, hidden_state
