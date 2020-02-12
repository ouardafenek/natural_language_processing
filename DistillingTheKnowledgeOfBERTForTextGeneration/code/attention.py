import torch
import torch.nn as nn
import torch.nn.functional as F


"""
**Attention Decoder**

If only the context vector is passed betweeen the encoder and decoder, 
that single vector carries the burden of encoding the entire sentence.

Attention allows the decoder network to “focus” on a different part of the encoder’s 
outputs for every step of the decoder’s own outputs. 
First we calculate a set of attention weights. 
These will be multiplied by the encoder output vectors to create a weighted combination. 
The result should contain information about that specific part of the input sequence, 
and thus help the decoder choose the right output words.

Calculating the attention weights is done with another feed-forward layer attn, 
using the decoder’s input and hidden state as inputs. 
Because there are sentences of all sizes in the training data, 
to actually create and train this layer we have to choose a maximum sentence length 
(input length, for encoder outputs) that it can apply to. 
Sentences of the maximum length will use all the attention weights, 
while shorter sentences will only use the first few.

"""

# This is the version proposed by IBM for Ses2Seq tasks 

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn