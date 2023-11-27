import sys
import os
import numpy as np

import textwrap
wrapper = textwrap.TextWrapper(width=70)

import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp

def tokenize(input_str, EOS=1):
    """Input str to features dict, ready for inference"""
    inputs =  next(trax.data.tokenize(iter([input_str]),
                                      vocab_dir='vocab_dir/',
                                      vocab_file='summarize32k.subword.subwords'))
    return list(inputs) + [EOS]

def detokenize(integers):
    """List of ints to str"""
    s = trax.data.detokenize(integers,
                             vocab_dir='vocab_dir/',
                             vocab_file='summarize32k.subword.subwords')
    return wrapper.fill(s)

def next_symbol(cur_output_tokens, model):
    """Returns the next symbol for a given sentence.

    Args:
        cur_output_tokens (list): tokenized sentence with EOS and PAD tokens at the end.
        model (trax.layers.combinators.Serial): The transformer model.

    Returns:
        int: tokenized symbol.
    """
    # current output tokens length
    token_length = len(cur_output_tokens)
    padded_length = 2**int(np.ceil(np.log2(token_length + 1)))
    # Fill cur_output_tokens with 0's until it reaches padded_length
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :] 
    # get the outputs from the model
    # model expects padded tensors (with batch)
    output = model(padded_with_batch)
    log_probs = output[0, token_length, :]
    return int(np.argmax(log_probs))

def greedy_decode(input_sentence, model, next_symbol=next_symbol, tokenize=tokenize, detokenize=detokenize):
    """Greedy decode function.

    Args:
        input_sentence (string): a sentence or article.
        model (trax.layers.combinators.Serial): Transformer model.

    Returns:
        string: summary of the input.
    """
    cur_output_tokens = tokenize(input_sentence, ) + [0]    
    generated_output = [] 
    cur_output = 0 
    EOS = 1 
    while cur_output != EOS:
        # Get next symbol
        cur_output = next_symbol(cur_output_tokens, model)
        # Append next symbol to original sentence
        cur_output_tokens.append(cur_output)
        # Append next symbol to generated sentence
        generated_output.append(cur_output)
    return detokenize(generated_output)

def DotProductAttention(query, key, value, mask):
    """Dot product self-attention.
    Args:
        query (jax.interpreters.xla.DeviceArray): array of query representations with shape (L_q by d)
        key (jax.interpreters.xla.DeviceArray): array of key representations with shape (L_k by d)
        value (jax.interpreters.xla.DeviceArray): array of value representations with shape (L_k by d) where L_v = L_k
        mask (jax.interpreters.xla.DeviceArray): attention-mask, gates attention with shape (L_q by L_k)

    Returns:
        jax.interpreters.xla.DeviceArray: Self-attention array for q, k, v arrays. (L_q by d)
    """
    assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"
    # Save depth/dimension of the query embedding for scaling down the dot product
    depth = query.shape[-1]
    # Calculate scaled query key dot product according to formula above
    dots = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(depth)
    # Apply the mask
    if mask is not None: # You do not need to replace the 'None' on this line
        # np.where(condition, value if condition true, value if condition false) 
        # np.full_like(like this array, full of these values)
        dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
    # Softmax formula implementation
    # Use trax.fastmath.logsumexp of masked_qkT to avoid underflow by division by large numbers
    logsumexp = trax.fastmath.logsumexp(dots, axis=-1, keepdims=True)
    # Take exponential of dots minus logsumexp to get softmax
    # Use jnp.exp()
    dots = jnp.exp(dots - logsumexp)
    # Multiply dots by value to get self-attention
    # Use jnp.matmul()
    attention = jnp.matmul(dots, value)
    return attention

def compute_attention_heads_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads
        n_heads (int): number of attention heads
    Returns:
        function: compute_attention_heads function
    """
    def compute_attention_heads(x):
        """ Compute the attention heads.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (n_batch, seqlen, n_heads X d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (n_batch X n_heads, seqlen, d_head).
        """
        # Size of the x's batch dimension
        batch_size = x.shape[0]
        # Length of the sequence
        # Should be size of x's first dimension without counting the batch dim
        seqlen = x.shape[1]
        # Reshape x using jnp.reshape()
        # n_batch, seqlen, n_heads*d_head -> n_batch, seqlen, n_heads, d_head
        x = jnp.reshape(x, (batch_size, seqlen, n_heads, d_head))
        # Transpose x using jnp.transpose()
        # n_batch, seqlen, n_heads, d_head -> n_batch, n_heads, seqlen, d_head
        # Note that the values within the tuple are the indexes of the dimensions of x and you must rearrange them
        x = jnp.transpose(x, (0, 2, 1, 3))
        # Reshape x using jnp.reshape()
        # n_batch, n_heads, seqlen, d_head -> n_batch*n_heads, seqlen, d_head
        x = jnp.reshape(x, (batch_size * n_heads, seqlen, d_head))
        return x
    return compute_attention_heads

def dot_product_self_attention(q, k, v):
    """ Masked dot product self attention.
    Args:
        q (jax.interpreters.xla.DeviceArray): queries.
        k (jax.interpreters.xla.DeviceArray): keys.
        v (jax.interpreters.xla.DeviceArray): values.
    Returns:
        jax.interpreters.xla.DeviceArray: masked dot product self attention tensor.
    """
    mask_size = q.shape[1]
    # Creates a matrix with ones below the diagonal and 0s above. It should have shape (1, mask_size, mask_size)
    # Notice that 1's and 0's get casted to True/False by setting dtype to jnp.bool_
    # Use jnp.tril() - Lower triangle of an array and jnp.ones()
    mask = jnp.tril(jnp.ones((1, mask_size, mask_size),dtype=jnp.bool_), k=0)
    return DotProductAttention(q, k, v, mask)

def compute_attention_output_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads
        n_heads (int): number of attention heads
    Returns:
        function: compute_attention_output function
    """
    def compute_attention_output(x):
        """ Compute the attention output.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (n_batch X n_heads, seqlen, d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (n_batch, seqlen, n_heads X d_head).
        """
        # Length of the sequence
        # Should be size of x's first dimension without counting the batch dim
        seqlen = x.shape[1]
        # Reshape x using jnp.reshape() to shape (n_batch, n_heads, seqlen, d_head)
        # Use '-1' for `n_batch` in this case
        x = jnp.reshape(x, (-1, n_heads, seqlen, d_head))
        # Transpose x using jnp.transpose() to shape (n_batch, seqlen, n_heads, d_head)
        x = jnp.transpose(x, (0, 2, 1, 3))
        return jnp.reshape(x, (-1, seqlen, n_heads * d_head))
    return compute_attention_output

def CausalAttention(d_feature, 
                    n_heads, 
                    compute_attention_heads_closure=compute_attention_heads_closure,
                    dot_product_self_attention=dot_product_self_attention,
                    compute_attention_output_closure=compute_attention_output_closure,
                    mode='train'):
    """Transformer-style multi-headed causal attention.
    Args:
        d_feature (int):  dimensionality of feature embedding.
        n_heads (int): number of attention heads.
        compute_attention_heads_closure (function): Closure around compute_attention heads.
        dot_product_self_attention (function): dot_product_self_attention function. 
        compute_attention_output_closure (function): Closure around compute_attention_output. 
        mode (str): 'train' or 'eval'.

    Returns:
        trax.layers.combinators.Serial: Multi-headed self-attention model.
    """
    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads
    ComputeAttentionHeads = tl.Fn('AttnHeads', compute_attention_heads_closure(n_heads, d_head), n_out=1)
    return tl.Serial(
        tl.Branch( # creates three towers for one input, takes activations and creates queries keys and values
            [tl.Dense(d_feature), ComputeAttentionHeads], # queries
            [tl.Dense(d_feature), ComputeAttentionHeads], # keys
            [tl.Dense(d_feature), ComputeAttentionHeads], # values
        ),
        tl.Fn('DotProductAttn', dot_product_self_attention, n_out=1), # takes QKV
        tl.Fn('AttnOutput', compute_attention_output_closure(n_heads, d_head), n_out=1), # to allow for parallel
        tl.Dense(d_feature)
    )

def DecoderBlock(d_model, d_ff, n_heads,
                 dropout, mode, ff_activation):
    """Returns a list of layers that implements a Transformer decoder block.
    The input is an activation tensor.
    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.
    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """
    # Create masked multi-head attention block using CausalAttention function
    causal_attention = CausalAttention( 
                        d_model,
                        n_heads=n_heads,
                        mode=mode
                        )
    # Create feed-forward block (list) with two dense layers with dropout and input normalized
    feed_forward = [ 
        # Normalize layer inputs
        tl.LayerNorm(),
        # Add first feed forward (dense) layer (don't forget to set the correct value for n_units)
        tl.Dense(n_units=d_ff),
        # Add activation function passed in as a parameter (you need to call it!)
        ff_activation(), # Generally ReLU
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tl.Dropout(rate=dropout, mode=mode),
        # Add second feed forward layer (don't forget to set the correct value for n_units)
        tl.Dense(n_units=d_model),
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tl.Dropout(rate=dropout, mode=mode)
    ]
    # Add list of two Residual blocks: the attention with normalization and dropout and feed-forward blocks
    return [
      tl.Residual(
          # Normalize layer input
          tl.LayerNorm(),
          # Add causal attention block previously defined (without parentheses)
          causal_attention,
          # Add dropout with rate and mode specified
          tl.Dropout(rate=dropout, mode=mode)
        ),
      tl.Residual(
          # Add feed forward block (without parentheses)
          feed_forward
        ),
      ]
    
def TransformerLM(vocab_size=33300,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=4096,
                  mode='train',
                  ff_activation=tl.Relu):
    """Returns a Transformer language model.
    The input to the model is a tensor of tokens. (This model uses only the
    decoder part of the overall Transformer.)
    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_layers (int): number of decoder layers.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train', 'eval' or 'predict', predict mode is for fast inference.
        ff_activation (function): the non-linearity in feed-forward layer.
    Returns:
        trax.layers.combinators.Serial: A Transformer language model as a layer that maps from a tensor of tokens
        to activations over a vocab set.
    """
    # Embedding inputs and positional encoder
    positional_encoder = [ 
        # Add embedding layer of dimension (vocab_size, d_model)
        tl.Embedding(vocab_size, d_model),
        # Use dropout with rate and mode specified
        tl.Dropout(rate=dropout, mode=mode),
        # Add positional encoding layer with maximum input length and mode specified
        tl.PositionalEncoding(max_len=max_len, mode=mode)]
    # Create stack (list) of decoder blocks with n_layers with necessary parameters
    decoder_blocks = [ 
        DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation) for _ in range(n_layers)]
    # Create the complete model as written in the figure
    return tl.Serial(
        # Use teacher forcing (feed output of previous step to current step)
        tl.ShiftRight(mode=mode), # Specify the mode!
        # Add positional encoder
        positional_encoder,
        # Add decoder blocks
        decoder_blocks,
        # Normalize layer
        tl.LayerNorm(),
        # Add dense layer of vocab_size (since need to select a word to translate to)
        # (a.k.a., logits layer. Note: activation already set by ff_activation)
        tl.Dense(n_units=vocab_size),
        # Get probabilities with Logsoftmax
        tl.LogSoftmax()
    )

model = TransformerLM(mode='eval')

# Load the pre-trained weights
model.init_from_file('model.pkl.gz', weights_only=True)

examples = [
    "It was a sunny day when I went to the market to buy some flowers. But I only found roses, not tulips.",
    "It’s the posing craze sweeping the U.S. after being brought to fame by skier Lindsey Vonn, soccer star Omar Cummings, baseball player Albert Pujols - and even Republican politician Rick Perry. But now four students at Riverhead High School on Long Island, New York, have been suspended for dropping to a knee and taking up a prayer pose to mimic Denver Broncos quarterback Tim Tebow. Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were all suspended for one day because the ‘Tebowing’ craze was blocking the hallway and presenting a safety hazard to students. Scroll down for video. Banned: Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll (all pictured left) were all suspended for one day by Riverhead High School on Long Island, New York, for their tribute to Broncos quarterback Tim Tebow. Issue: Four of the pupils were suspended for one day because they allegedly did not heed to warnings that the 'Tebowing' craze at the school was blocking the hallway and presenting a safety hazard to students.",
    "Hamas released a second group of Israeli and foreign hostages on Saturday night in exchange for the release of Palestinian prisoners, the Israeli authorities said Sunday morning, after an hourslong delay raised fears that a fragile truce in Gaza could collapse altogether. Qatar, which helped broker the deal alongside Egypt, said that two mediators had managed to overcome an impasse between Israel and Hamas. In the end, Israel confirmed that Hamas had handed 13 Israelis — eight children and five women — to the International Committee of the Red Cross in Gaza. They were taken in a convoy across the Rafah crossing to Egypt, then transported to Israel, where they were delivered to hospitals, the Israeli authorities said. Four Thai nationals were also released. Within hours, 39 Palestinian prisoners were released by Israel, Israel’s prison service said early on Sunday. There was a similar swap on Friday. The prisoners affairs commission of the Palestinian Authority confirmed that Red Cross buses with detainees had left Ofer prison, outside the West Bank city of Ramallah, to take them to Al-Bireh Municipality. The resumption of the deal late Saturday came after a tense day in which it appeared the fragile temporary cease-fire agreement might crumble. Hamas had threatened to postpone the second hostages-for-prisoners trade, claiming Israel had reneged on parts of the agreement. The armed group, which controls Gaza, said Israel had not allowed enough aid to reach northern Gaza and had not released Palestinian prisoners according to agreed-upon terms.",
    "The stabbing on Friday of Derek Chauvin, the former Minneapolis police officer convicted of murdering George Floyd in 2020, at a special unit inside a Tucson, Ariz., prison is the latest in a series of attacks against high-profile inmates in the troubled, short-staffed federal Bureau of Prisons. The assault comes less than five months after Larry Nassar, the doctor convicted of sexually abusing young female gymnasts, was stabbed multiple times at the federal prison in Florida. It also follows the release of Justice Department reports detailing incompetence and mismanagement at federal detention centers that led to the deaths in recent years of James Bulger, the Boston gangster known as Whitey, and Jeffrey Epstein, who had been charged with sex trafficking. The Federal Bureau of Prisons confirmed that an inmate at the Tucson prison was stabbed around 12:30 p.m. on Friday, though the bureau did not identify Mr. Chauvin, 47, by name. The agency said in a statement that the inmate required “life-saving measures” before being rushed to a hospital emergency room nearby. The office of Keith Ellison, the Minnesota attorney general who prosecuted the former police officer, identified the inmate as Mr. Chauvin. He is likely to survive, according to two people with knowledge of the situation who were not authorized to discuss the incident publicly. On Saturday, the prison remained on lockdown while law enforcement agencies, including the Federal Bureau of Investigation, examined the crime scene and interviewed witnesses. Family visits to the facility have been suspended indefinitely, according to the prison’s website. The facility in Tucson where Mr. Chauvin was stabbed is referred to as a “dropout yard,” one of several special protective units within the Federal Bureau of Prisons system housing informants, people convicted of sex crimes, former gang members and former law enforcement personnel, among others, according to Joe Rojas, who retired earlier this month as president of the union local representing workers at the Federal Correctional Complex near Coleman, Fla.",
]

for sentence in examples: 
    print(wrapper.fill(sentence), '\n')
    print(greedy_decode(sentence, model))
