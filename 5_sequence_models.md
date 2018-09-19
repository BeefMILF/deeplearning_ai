# Why sequence models:

* many tasks (DL makes easier to bridge from very short term to bigger structures)
* note that either input or output can be a sequence, or both





# Notation:

* x: input sequence. Positions are specified with superset angle brackets, so `x^{<t>}` is the word at position `t`, starting by 1.
* y: output sequence. For entity recognition, could be a binary flag. superset index same as with x.
* length of sequences: `T_x` and `T_y` respectively. Since t begins with 1, T_x is the biggest t in x
* if x is a collection of inputs: `X`, where `X^{(i)}` is the `i`th input. Therefore, `X^{(i)(t)}` is the t word of input i. Also, respective lengths would be `T_x^{(i)}`

### Representing words

* by indexes in a dictionary. A dict of 10k words is small, 100k is not uncommon and large internet companies 1million or more.

* one-hot vectors of size(dict) dimensions.
* one special `<unk>` token for unknown words.





# Recurrent NN Model:

* Why not standard NN:
  1. variable length
  2. doesn't share features across different positions

*assuming T_x=T_y*.
**PROBLEM: recurrent info is only based in the past. Info of the future is useful (tackled by Bidirectional RNNs)**
* As usual, for an input `x^{<t>}` the network outputs a prediction `y^{<t>}`. But the network also stores the generated  hidden state `a^{<t>}`. So in `t+1`, both `x^{<t+1>}` and `a^{<t>}` are used to predict `y^{<t+1>}`. This means that the network stores its states in an extra cell with delay of 1 step. Some people initialize `a^{<0>}`randomly, but most common practice is to initialize it with zeros.

* Therefore, there are 3 groups of parameters that are **shared across all iterations of `t`**: 
  1. `W_{ax}` regulate the transition from `x^{<t>}` to `a^{<t>}`
  2. `W_{aa}`, the "horizontal parameters", regulate the "recurrent" transition from `a^{<t-1>}` to `a^{<t>}`
  3. `W_{ya}`, regulate the transition from `a^{<t>}` to `y^{<t>}`
  

### Forward propagation:
  1. `a^{<t>} = g(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a)`, where `g` is activation fn, (tanh, ReLU). tanh is common, ways to prevent gradient vanishing explained later
  2. `y^{<t>} = g(W_{ya}a^{<t>} + b_y)`, the activation here depends on problem domain (sigmoid/softmax for classification is usual)
  
### Simplified RNN notation:

* Alias: `W_{ya}` becomes `W_{y}` because it always receives some vector from hidden state `a`.
* `W_a` is the result of stacking the `W_ax, W_aa` matrices, so `a^{<t>} = g(W_a[a^{<t-1>}, x^{<t>}] + b_a)`. The dimensionality will be `dim(a)` times `dim(a)+dim(dict)`





# BPTT:

* Assuming a loss function `L(ŷ, y) = sum_t L^{<t>}(ŷ^{<t>}, y^{<t>})` note that the overall loss for the whole sequence `y` is the sum of the losses for the respective elements in the sequence. The forwardprop can be reformulated in a way that explicitly shows how the subsequential `a` hidden states are built. Note that this isn't exactly the formulation of the loss:

`W_y*a_1 + W_y*a_2 + ... + W_y*a_T` = `W_y*(W_a[a_0, x_1]) + W_y*(W_a[a_1, x_2]) + ... W_y*(w_a[a_{T-1}, x_T])` = `W_y*(W_a[a_0, x_1]) + W_y*(W_a[W_a[a_0, x_1], x_2]) + ... W_y*(W_a[W_a[W_a[...]], x_T])`


Note that once the contribution for `W_a` at a given stage `t` is known, it can be used to compute the contribution at further stages. And therefore, the gradients for `W_a` can be sequentially computed. More on this later on.





# Different types of RNNs:

* many-to-one: sentiment mining (sentence -> star rating)
* one-to-many: music generation (criteria -> melody)
* many-to-many fixed: entity recognition (sentence -> marked sentence)
* many-to-many variable: machine translation (sentence->encoder->(latent space)->decoder->sentence)





# Language model and sequence generation:

* Speech recognition: probability of a sentence (among all the possible valid sentences) given an audio? we need a language model to constraint the space of valid sentences, or more generally to "estimate that probability".

* Pick a sentence, tokenize it adding the `<unk>` and (optionally) `<eos>` tokens, and set `x^{<1>} = 0, x^{<t+1>} = y^{<t>}` for the whole sequence (basically the input sentence equals the zero vector followed by the ground truth).

* For a softmax output, the prob. of a sentence is the multiplication of the single outputs, since in the sequence `abc`, `P(a,b,c) = P(a)*P(b|a)*P(c|a,b) = y_1 * y_2 * y_3`.





# Sampling novel sequences:

* Picking an existing (pre-trained) network, feed x^0=0 to collect the ŷ^1 softmax distribution. Then sample from it (pick the max or sample from the whole distribution) and feed the chosen result as x2, etc...

* It is also possible to do character modelling, but it is much slower to train and less sensitive to long-term dependencies. It is used in specialized contexts or when working with many unknown words...




# Vanishing gradients with RNNs:

* gradients grow/decrease exponentially with the no. of layers. So in RNNs, passing a sequence with 1000 elements is basically equivalent to having a 1000-layer NN.
  1. If gradients are exploding it is easy to spot, usually we have `NaN` weights. Gradient clipping is a robust solution
  2. Gradient vanishing is the bigger problem: in long sequences, it may be difficult for the RNN to capture long-term dependencies, because the information in our optimization procedure doesn't reach that far away. Solutions in next video.
  
  
  
  
  # Gated Recurrent Unit (GRU):
  
* In our vanilla RNN, the horizontal activation is stateless, that is, a "pure function" like tanh. Gated RNNs add an extra "state" to it: for each hidden state `a^{<t>}` there is a corresponding `c^{<t>}` state of the state's gate. Simplified explanation:
  1. The candidate for updating hidden state is as before `¢^{<t>} = tanh(W_c[c^{<t-1>, x^{<t>}] + b_c)`.
  2. In the vanilla version this value is directly taken, but here it is passed to a gate `Gamma_u`. The gate's definition is identical to the `¢` one, except for using a sigmoid instead of a tanh. The interpretation for this is that it acts like a "gate", being almost always close to either zero (if `¢` has low activation) or one (high activation).
  3. The actual update is then: `c^{<t>} = Gamma_u*¢<t> + (1-Gamma_u) * ¢<t-1>`. As we see, a strong current term will open the gate and have more impact on the update, whereas a weaker term will let pass more of the previous, `t-1` state.
  
  
