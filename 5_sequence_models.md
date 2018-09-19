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
  1. The candidate for updating hidden state is as before `¢^{<t>} = tanh(W_c[c^{<t-1>}, x^{<t>}] + b_c)`.
  2. In the vanilla version this value is directly taken, but here it is passed to a gate `Gamma_u`. The gate's definition is identical to the `¢` one, except for using a sigmoid instead of a tanh (and its own `W_u, b_u` parameters): `Gamma_u = sigmoid(W_u[c^{<t-1>}, x^{<t>}] + b_u)`.
  . The interpretation for this is that it acts like a "gate", being almost always close to either zero (if `¢` has low activation) or one (high activation).
  3. The actual update is then: `c^{<t>} = Gamma_u*¢<t> + (1-Gamma_u) * ¢<t-1>`. As we see, a strong current term will open the gate and have more impact on the update, whereas a weaker term will let pass more of the previous, `t-1` state.
  
Note that this applies once per feature, so if c, ¢ and Gamma are vectors, the multiplications in 3. are element-wise.

### Complete version of GRU:

The model described before was simplified. Actually, to compute the candidate, another gate for "relevance" `Gamma_r` is used:
1. `¢^{<t>} = tanh(W_c[   Gamma_r  *    c^{<t-1>}, x^{<t>}] + b_c)`
2. The formula for `Gamma_r` is the exact same as `Gamma_u`, but with its own `W_r, b_r` parameters.
The intuition behind this is that the `Gamma_r` will decide wether to update the candidate or not based on its response, but the response itself will not only depend on the previous `c, x`: `c` itself will be passed through a relevance filter: this still **won't have an impact on the x part of the matmul**, but may nullify the contribution of the c part.

Note that this design is somewhat arbitrary but researchers agree on its efectiveness and together with LSTMs it became a standard. In any case it is possible to customize the design of recurrent units.




# LSTMs:

LSTM units are more complicated, powerful and slow that GRUs. They still are the first go-to model, but recently GRUs have a revival as they also show effectivity and allow to build bigger models.

* The GRU has 2 gates that (apart from their independent weights) are computed identically. LSTMs, is the same, but with 3 gates, called *update, forget, and output*.

* On the other hand, the computation of  `¢` candidate is simplified: `tanh(W_c[a^{<t-1>}, x^{<t>}] + b_c)` the candidate is computed as in the vanilla RNN.

* The complexity comes at the top:
  1. The update and forget gates are no longer mutually exclusive: `c^{<t>} = Gamma_u*¢<t> + Gamma_f * c<t-1>`: any [0,1] linear combination of update_current_candidate and forget_prior_c is allowed.
  2. The new hidden state is regulated by the output gate: `a^{<t>} = Gamma_o*c<t>`

So the current candidate depends on the prior a, the current c depends on the current candidate and prior c, and the current a (the actual output) depends on the current c.


### Peephole connection:

* A way of adding more capacity to the LSTM is to allow the gate to look into the actual previous c (remember that the LSTM gates depend on `[a,x]` and `a!=c`). I.e: `Gamma_u = sigmoid(W_u[a^{<t-1>}, x^{t}, c^{<t-1>}] + b_u)`.

  
  
# Bidirectional RNN:

The idea is, for each hidden state `a^{<t>}`, add `ā^{<t>}`, whose forwardprop goes 'backwards', i.e. it starts at `t=T` and ends at `t=1`. Effectively this means having two independent models of the same flipped inputs. They are combined at `y^{<t>} = g(W_y[a^{<t>}, ā^{<t>}] + b_y)`: note that this means that the parameters `W_y` have twice the size.

* BRNN+LSTM appears in many NLP setups and is a very reasonable first thing to try.
* Note that this vanilla formulation requires to have the whole sequence to predict or be trained. This can be a problem in real-time applications, although there are more complex models to overcome that.


With this is easy to see that an early value for `c^{<0>}` can pass all the wat to the end, if the forget and output gate are constantly high and the update low.




# Deep RNNs:

* Sometimes is useful to stack many layers of recurrent units, but it is computationally very expensive compared to standard NNs: 3 layers is already considered deep.
* Usually hybrid models (f.e. substitute the last `W_y` with a full blown NN).

* Notation: deep RNN means multiple, stacked hidden `a^{<t>}` states. Therefore, for each layer `l`, they are referred here as `a^{[l]<t>}`, and the corresponding parameters are `W_a^{[l]}, b_a^{[l]}`.

* To compute the new state of a, we still have the vanilla `a^{<t>} = W_a*[a^{<t-1>}, x^{<t>}]` for the lowest layer. But for the upper layers, `x` gets replaced by the hidden, lowerer layer, i.e.: `a^{[l]<t>} = W_a*[a^{[l]<t-1>}, a^{[l-1]<t>}]`.



