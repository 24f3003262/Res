import jax
import jax.numpy as jnp
from jax import jit
def init_fl_rnn_params(key,input_dimension,hidden_dimension,num_rules):
    k1,k2,k3,k4,k5=jax.random.split(key,5)
    return {
        # Linear projection params : z_t=Wx*xt + Wh * ht-1 + b
        'Wx': jax.random.normal(k1,(input_dimension,hidden_dimension))*0.1,
        'Wh': jax.random.normal(k2,(hidden_dimension,hidden_dimension))*0.1,
        'b':jnp.zeroes(hidden_dimension),
        
        # Fuzzification Layer: K Gaussian MFs for each neuron j
        'c': jax.random.normal(k3,(num_rules,hidden_dimension)),
        'sigma':jnp.ones((hidden_dimension,num_rules)),

        # Rule Application: TSK consequents
        'q':jax.random.normal(k4,(num_rules,hidden_dimension)),
        
        # RNN tanh update weight
        'Wc':jax.random.normal(k5,(input_dimension,hidden_dimension))*0.1
    }

def fl_rnn_cell(params,h_prev,x_t):
    # 1. Linear Projection: z_t = Wx*xt+Wh*ht-1+b
    z_t=jnp.dot(x_t,params['Wx'])+jnp.dot(h_prev,params['Wh'])+params['b']

    # 2. Fuzzification layer : Gaussian Membership Functions (MFs)
    # Shape of z_t is (hidden_dim,), we expand to (hidden_dim,num_rules)
    z_expanded = z_t[:,jnp.newaxis]
    mu=jnp.exp(-((z_expanded-params['c'])**2)/(2*params['sigma']**2))

    # 3. Rule Application (TSK Inference): Weighted average
    # g_tj = Σ (mu * q) / Σ (mu)
    numerator = jnp.sum(mu * params['q'], axis=1)
    denominator = jnp.sum(mu, axis=1) + 1e-8
    g_t = numerator / denominator # The Fuzzy Gate output

    # 4. RNN Update with Fuzzy Gates
    # h_t = (1 - g_t) ⊙ h_prev + g_t ⊙ tanh(Wc * x_t)
    candidate = jnp.tanh(jnp.dot(x_t, params['Wc']))
    h_t = (1 - g_t) * h_prev + g_t * candidate

    return h_t,h_t

@jax.jit
def fl_rnn_predict(params,x_sequence):
    # x_seq shape: (seq_len,input_dimension)
    hidden_dimension=params['Wx'].shape[0]
    h_init=jnp.zeros(hidden_dimension)

    # Define the scan function to carry the hidden state across time
    def scan_fn(h_prev, x_t):
        h_next, _ = fl_rnn_cell(params, h_prev, x_t)
        return h_next, h_next
    
    # Process the entire sequence
    final_h = jax.lax.scan(scan_fn, h_init, x_sequence)
    return final_h # The final hidden state representing the sequence