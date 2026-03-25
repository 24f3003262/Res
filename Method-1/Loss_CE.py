import jax
import jax.numpy as jnp

def compute_classification_loss(params,X_weighted,y,num_classes=2):
    # --- Classification Path 

    # Hidden Layer 1: ReLU(W1*X_weighted)
    h1=jax.nn.relu(jnp.dot(X_weighted,params['W1']+params['b1']))
    h1=jax.nn.dropout(h1,rate=0.5,deterministic=False) # Dropout (p=0.5)

    # Hidden layer 2: ReLU(W2*h1+b2)
    h2=jax.nn.relu(jnp.dot(h1,params['W2'])+params['b2'])
    h2=jax.nn.dropout(h2,rate=0.5,deterministic=False)

    #Output layer : y_hat=W3*h2+b3
    logits=jnp.dot(h2,params['W3']+params['b3'])

    # Calculate mean entropy loss

    one_hot_y=jax.nn.one_hot(y,num_classes=num_classes)
    
    log_probs=logits-jax.nn.logsumexp(logits,axis=1,keepdims=True)

    l_ce=-jnp.mean(jnp.sum(one_hot_y*log_probs,axis=-1))

    return l_ce