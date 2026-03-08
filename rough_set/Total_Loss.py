import jax
import jax.numpy as jnp
from jax import jit


## Loss RS
#return R
def compute_fuzzy_similarity(X,weights,sigma=1.0):
    # weights are already sigmoid(w)
    # X shape: (n,m) (50,4)

    # Weighted difference: (X_i - X_j)^2 * w^2
    # We use broadcasting to get (N,M,M) differences
    diff=X[:,jnp.newaxis,:]-X[jnp.newaxis,:,:] 


    weighted_sq_dist=jnp.sum((weights**2)*(diff**2),axis=-1)

    return jnp.exp(-weighted_sq_dist/(2 * sigma**2))



## Original implementation (Stable gradient for future weight learning path but slower and has chances of division by 0 errors. It will ensure calculation is theoretically accurate)
# def soft_lower_approximation(R, labels, alpha=10.0):
#     # 1. Mask for samples NOT in class c (j ∉ Cc)
#     enemy_mask = labels[:, jnp.newaxis] != labels[jnp.newaxis, :]
    
#     # 2. Compute the exponent part: e^(α * Rij)
#     # We subtract a max value for numerical stability (common JAX practice)
#     # but strictly follow the paper's logic.
#     exp_R = jnp.exp(alpha * R)
    
#     # 3. Apply the mask so we only sum over enemies (j ∉ Cc)
#     weighted_R = jnp.where(enemy_mask, R * exp_R, 0.0)
#     sum_exp = jnp.where(enemy_mask, exp_R, 0.0)
    
#     # 4. The Smooth Maximum Approximation fraction from your paper
#     # numerator: Σ (Rij * e^(α * Rij))
#     # denominator: Σ (e^(α * Rij))
#     soft_max_R = jnp.sum(weighted_R, axis=1) / (jnp.sum(sum_exp, axis=1) + 1e-8)
    
#     # 5. Final membership: μ = 1 - smooth_max
#     mu = 1.0 - soft_max_R
    
#     return mu


## Optimised and more aligned with proper jax practices (to be used for high speed for extremely large datasets and to prevent division by 0 error)
def soft_lower_approximation(R,labels,alpha=10.0):
    #labels is the vector of shape (N,) [1,0,1]
    #R is the similaity matrix found from the prev func

    #Mask[i,j] true if i and j have diff labels
    diff_mask=labels[:,jnp.newaxis]!=labels[jnp.newaxis,:]
    diff_similarities=jnp.where(diff_mask,R,-1e9)
    worst_diff_similarity=jax.nn.logsumexp(alpha*diff_similarities,axis=1)/alpha 
    mu=1.0-worst_diff_similarity
    return mu
def calculate_rs_loss(mu):
    gamma=jnp.mean(mu)
    return 1.0-gamma


##Loss CE

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


def total_loss_fn(params,X,y,lambda1=0.1,lambda2=0.05):

    #Feature Weighting
    weights=jax.nn.sigmoid(params['w'])
    X_weighted=X*weights # X ⊙ w

    # Rough Set Path
    R=compute_fuzzy_similarity(X,weights)
    mu=soft_lower_approximation(R,y)
    l_rs=calculate_rs_loss(mu)

    # Classificatio loss
    l_ce=compute_classification_loss(params,X_weighted,y)

    # Sparse Penalty ||w||1
    l_l1=jnp.sum(jnp.abs(weights))

    #Final loss
    return l_ce+lambda1*l_rs+lambda2*l_l1