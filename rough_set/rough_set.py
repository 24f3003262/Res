import jax
import jax.numpy as jnp
from jax import jit

def compute_fuzzy_similarity(X,weights,sigma=1.0):
    # weights are already sigmoid(w)
    # X shape: (n,m)

    # Weighted difference: (X_i - X_j)^2 * w^2
    # We use broadcasting to get (N,M,M) differences
    diff=X[:,jnp.newaxis,:]-X[jnp.newaxis,:,:]

    weighted_sq_dist=jnp.sum((weights**2)*(diff**2),axis=-1)

    return jnp.exp(-weighted_sq_dist/(2 * sigma**2))

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
def total_loss_fn(params,X,y,lambda1=0.1,lambda2=0.07):
    #converting raw weights into sigmoid so that it lies between 0 and 1
    weights=jax.nn.sigmoid(params['w'])

    #1. Fuzzy reln
    R=compute_fuzzy_similarity(X,weights)
    
    #2. Soft layer approx
    mu=soft_lower_approximation(R,y)

    #3. RS loss
    l_rs=calculate_rs_loss(mu)

    #4. L1 regularisation to improve accuracy (Feature selection penalty)
    l_l1=jnp.sum(jnp.abs(weights))

    #Total loss= classifier loss (cross entropy)+lambda1*RS_Loss+lambda2
    return lambda1*l_rs+lambda2*l_l1

key=jax.random.PRNGKey(39)
N,M=300,4
X=jax.random.normal(key,(N,M))
y=(X[:,0]>0).astype(jnp.int32)
initial_raw_value = jnp.log(0.6 / (1 - 0.6))
#Weight gen
params = {
    'w': jnp.full((M,), initial_raw_value)
}

@jit
def update(params,x,y):
    grads=jax.grad(total_loss_fn)(params,X,y)
    #Simple Gradient Descent
    return {k : v-0.1*grads[k] for k,v in params.items()}


#Train
print("----------------------TRAINING START--------------------")

for i in range(500):
    params=update(params,X,y)
    if i%20==0:
        current_weights=jax.nn.sigmoid(params['w'])
        loss=total_loss_fn(params,X,y)
        print(f"Iter {i:3} | Loss {loss:.4f} | Weights: {current_weights}")

final_weights = jax.nn.sigmoid(params['w'])
for i, w in enumerate(final_weights):
    status = "SELECTED" if w > 0.5 else "DISCARDED"
    print(f"Feature {i}: Weight {w:.4f} -> {status}")
