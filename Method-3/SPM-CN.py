# Phase : Sequential Pattern Mining
from collections import Counter
import jax
import jax.numpy as jnp
def get_top_patterns(sequences,min_support=2,top_k=50,len_of_seq=2):
    patterns=Counter()
    for seq in sequences:
        for i in range(len(seq)-1):
            patterns[tuple(seq[i:i+len_of_seq+1])]+=1

    
    return [list(p) for p,count in patterns.most_common(top_k)]


@jax.jit
def pattern_to_image(sequence,top_patterns,max_len=50,len_of_seq=2):

    num_patterns=len(top_patterns)
    # Initialise empty image (Time axis X Pattern axis)
    matrix=jnp.zeros((max_len,num_patterns))

    # Pattern-to-Image network creation
    def check_pattern(carry,t):

        start_idx=jnp.maximum(0,t-len_of_seq+1)
        current_window=jax.lax.dynamic_slice(sequence,(start_idx,),(len_of_seq,))

        #compare with existing patterns
        # Array with boolean value - output

        matches=jnp.all(top_patterns==current_window,axis=1)

        row_signal=matches.astype(jnp.float32)

        return carry,row_signal
    
    # Range of time steps
    time_indices=jnp.arange(max_len) # Generates an array of elem from 0 to maxlen

    # Run the scan n build image row by row
    _,pattern_matrix=jax.lax.scan(check_pattern,None,time_indices)

    return pattern_matrix
    


class SPM_CN_Pipeline:
    def __init__(self,num_patterns,num_classes):
        self.K=num_patterns
        self.num_classes=num_classes

    def init_params(self,key):
        k1,k2=jax.random.split(key,2)
        return {
            # Filter size 3 x K
            'W_conv': jax.random.normal(k1,(3,self.K,16))*0.1,
            'W_out': jax.random.normal(k2,(16,self.num_classes))*0.1,
            'b_out':jnp.zeros(self.num_classes)
        }
    
    def forward(self,params,pattern_image):
        # 1D Convolution along the time axis
        # pattern_image shape: (Time,Patterns)
        x=jnp.expand_dims(pattern_image,axis=0)

        # Standard Conv implementation
        conv_out=jax.lax.conv_general_dilated(
            lhs=x,
            rhs=jnp.expand_dims(params['W_conv'],axis=0),
            window_strides=(1,),
            padding='SAME'
        )

        # Global Max Pooling and final dense layer
        pooled=jnp.max(conv_out,axis=1)
        logits=jnp.dot(pooled,params['W_out'])+params['b_out']

        return logits
    
