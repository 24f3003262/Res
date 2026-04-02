import jax
from jax import jit
import jax.numpy as jnp
import optax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, precision_recall_curve, auc, average_precision_score, roc_curve
)

# --- 1. CONFIGURATION ---
BATCH_SIZE = 512
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
EMBED_DIM = 128
HIDDEN_DIM = 512
NUM_CLASSES = 3

# --- 2. DATA LOAD ---
print("--- Loading Twitter_Data.csv ---")
df = pd.read_csv('Twitter_Data.csv').dropna(subset=['clean_text', 'category'])
label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
df['category'] = df['category'].map(label_map).astype(int)
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_in_chunks(texts, batch_size=5000):
    all_ids = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size].tolist()
        tokens = tokenizer(chunk, truncation=True, padding="max_length", max_length=64, return_tensors="np")["input_ids"]
        all_ids.append(tokens)
    return jnp.concatenate([jnp.array(x) for x in all_ids], axis=0)

train_ids = jax.device_put(tokenize_in_chunks(train_df["clean_text"]))
train_labels = jax.device_put(jnp.array(train_df["category"].values, dtype=jnp.int32))
test_ids = jax.device_put(tokenize_in_chunks(test_df["clean_text"]))
test_labels = jax.device_put(jnp.array(test_df["category"].values, dtype=jnp.int32))

# --- 3. ORIGINAL MATHEMATICAL FUNCTIONS (REVERTED) ---
def init_params_research(vocab_size):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    params = {
        'w': jax.random.normal(k1, (vocab_size,)) * 0.01,
        'emb': jax.random.normal(k2, (vocab_size, EMBED_DIM)) * 0.01,
        'W1': jax.nn.initializers.glorot_normal()(k3, (EMBED_DIM, HIDDEN_DIM)),
        'b1': jnp.zeros(HIDDEN_DIM),
        'W2': jax.nn.initializers.glorot_normal()(k4, (HIDDEN_DIM, 128)),
        'b2': jnp.zeros(128),
        'W3': jax.nn.initializers.glorot_normal()(k5, (128, NUM_CLASSES)),
        'b3': jnp.zeros(NUM_CLASSES)
    }
    return jax.device_put(params)

@jax.jit
def total_loss_fn(params, token_ids, labels, l1_val=1.0, l2_val=0.01, temp=1.0):
    w_sig = jax.nn.sigmoid(params['w'] * temp)
    batch_w = w_sig[token_ids]
    X_prime = jnp.mean(params['emb'][token_ids] * batch_w[:, :, jnp.newaxis], axis=1)

    diff = X_prime[:, jnp.newaxis, :] - X_prime[jnp.newaxis, :, :]
    R = jnp.exp(-jnp.sum(diff**2, axis=-1) / 2.0)
    enemy_mask = labels[:, jnp.newaxis] != labels[jnp.newaxis, :]
    mu = 1.0 - (jax.nn.logsumexp(10.0 * jnp.where(enemy_mask, R, -1e9), axis=1) / 10.0)
    l_rs = 1.0 - jnp.mean(mu)

    h1 = jax.nn.relu(jnp.dot(X_prime, params['W1']) + params['b1'])
    h2 = jax.nn.relu(jnp.dot(h1, params['W2']) + params['b2'])
    logits = jnp.dot(h2, params['W3']) + params['b3']

    l_ce = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, NUM_CLASSES)))
    return l_ce + (l1_val * l_rs) + (l2_val * jnp.mean(w_sig))

@jax.jit
def evaluate_metrics_gpu(params, x_ids, y, temp=1.0):
    w_sig = jax.nn.sigmoid(params['w'] * temp)
    X_prime = jnp.mean(params['emb'][x_ids] * w_sig[x_ids][:, :, jnp.newaxis], axis=1)
    h1 = jax.nn.relu(jnp.dot(X_prime, params['W1']) + params['b1'])
    h2 = jax.nn.relu(jnp.dot(h1, params['W2']) + params['b2'])
    logits = jnp.dot(h2, params['W3']) + params['b3']
    probs = jax.nn.softmax(logits)
    preds = jnp.argmax(logits, axis=1)
    ll = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, NUM_CLASSES)))
    k = jnp.sum(w_sig > 0.5)
    return preds, probs, ll, k

@jax.jit
def evaluate_baseline_direct(params, x_ids, y):
    # TRUE ACCURATE BASELINE: Identity Mask (k = 30,522)
    X_prime = jnp.mean(params['emb'][x_ids], axis=1)
    logits = jnp.dot(jax.nn.relu(jnp.dot(jax.nn.relu(jnp.dot(X_prime, params['W1']) + params['b1']), params['W2']) + params['b2']), params['W3']) + params['b3']
    return jnp.argmax(logits, axis=1), jax.nn.softmax(logits), jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, NUM_CLASSES))), params['emb'].shape[0]

# --- 4. AUDIT HELPERS ---
def calculate_audit(y_true, y_pred, y_probs, k, ll):
    y_t, y_p, y_pb = jax.device_get(y_true), jax.device_get(y_pred), jax.device_get(y_probs)
    cm = confusion_matrix(y_t, y_p)
    recalls = np.diag(cm) / (np.sum(cm, axis=1) + 1e-9)
    g_mean = np.exp(np.mean(np.log(recalls + 1e-9)))
    
    stats = {
        "k": int(k), "Reduct_%": (1 - (int(k) / tokenizer.vocab_size)) * 100,
        "Accuracy": accuracy_score(y_t, y_p), 
        "Precision": precision_score(y_t, y_p, average='weighted', zero_division=0),
        "Recall": recall_score(y_t, y_p, average='weighted', zero_division=0), 
        "F1": f1_score(y_t, y_p, average='weighted'),
        "G-Mean": g_mean, "MCC": matthews_corrcoef(y_t, y_p), "Kappa": cohen_kappa_score(y_t, y_p),
        "AUC": roc_auc_score(np.eye(NUM_CLASSES)[y_t], y_pb, multi_class='ovr', average='weighted'),
        "AUPRC": average_precision_score(np.eye(NUM_CLASSES)[y_t], y_pb, average='weighted'), 
        "AIC": 2 * int(k) + 2 * float(ll) * len(y_t)
    }
    return stats, cm

def plot_curves(y_true, y_probs, model_name):
    y_t_oh = np.eye(NUM_CLASSES)[jax.device_get(y_true)]
    y_pb = jax.device_get(y_probs)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_t_oh[:, i], y_pb[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.title(f'{model_name} ROC'); plt.legend()
    plt.subplot(1, 2, 2)
    for i in range(NUM_CLASSES):
        p, r, _ = precision_recall_curve(y_t_oh[:, i], y_pb[:, i])
        plt.plot(r, p, label=f'Class {i}')
    plt.title(f'{model_name} PR Curve'); plt.legend(); plt.savefig(f"{model_name}_curves.png"); plt.show()

# --- 5. EXECUTION LOOP ---
optimizer = optax.adam(LEARNING_RATE)
params = init_params_research(tokenizer.vocab_size)
opt_state = optimizer.init(params)
best_aic, best_params, history, prev_set = float('inf'), None, [], set()

print(f"--- Starting 25 Iteration Run on {jax.devices()[0]} ---")
for epoch in range(NUM_EPOCHS):
    idx = jax.random.permutation(jax.random.PRNGKey(epoch), len(train_ids))
    epoch_x, epoch_y = train_ids[idx], train_labels[idx]
    for i in range(0, len(epoch_x), BATCH_SIZE):
        b_x, b_y = epoch_x[i:i+BATCH_SIZE], epoch_y[i:i+BATCH_SIZE]
        if len(b_x) < BATCH_SIZE: continue
        # USER SETTING: If you want k ~ 1500, set l2_val to 0.001. If k ~ 600, use 0.01.
        grads = jax.grad(total_loss_fn)(params, b_x, b_y, 1.0, 0.001, 1.0)
        params = optax.apply_updates(params, optimizer.update(grads, opt_state, params)[0])

    p, pr, ll, k = evaluate_metrics_gpu(params, test_ids, test_labels)
    stats, _ = calculate_audit(test_labels, p, pr, k, ll)
    
    curr_set = set(np.where(jax.device_get(jax.nn.sigmoid(params['w'])) > 0.5)[0])
    stability = len(curr_set & prev_set) / len(curr_set | prev_set) if prev_set else 0.0
    prev_set = curr_set

    if stats["AIC"] < best_aic:
        best_aic, best_params = stats["AIC"], jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        print(f"[*] New Best: Epoch {epoch} | AIC: {best_aic:.2f} | k: {int(k)}")

    history.append([epoch, stats["Accuracy"], stats["Precision"], stats["Recall"], stats["F1"], stats["G-Mean"], stats["MCC"], stats["AIC"], int(k), stats["Reduct_%"], stability])
    print(f"Epoch {epoch:02d} | F1: {stats['F1']:.4f} | G-Mean: {stats['G-Mean']:.4f} | k: {int(k)} | Stab: {stability:.4f}")

# --- 6. SEPARATE EXPORTS ---
print("--- Exporting Forensic Reports ---")
p_r, pr_r, ll_r, k_r = evaluate_metrics_gpu(best_params, test_ids, test_labels)
stats_r, cm_r = calculate_audit(test_labels, p_r, pr_r, k_r, ll_r)
pd.DataFrame([stats_r]).to_csv("reduced_final_metrics.csv", index=False)
plot_curves(test_labels, pr_r, "Reduced_DRSAR")

p_b, pr_b, ll_b, k_b = evaluate_baseline_direct(best_params, test_ids, test_labels)
stats_b, cm_b = calculate_audit(test_labels, p_b, pr_b, k_b, ll_b)
pd.DataFrame([stats_b]).to_csv("baseline_final_metrics.csv", index=False)
plot_curves(test_labels, pr_b, "Original_Baseline")

h_cols = ["Epoch", "Acc", "Precision", "Recall", "F1", "GMI", "MCC", "AIC", "k", "Reduct%", "Stability"]
pd.DataFrame(history, columns=h_cols).to_csv("training_history_full.csv", index=False)

print("Done. Three CSVs and all curves generated.")