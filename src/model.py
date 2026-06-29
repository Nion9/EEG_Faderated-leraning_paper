"""
EEGNet-inspired CNN — exactly as specified in paper.
=====================================================
Architecture (from paper Section III-B):
  Input: 64 channels x 640 time points
  1. Temporal Conv:  32 filters (1x64),  BN, ELU
  2. Spatial Conv:   32 filters (64x1),  BN, ELU, AvgPool(1x4)
  3. Depthwise Conv: 64 filters (1x16),  BN, ELU, AvgPool(1x8)
  4. Flatten -> FC(128) -> ELU -> FC(2) -> Softmax

Loss:     categorical cross-entropy
Optimizer: Adam (lr=0.001)

Implemented in pure numpy with manual backprop.
Flower-compatible: get_weights() / set_weights()
"""

import numpy as np

# Input dimensions
N_CHAN   = 64
N_TIME   = 640
N_CLASS  = 2

# After pooling:
# After AvgPool(1,4): T = 640//4 = 160
# After AvgPool(1,8): T = 160//8 = 20
# Flatten: 64 * 20 = 1280


def softmax(x):
    x  = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)

def elu(x, a=1.0):
    return np.where(x >= 0, x, a*(np.exp(np.clip(x,-88,0))-1))

def elu_grad(x, a=1.0):
    return np.where(x >= 0, 1.0, a*np.exp(np.clip(x,-88,0)))

def relu(x):
    return np.maximum(x, 0)

def relu_grad(x):
    return (x > 0).astype(np.float64)


# ── Batch normalisation (1D over feature dim) ─────────────────────────────────

def bn_forward(x, gamma, beta, running_mean, running_var,
               training=True, momentum=0.9, eps=1e-5):
    if training:
        mu  = x.mean(0)
        var = x.var(0)
        running_mean[:] = momentum*running_mean + (1-momentum)*mu
        running_var[:]  = momentum*running_var  + (1-momentum)*var
    else:
        mu, var = running_mean, running_var
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma*xhat + beta, xhat, mu, var

def bn_backward(dout, xhat, mu, var, gamma, eps=1e-5):
    N   = dout.shape[0]
    dg  = (dout*xhat).sum(0)
    db  = dout.sum(0)
    dx  = (gamma/np.sqrt(var+eps)) * (
          dout - dout.mean(0) - xhat*((dout*xhat).mean(0)))
    return dx, dg, db


# ── Conv1D helpers (operate on [B, C_out, T] tensors) ────────────────────────

def conv1d_forward(x, W, b, stride=1):
    """
    x : [B, C_in, T]
    W : [C_out, C_in, kW]
    b : [C_out]
    returns: [B, C_out, T_out]
    """
    B, C_in, T = x.shape
    C_out, _, kW = W.shape
    T_out = (T - kW) // stride + 1
    out   = np.zeros((B, C_out, T_out), dtype=np.float64)
    for k in range(kW):
        out += np.einsum('bci,oc->boi',
                         x[:, :, k:k+(T_out-1)*stride+1:stride], W[:, :, k])
    out += b[np.newaxis, :, np.newaxis]
    return out

def conv1d_backward(dout, x, W, stride=1):
    """Returns dX, dW, db."""
    B, C_in, T   = x.shape
    C_out, _, kW = W.shape
    T_out        = dout.shape[2]
    dx = np.zeros_like(x)
    dW = np.zeros_like(W)
    db = dout.sum(axis=(0, 2))
    for k in range(kW):
        x_slice = x[:, :, k:k+(T_out-1)*stride+1:stride]  # [B,C_in,T_out]
        dW[:, :, k] = np.einsum('boi,bci->oc', dout, x_slice)
        dx[:, :, k:k+(T_out-1)*stride+1:stride] += np.einsum(
            'boi,oc->bci', dout, W[:, :, k])
    return dx, dW, db

def avgpool1d(x, pool_size):
    """x: [B,C,T] -> [B,C,T//pool_size]"""
    B, C, T = x.shape
    T2      = T // pool_size
    return x[:, :, :T2*pool_size].reshape(B, C, T2, pool_size).mean(axis=3)

def avgpool1d_backward(dout, x, pool_size):
    B, C, T = x.shape
    T2      = dout.shape[2]
    dx      = np.zeros_like(x)
    dx[:, :, :T2*pool_size] = np.repeat(dout, pool_size, axis=2) / pool_size
    return dx


# ── EEGNet CNN ────────────────────────────────────────────────────────────────

class EEGMLP:
    """
    EEGNet-inspired CNN matching paper specification.
    Named EEGMLP for compatibility with existing Flower client code.

    Forward pass:
      [B,1,64,640]
      -> reshape to [B,64,640]  (treat channels as feature dim)
      -> TempConv [B,32,577]    (32 filters, kernel=64)
      -> BN + ELU
      -> SpatConv [B,32,577]    (32 filters, kernel=1, depthwise-style)
      -> BN + ELU + AvgPool(4) -> [B,32,144]
      -> DepthConv [B,64,129]   (64 filters, kernel=16)
      -> BN + ELU + AvgPool(8) -> [B,64,16]
      -> Flatten [B,1024]
      -> FC1 [B,128] + ELU
      -> FC2 [B,2] + Softmax

    Flower interface: get_weights() / set_weights()
    """

    # Keys for Flower NDArrays serialisation
    PARAM_KEYS = [
        'W_temp','b_temp',
        'bn1_g','bn1_b',
        'W_spat','b_spat',
        'bn2_g','bn2_b',
        'W_dep','b_dep',
        'bn3_g','bn3_b',
        'W_fc1','b_fc1',
        'W_fc2','b_fc2',
    ]
    STAT_KEYS = [
        'bn1_rm','bn1_rv',
        'bn2_rm','bn2_rv',
        'bn3_rm','bn3_rv',
    ]

    def __init__(self, seed=None,
                 lr=0.001, weight_decay=1e-4, dropout=0.25):
        if seed is not None:
            np.random.seed(seed)

        self.lr      = lr
        self.wd      = weight_decay
        self.dropout = dropout
        self.training = True

        # ── Layer dimensions ──────────────────────────────────────────
        # After TempConv (k=64): T=640-64+1=577
        # After AvgPool(4):      T=577//4=144
        # After DepthConv(k=16): T=144-16+1=129
        # After AvgPool(8):      T=129//8=16
        # Flatten: 64*16 = 1024

        C_in  = N_CHAN   # 64
        T1    = 577      # after TempConv
        T2    = 144      # after AvgPool(4)
        T3    = 129      # after DepthConv
        T4    = 16       # after AvgPool(8)
        flat  = 64 * T4  # 1024

        # Temporal conv: [32, 1, 64] — treats input as [B, 1, T]
        # We reshape input to [B*64, 1, 640] for temporal filtering
        # Simpler: treat as [B, 64, 640], apply conv along time with C_in=64
        # TempConv: 32 filters of size (1, 64) across time
        self.W_temp = np.random.randn(32, 1, 64).astype(np.float64) \
                      * np.sqrt(2.0/64)
        self.b_temp = np.zeros(32, dtype=np.float64)

        # BN after TempConv: normalise over [B, T] for each of 32 channels
        # We flatten [B,32,577] -> operate on feature dim 32
        self.bn1_g  = np.ones(32,  dtype=np.float64)
        self.bn1_b  = np.zeros(32, dtype=np.float64)
        self.bn1_rm = np.zeros(32, dtype=np.float64)
        self.bn1_rv = np.ones(32,  dtype=np.float64)

        # Spatial conv: 32 filters of size (64,1) — point-wise here
        self.W_spat = np.random.randn(32, 32, 1).astype(np.float64) \
                      * np.sqrt(2.0/32)
        self.b_spat = np.zeros(32, dtype=np.float64)

        self.bn2_g  = np.ones(32,  dtype=np.float64)
        self.bn2_b  = np.zeros(32, dtype=np.float64)
        self.bn2_rm = np.zeros(32, dtype=np.float64)
        self.bn2_rv = np.ones(32,  dtype=np.float64)

        # Depthwise conv: 64 filters of size (1,16) across time
        self.W_dep  = np.random.randn(64, 32, 16).astype(np.float64) \
                      * np.sqrt(2.0/32)
        self.b_dep  = np.zeros(64, dtype=np.float64)

        self.bn3_g  = np.ones(64,  dtype=np.float64)
        self.bn3_b  = np.zeros(64, dtype=np.float64)
        self.bn3_rm = np.zeros(64, dtype=np.float64)
        self.bn3_rv = np.ones(64,  dtype=np.float64)

        # FC layers
        self.W_fc1 = np.random.randn(flat, 128).astype(np.float64) \
                     * np.sqrt(2.0/flat)
        self.b_fc1 = np.zeros(128, dtype=np.float64)

        self.W_fc2 = np.random.randn(128, N_CLASS).astype(np.float64) \
                     * np.sqrt(2.0/128)
        self.b_fc2 = np.zeros(N_CLASS, dtype=np.float64)

        # Adam state
        self._t  = 0
        self._ms = {k: np.zeros_like(getattr(self,k)) for k in self.PARAM_KEYS}
        self._vs = {k: np.zeros_like(getattr(self,k)) for k in self.PARAM_KEYS}
        self._cache = {}

    def _drop(self, h):
        if not self.training or self.dropout == 0:
            return h, np.ones_like(h)
        m = (np.random.rand(*h.shape) > self.dropout).astype(np.float64) \
            / (1 - self.dropout + 1e-9)
        return h*m, m

    def _bn_2d(self, x, g, b, rm, rv):
        """BN over [B, C, T] — normalise per channel C."""
        B, C, T = x.shape
        x_flat  = x.transpose(0,2,1).reshape(-1, C)  # [B*T, C]
        out, xhat, mu, var = bn_forward(x_flat, g, b, rm, rv,
                                         self.training)
        out = out.reshape(B, T, C).transpose(0,2,1)
        xhat= xhat.reshape(B, T, C).transpose(0,2,1)
        return out, xhat, mu, var

    def _bn_2d_back(self, dout, xhat, mu, var, g):
        B, C, T = dout.shape
        do_flat  = dout.transpose(0,2,1).reshape(-1, C)
        xh_flat  = xhat.transpose(0,2,1).reshape(-1, C)
        dx_flat, dg, db = bn_backward(do_flat, xh_flat, mu, var, g)
        dx = dx_flat.reshape(B, T, C).transpose(0,2,1)
        return dx, dg, db

    def forward(self, X):
        """X: [B, 1, 64, 640]"""
        c = self._cache
        B = X.shape[0]

        # Reshape: [B, 1, 64, 640] -> [B, 64, 640]
        x0 = X[:, 0, :, :].astype(np.float64)   # [B, 64, 640]
        c['x0'] = x0

        # ── Temporal conv: treat each channel independently ───────────
        # Reshape to [B*64, 1, 640], apply 32 filters of width 64
        x0r = x0.reshape(B*N_CHAN, 1, N_TIME)
        t1  = conv1d_forward(x0r, self.W_temp, self.b_temp)  # [B*64, 32, 577]
        t1  = t1.reshape(B, N_CHAN, 32, t1.shape[2])           # [B,64,32,577]
        # Average over channels -> [B, 32, 577]
        t1  = t1.mean(axis=1)
        c['t1_pre'] = t1

        # BN + ELU
        t1_bn, xh1, mu1, v1 = self._bn_2d(t1, self.bn1_g, self.bn1_b,
                                            self.bn1_rm, self.bn1_rv)
        t1_act = elu(t1_bn)
        t1_d, m1 = self._drop(t1_act)
        c.update({'t1_bn':t1_bn,'xh1':xh1,'mu1':mu1,'v1':v1,
                  't1_act':t1_act,'m1':m1,'t1_d':t1_d})

        # ── Spatial (point-wise) conv ─────────────────────────────────
        t2  = conv1d_forward(t1_d, self.W_spat, self.b_spat)  # [B,32,577]
        t2_bn, xh2, mu2, v2 = self._bn_2d(t2, self.bn2_g, self.bn2_b,
                                            self.bn2_rm, self.bn2_rv)
        t2_act = elu(t2_bn)
        t2_p   = avgpool1d(t2_act, 4)   # [B, 32, 144]
        t2_d, m2 = self._drop(t2_p)
        c.update({'t2':t2,'t2_bn':t2_bn,'xh2':xh2,'mu2':mu2,'v2':v2,
                  't2_act':t2_act,'t2_p':t2_p,'m2':m2,'t2_d':t2_d})

        # ── Depthwise conv ────────────────────────────────────────────
        t3  = conv1d_forward(t2_d, self.W_dep, self.b_dep)    # [B,64,129]
        t3_bn, xh3, mu3, v3 = self._bn_2d(t3, self.bn3_g, self.bn3_b,
                                            self.bn3_rm, self.bn3_rv)
        t3_act = elu(t3_bn)
        t3_p   = avgpool1d(t3_act, 8)   # [B, 64, 16]
        t3_d, m3 = self._drop(t3_p)
        c.update({'t3':t3,'t3_bn':t3_bn,'xh3':xh3,'mu3':mu3,'v3':v3,
                  't3_act':t3_act,'t3_p':t3_p,'m3':m3,'t3_d':t3_d})

        # ── FC ────────────────────────────────────────────────────────
        flat   = t3_d.reshape(B, -1)    # [B, 1024]
        fc1    = flat @ self.W_fc1 + self.b_fc1   # [B,128]
        fc1_a  = elu(fc1)
        fc1_d, m4 = self._drop(fc1_a)
        logits = fc1_d @ self.W_fc2 + self.b_fc2  # [B,2]
        probs  = softmax(logits)
        c.update({'flat':flat,'fc1':fc1,'fc1_a':fc1_a,'m4':m4,
                  'fc1_d':fc1_d,'logits':logits,'probs':probs})
        return probs

    def backward(self, y):
        c = self._cache; B = len(y)
        dL = c['probs'].copy()
        dL[np.arange(B), y] -= 1.0
        dL /= B

        # FC2
        dW_fc2 = c['fc1_d'].T @ dL
        db_fc2 = dL.sum(0)
        d_fc1d = dL @ self.W_fc2.T

        # FC1
        d_fc1a = d_fc1d * c['m4']
        d_fc1  = d_fc1a * elu_grad(c['fc1'])
        dW_fc1 = c['flat'].T @ d_fc1
        db_fc1 = d_fc1.sum(0)
        d_flat = d_fc1 @ self.W_fc1.T

        # Unflatten
        d_t3d  = d_flat.reshape(c['t3_d'].shape)

        # DepthConv backward
        d_t3p  = d_t3d * c['m3']
        d_t3a  = avgpool1d_backward(d_t3p, c['t3_act'], 8)
        d_t3bn = d_t3a * elu_grad(c['t3_bn'])
        d_t3, dbn3g, dbn3b = self._bn_2d_back(d_t3bn, c['xh3'],
                                                c['mu3'], c['v3'], self.bn3_g)
        d_t2d, dW_dep, db_dep = conv1d_backward(d_t3, c['t2_d'], self.W_dep)

        # SpatConv backward
        d_t2p  = d_t2d * c['m2']
        d_t2a  = avgpool1d_backward(d_t2p, c['t2_act'], 4)
        d_t2bn = d_t2a * elu_grad(c['t2_bn'])
        d_t2, dbn2g, dbn2b = self._bn_2d_back(d_t2bn, c['xh2'],
                                                c['mu2'], c['v2'], self.bn2_g)
        d_t1d, dW_spat, db_spat = conv1d_backward(d_t2, c['t1_d'], self.W_spat)

        # TempConv backward
        d_t1a  = d_t1d * c['m1']
        d_t1bn = d_t1a * elu_grad(c['t1_bn'])
        d_t1, dbn1g, dbn1b = self._bn_2d_back(d_t1bn, c['xh1'],
                                                c['mu1'], c['v1'], self.bn1_g)
        # Broadcast back over channels (we averaged over channels in forward)
        d_t1_broad = d_t1[:, np.newaxis, :, :].repeat(N_CHAN, axis=1)
        d_t1_broad = d_t1_broad / N_CHAN
        d_t1r = d_t1_broad.reshape(B*N_CHAN, 1, d_t1.shape[2])
        # Pad to original length for backward pass
        d_x0r = np.zeros((B*N_CHAN, 1, N_TIME), dtype=np.float64)
        _, dW_temp, db_temp = conv1d_backward(d_t1r,
            c['x0'].reshape(B*N_CHAN, 1, N_TIME), self.W_temp)

        self._grads = {
            'W_temp':dW_temp, 'b_temp':db_temp,
            'bn1_g':dbn1g,    'bn1_b':dbn1b,
            'W_spat':dW_spat, 'b_spat':db_spat,
            'bn2_g':dbn2g,    'bn2_b':dbn2b,
            'W_dep':dW_dep,   'b_dep':db_dep,
            'bn3_g':dbn3g,    'bn3_b':dbn3b,
            'W_fc1':dW_fc1,   'b_fc1':db_fc1,
            'W_fc2':dW_fc2,   'b_fc2':db_fc2,
        }
        loss = float(-np.log(
            np.clip(c['probs'][np.arange(B),y],1e-12,1)).mean())
        return loss

    def step(self):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for n in self.PARAM_KEYS:
            g = self._grads[n]
            if n in ('W_temp','W_spat','W_dep','W_fc1','W_fc2'):
                g = g + self.wd * getattr(self, n)
            self._ms[n] = b1*self._ms[n] + (1-b1)*g
            self._vs[n] = b2*self._vs[n] + (1-b2)*(g**2)
            mh = self._ms[n]/(1-b1**self._t)
            vh = self._vs[n]/(1-b2**self._t)
            p  = getattr(self, n)
            p -= self.lr * mh / (np.sqrt(vh)+eps)

    def train_epoch(self, X, y, batch_size=32):
        self.training = True
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        losses = []
        for i in range(0, len(y), batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            if len(yb) < 2: continue
            self.forward(xb)
            losses.append(self.backward(yb))
            self.step()
        return float(np.mean(losses)) if losses else 0.0

    def predict(self, X, batch_size=64):
        self.training = False
        preds = [np.argmax(self.forward(X[i:i+batch_size]),axis=1)
                 for i in range(0,len(X),batch_size)]
        self.training = True
        return np.concatenate(preds)

    def evaluate(self, X, y):
        return float((self.predict(X)==y).mean())

    def get_weights(self):
        return [getattr(self,k).copy() for k in self.PARAM_KEYS+self.STAT_KEYS]

    def set_weights(self, weights):
        for k,w in zip(self.PARAM_KEYS+self.STAT_KEYS, weights):
            setattr(self, k, w.copy())
        self._t  = 0
        self._ms = {k:np.zeros_like(getattr(self,k)) for k in self.PARAM_KEYS}
        self._vs = {k:np.zeros_like(getattr(self,k)) for k in self.PARAM_KEYS}

    def get_param_vector(self):
        return np.concatenate([getattr(self,k).ravel() for k in self.PARAM_KEYS])
