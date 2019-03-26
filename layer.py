class Linear():
    # y = wx + b, x is (#samples)*in_dim, y is (#samples)*out_dim
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        bound = (6.0/(in_dim + out_dim))**0.5
        self.w = np.random.uniform(-bound,bound,size=(out_dim, in_dim))
        # self.w = np.zeros((out_dim, in_dim))
        self.b = np.zeros(out_dim)
        self.vw = np.zeros(self.w.shape)
        self.vb = np.zeros(self.b.shape)
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        y = x.dot(self.w.T) + self.b.reshape(1,-1)
        return y

    def backward(self, dout):
        self.db = dout.sum(axis=0) / len(self.x)
        self.dw = dout.T.dot(self.x) / len(self.x)
        dx = dout.dot(self.w)
        return dx
class Sigmoid():
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.dy = y * (1 - y)
        return y
    def backward(self, delta):
        return self.dy * delta
class CrossEntropy():
    def __init__(self):
        self.loss = None
        self.eps = 1e-18

    def __call__(self, x, y):
        return self.forward(x,y)

    def onehot(self,y):
        return np.eye(self.num_class,dtype=int)[y]
        
    def __softmax(self, x):
        ex = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return ex / np.sum( ex, axis = 1, keepdims = True)

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        self.sm = self.__softmax(self.logits) + self.eps
        self.loss = np.sum( - np.log(self.sm) * self.labels) / len(x)
        return self.loss        
    def backward(self):
        return self.sm - self.labels