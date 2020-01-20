import torch
from sklearn.linear_model import LogisticRegression
from dataloader import DataLoader


class LogisticRegSK:
    def __init__(self, batch_size=False, full_batch=False, n_batch=False):
        super(LogisticReg, self).__init__()
        self.trainloader = DataLoader('Kid.csv', batch_size=batch_size,
                                      full_batch=full_batch, n_batch=n_batch,
                                      standardize=True)
        self.input_dims = self.trainloader.input_dims
        self.output_dims = self.trainloader.output_dims
        self.logistic = LogisticRegression(fit_intercept=False)

    def fit_logit_reg(self, epochs=100):
        print(f" With batch size: {self.trainloader.batch_size}")
        for epoch in range(epochs):
            loss_val = 0
            for i, (x, y) in enumerate(self.trainloader):
                if y.sum() ==0:
                    print(f'all ones')
                else:
                    self.logistic.fit(x, y)
                # print(f'This will have has had been batch number  {i}')
            # print(f'This will have had now been Epoch: {epoch} with Loss: {loss_val} ')
        return self.logistic.coef_
