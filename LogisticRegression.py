import torch
from dataloader import DataLoader


class LogisticReg(torch.nn.Module):
    def __init__(self, batch_size=False, full_batch=False, n_batch=False):
        super(LogisticReg, self).__init__()
        self.trainloader = DataLoader('candy-data.csv', batch_size=batch_size,
                                      full_batch=full_batch, n_batch=n_batch,
                                      standardize=True)
        self.input_dims = self.trainloader.input_dims
        self.output_dims = self.trainloader.output_dims
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.logistic = self._build_logistic()
        self.optim = torch.optim.SGD(self.logistic.parameters(), lr=0.01)

    def fit_logit_reg(self, epochs=100):
        for epoch in range(epochs):
            loss_val = 0
            for i, (x, y) in enumerate(self.trainloader):
                mini_length = y.size()[0]
                y_pred = self.logistic(x.float())
                y = y.resize_(mini_length, self.output_dims)
                loss = self.loss(y_pred, y.float())
                loss_val = loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # print(f'This will have has had been batch number  {i}')
            print(f'This will have had now been Epoch: {epoch} with Loss: {loss_val} ')
        return self.logistic.parameters()

    def _build_logistic(self):
        logistic = torch.nn.Linear(self.input_dims, self.output_dims, bias=False)
        return logistic

    @staticmethod
    def do_sigmoid(logistic):
        out = torch.sigmoid(logistic)
        return out

    def call_and_write_results(self, epochs=1):
        params = self.fit_logit_reg(epochs)
        results = f' \n \n DataSet: {self.trainloader.filename} \nTorch Logistic Reg with BatchSize: {self.trainloader.batch_size} | Epochs: {epochs} \n' \
                  f'Point Estimates {[par for par in params]} \n \n'
        file = open("Lower_Bound_Results.txt", "a")
        file.write(results)
        file.close()


