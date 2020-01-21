import torch
from dataloader import DataLoader
from utils import tensordot_pytorch
import matplotlib.pyplot as plt


class EstimateLowerBound:
    def __init__(self, batch_size=False, full_batch=False, n_batch=False):
        self.batch_size = batch_size
        self.full_batch = full_batch
        self.n_batch = n_batch
        self.trainloader = DataLoader('Kid.csv', batch_size=self.batch_size,
                                      full_batch=self.full_batch, n_batch=self.n_batch,
                                      standardize=True)
        self.input_dims = self.trainloader.input_dims
        self.output_dims = self.trainloader.output_dims
        self.prior_var = torch.eye(self.input_dims)
        self.prior_precision = torch.inverse(self.prior_var)
        self.prior_mean = torch.zeros(self.input_dims)

    @staticmethod
    def calculate_epsilon(x, variance, mean):
        XT = x.unsqueeze(1)
        X = x.unsqueeze(2)
        mean = mean.reshape(mean.shape[0], 1)
        eps = torch.sqrt(XT @ variance @ X + (XT @ mean) ** 2)
        eps = eps.reshape(-1)
        return eps

    @staticmethod
    def calculate_lambda(epsilon):
        lambda_vals = torch.tanh(epsilon / 2) / (4 * epsilon)
        lambda_vals = lambda_vals.reshape(-1)
        return lambda_vals

    def calculate_precision(self, x, lambda_vals):
        XT = x.unsqueeze(1)
        X = x.unsqueeze(2)
        vec = torch.matmul(X, XT)
        precision = 2 * tensordot_pytorch(lambda_vals.t(), vec, axes=1) + self.prior_precision
        return precision

    def calculate_mean(self, variance, x, y):
        halves = torch.ones((y.size()[0])) * 0.5
        product = (y - halves).unsqueeze(1)
        sum = (product * x).sum(0)
        mean = (self.prior_precision @ self.prior_mean) + sum
        mean = variance @ mean.t()
        mean = mean.reshape(-1)
        return mean

    def calculate_loss(self, variance, precision, mean, lambdas, epsilons):
        L = (-torch.log(1 + torch.exp(-epsilons)).sum() - (epsilons / 2).sum() + (lambdas * epsilons ** 2).sum()
             - 0.5 * (self.prior_mean.t() @ self.prior_precision @ self.prior_mean)
             + 0.5 * (mean.t() @ precision @ mean)
             + 0.5 * torch.log(torch.div(
                    torch.det(variance), torch.det(self.prior_var)
                )
                )
             )
        return L

    def do_call(self, variance, mean, x, y):
        epsilons = self.__class__.calculate_epsilon(x, variance, mean)
        lambda_vals = self.__class__.calculate_lambda(epsilons)
        precision = self.calculate_precision(x, lambda_vals)
        variance = torch.inverse(precision)
        mean = self.calculate_mean(variance, x, y)

        return epsilons, lambda_vals, precision, variance, mean

    def __call__(self, x, y, threshold=0.0001, max_iters=100):
        variance = self.prior_var
        mean = self.prior_mean
        lossprev = 0
        i = 0
        while threshold is not False:
            epsilons, lambda_vals, precision, variance, mean = self.do_call(variance, mean, x, y)
            loss = self.calculate_loss(variance, precision, mean, lambda_vals, epsilons)
            self.prior_mean = mean
            self.prior_var = variance
            self.prior_precision = precision
            if abs(loss / lossprev - 1) < threshold or abs(lossprev / loss - 1) < threshold:
                threshold = False
            lossprev = loss
            i += 1
            if i > max_iters:
                break

        return mean, variance

    def estimate_lower_bound(self, epochs=1):
        with torch.no_grad():
            batch_mean_variance = torch.zeros((self.trainloader.num_batches, self.input_dims, self.input_dims))
            batch_mean_mean = torch.zeros((self.trainloader.num_batches, self.input_dims))
            for epoch in range(epochs):
                for i, (x, y) in enumerate(self.trainloader):
                    means, variance = self.__call__(x, y)
                    batch_mean_mean[i::] = means
                    batch_mean_variance[i::] = variance
                self.prior_mean = torch.mean(batch_mean_mean, dim=0)
                self.prior_var = torch.mean(batch_mean_variance, dim=0)
                self.prior_precision = torch.inverse(torch.mean(batch_mean_variance, dim=0))
                print(f'Epoch: {epoch} \n JJ Means: {means} \n JJ Variances: {torch.diag(variance)}')

        return means, variance, batch_mean_mean, batch_mean_variance

    def call_and_write_results(self, epochs=1):
        means, variance, _, _ = self.estimate_lower_bound(epochs)
        results = f' \n \n DataSet: {self.trainloader.filename} \n' \
                  f'Lower bound trial with BatchSize: {self.trainloader.batch_size} | Epochs: {epochs} \n' \
                  f'JJ Means: {means} \nJJ Variances: {torch.diag(variance)} \n \n'
        file = open("Lower_Bound_Results.txt", "a")
        file.write(results)
        file.close()

    def fetch_plot_data(self, batch_sizes=None, epochs=1):
        coef_list = []
        for i, batch in enumerate(batch_sizes):
            self.trainloader = DataLoader(self.trainloader.filename, batch_size=batch,
                                          full_batch=self.full_batch, n_batch=self.n_batch,
                                          standardize=True)
            _, _, bmeans, bvars = self.estimate_lower_bound(epochs)
            coef_list.append(bmeans)
        return coef_list

    def divide_plot_data(self, batch_sizes=None, epochs=1):
        data = self.fetch_plot_data(batch_sizes, epochs)
        data_dict = {}
        for j in range(data[0].size()[1]):
            data_dict["mean_" + str(j)] = []

        for i in range(len(data)):
            for j in range(data[i].size()[1]):
                data_dict["mean_" + str(j)].append(data[i][:, j])

        legend = [f" size = {size}" for size in batch_sizes]

        return data_dict, legend

    def generate_plots(self, batch_sizes=None, epochs=1):
        data, legend = self.divide_plot_data(batch_sizes)
        for i in range(len(data)):
            for j in range(len(data["mean_" + str(i)])):
                plt.plot(data["mean_" + str(i)][j])
                if j == len(data['mean_0']) - 1:
                    plt.hlines(data['mean_' + str(i)][-1][0], 0, batch_sizes[-1])
            plt.legend(legend)
            plt.title(f'Mean over batches for Param # {i}')
            plt.savefig(f'Figures/mean_batches_param_{i}_epochs_{epochs}.png')
            plt.close()

