import torch
from dataloader import DataLoader


class EstimateLowerBound:
    def __init__(self, batch_size=False, full_batch=False, n_batch=False):
        self.trainloader = DataLoader('Kid.csv', batch_size=batch_size,
                                      full_batch=full_batch, n_batch=n_batch,
                                      standardize=True)
        self.input_dims = self.trainloader.input_dims
        self.output_dims = self.trainloader.output_dims
        self.prior_var = torch.eye(self.input_dims)
        self.prior_precision = torch.inverse(self.prior_var)
        self.prior_mean = torch.zeros(self.input_dims)

    @staticmethod
    def calculate_epsilon(x, variance, mean, mini_size):
        eps = []
        for i in range(mini_size):
            eps.append(
                torch.sqrt((x[i].t() @ variance @ x[i]) + (x[i].t() @ mean) ** 2)
            )
        breakpoint()
        return eps

    @staticmethod
    def calculate_lambda(epsilon, mini_size):
        lambda_vals = []
        for i in range(mini_size):
            eps = epsilon[i]

            lambda_vals.append(
                (torch.tanh(eps / 2)) / (4 * eps)
            )
        return lambda_vals

    def calculate_precision(self, x, lambda_vals, mini_size):
        sum = torch.zeros((self.input_dims, self.input_dims))
        for i in range(mini_size):
            lam = lambda_vals[i]
            sum += lam * torch.ger(x[i], x[i].t())
        sum *= 2
        precision = self.prior_precision + sum

        return precision

    def calculate_mean(self, variance, x, y, mini_size):
        mean = (self.prior_precision @ self.prior_mean)
        mean_sum = torch.zeros(self.input_dims)
        for i in range(mini_size):
            mean_sum += ((y[i] - 0.5) * x[i])
        mean = variance @ (mean + mean_sum)

        return mean

    def calculate_loss(self, variance, precision, mean, lambda_vals, epsilons):
        L = 0
        eps = torch.tensor(epsilons)
        lam = torch.tensor(lambda_vals)
        L = (-torch.log(1 + torch.exp(-eps)).sum() - (eps / 2).sum() + (lam * eps ** 2).sum()
             - 0.5 * (self.prior_mean.t() @ self.prior_precision @ self.prior_mean)
             + 0.5 * (mean.t() @ precision @ mean)
             + 0.5 * torch.log(torch.div(
                    torch.det(variance), torch.det(self.prior_var)
                )
                )
             )

        return L

    def do_call(self, variance, mean, x, y, mini_size):
        epsilons = self.__class__.calculate_epsilon(x, variance, mean, mini_size)
        lambda_vals = self.__class__.calculate_lambda(epsilons, mini_size)
        precision = self.calculate_precision(x, lambda_vals, mini_size)
        variance = torch.inverse(precision)
        mean = self.calculate_mean(variance, x, y, mini_size)

        return epsilons, lambda_vals, precision, variance, mean

    def __call__(self, x, y, mini_size, threshold=0.0001, max_iters=100):
        variance = self.prior_var
        mean = self.prior_mean
        lossprev = 0
        i = 0
        while threshold is not False:
            epsilons, lambda_vals, precision, variance, mean = self.do_call(variance, mean, x, y, mini_size)
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
        batch_mean_variance = torch.zeros((self.trainloader.num_batches, self.input_dims, self.input_dims))
        batch_mean_mean = torch.zeros((self.trainloader.num_batches, self.input_dims))
        for epoch in range(epochs):
            for i, (x, y) in enumerate(self.trainloader):
                mini_length = y.size()[0]
                means, variance = self.__call__(x, y, mini_length)
                batch_mean_mean[i::] = means
                batch_mean_variance[i::] = variance
                if self.trainloader.total_data_len % (i + 1) == 2:
                    print(f'Halfway through the dataset')
            self.prior_mean = self.__class__.torch_n_minus_one_mean(batch_mean_mean)
            self.prior_var = self.__class__.torch_n_minus_one_mean(batch_mean_variance)
            self.prior_precision = torch.inverse(self.__class__.torch_n_minus_one_mean(batch_mean_variance))
            print(f'Epoch: {epoch} \n JJ Means: {means} \n JJ Variances: {torch.diag(variance)}')

        return means, variance

    @staticmethod
    def torch_n_minus_one_mean(a):
        length = a.size()[0]
        a = torch.div(torch.sum(a, dim=0), length)
        return a

    def call_and_write_results(self, epochs=1):
        means, variance = self.estimate_lower_bound(epochs)
        results = f' \n \nLower bound trial with BatchSize: {self.trainloader.batch_size} | Epochs: {epochs} \n' \
                  f'JJ Means: {means} \n JJ Variances: {torch.diag(variance)} \n \n'
        file = open("Lower_Bound_Results.txt", "a")
        file.write(results)
        file.close()
