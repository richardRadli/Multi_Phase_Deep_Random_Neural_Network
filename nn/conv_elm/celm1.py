import torch


def generate_normalized_data(X):
    return X / torch.norm(X, keepdim=True)


def compute_hidden_matrix(X, W, b):
    batch_size, channels, height, width = X.size()
    X_flat = X.view(batch_size, -1)
    return torch.nn.ReLU()(torch.matmul(X_flat, W) + b)


def compute_output_weights(H, T):
    return torch.pinverse(H).matmul(T)


def compute_filters_and_bias(beta):
    F_mat_T = beta[:, :-1].T
    B_T = beta[:, -1]

    return F_mat_T, B_T


def reshape_filter(F_mat):
    return F_mat.permute(1, 0).unsqueeze(-1).unsqueeze(-1)


def generate_conv_params(X, num_filters):
    # Generate normalized training data X_{n}
    X_N = generate_normalized_data(X)

    # Compute desired target T = [X_{N} | 1]
    batch_size, channels, height, width = X_N.size()
    T = torch.cat([X_N.view(batch_size, -1), torch.ones(batch_size, 1)], dim=1)

    # Generate randomly the input weights and biases W and b
    W = torch.randn(channels * height * width, num_filters)
    b = torch.zeros(1, num_filters)

    # Compute the hidden matrix H = act_fun(XW + b)
    H = compute_hidden_matrix(X_N, W, b)

    # Compute the output weights beta=H_{pseudo_inverse} T
    beta = compute_output_weights(H, T)

    # Compute filters and bias
    F_mat_T, B_T = compute_filters_and_bias(beta)

    # Reshape the filters matrix
    F = reshape_filter(F_mat_T)

    return F, B_T


def main():
    torch.manual_seed(42)
    input_data = torch.randn(60000, 1, 28, 28)
    num_filters = 1000
    F, B = generate_conv_params(input_data, num_filters)


if __name__ == '__main__':
    main()
