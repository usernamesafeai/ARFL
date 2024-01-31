import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from utils import calculate_robustness
epsilon = 0.05
epsilon2 = 0.2
print(f"defense epsilon = {epsilon}")
print(f"attack epsilon = {epsilon2}")
seeds =  range(45,50,1)
num_repeat = len(seeds) 
print(seeds)
gamma = 0.5
acc_sum_std = 0
accs_std = []
acc_sum_adv = 0
accs_adv = []
for i in range(num_repeat):
    SEED = seeds[i] 
    #gamma = 0.1#0.5# 1# 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate synthetic "moons" dataset
    X, y = make_moons(n_samples=10000, noise=0.2, random_state=1, shuffle=False)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    # Define the neural network
    class MoonClassifier(nn.Module):
        def __init__(self):
            super(MoonClassifier, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 1)
            self.act = nn.ReLU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            features = self.act(self.fc2(x))
            features_out = torch.sigmoid(features)
            x = self.fc3(features)
            out = torch.sigmoid(x)
            return out, features_out 

    # Initialize and train the model
    model = MoonClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5000):
        optimizer.zero_grad()
        outputs, _ = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Create the perturbed dataset
    X_perturbed = X.clone().detach().requires_grad_(True)
    outputs, _ = model(X_perturbed)
    loss = criterion(outputs, y)
    model.zero_grad()
    loss.backward()
    X_perturbed_grad = X_perturbed.grad.data
    X_perturbed = X_perturbed + epsilon * X_perturbed_grad.sign()
    y_perturbed = y.clone()

    # Combine original and perturbed datasets for training
    X_combined = torch.cat([X, X_perturbed], dim=0)
    y_combined = torch.cat([y, y_perturbed], dim=0)

    print(f'seed = {SEED}')
    # Initialize and train the model with adversarial dataset
    model_adv = MoonClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_adv.parameters(), lr=0.01)

    for epoch in range(5000):
        optimizer.zero_grad()
        outputs, features = model_adv(X_combined)
        reg = calculate_robustness(y_combined, features)
        ce = criterion(outputs, y_combined)
        '''
        if epoch % 100 == 0:
            print(f'epoch = {epoch} reg = {reg} ce = {ce}')
            print('---------')
        '''
        loss = ce - gamma * reg
        loss.backward()
        optimizer.step()

    # Generate adversarial perturbations for a single data point
    # Number of data points to display from each moon
    num_points_per_moon =  50

    # Lists to store original and perturbed points
    original_points = []
    perturbed_points = []

    # Collect data points from both moons
    points_collected_0 = 0
    points_collected_1 = 0
    index = 0

    X, y = make_moons(n_samples=100, noise=0.2, random_state=3, shuffle=False)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    #acc
    original_labels = []
    original_preds = []
    perturbed_preds = []
    while (points_collected_0 < num_points_per_moon or points_collected_1 < num_points_per_moon) and index < len(y):
        label = y[index].item()
        if (label == 0 and points_collected_0 < num_points_per_moon) or (label == 1 and points_collected_1 < num_points_per_moon):
            data_point = X[index].clone().detach().requires_grad_(True)
            output, features = model(data_point.unsqueeze(0))
            #acc
            output_std, _ = model_adv(data_point.unsqueeze(0))
            original_labels.append(label)
            original_preds.append(output_std.round().item())

            loss = criterion(output, y[index].unsqueeze(0))
            model.zero_grad()
            loss.backward()
            data_point_grad = data_point.grad.data
            perturbed_data = data_point + epsilon2 * data_point_grad.sign()
            
            #acc
            output_adv, _ = model_adv(perturbed_data.unsqueeze(0))
            perturbed_preds.append(output_adv.round().item())

            original_points.append(tuple(data_point.detach().numpy()))
            perturbed_points.append(tuple(perturbed_data.detach().numpy()))

            
            if label == 0:
                points_collected_0 += 1
            else:
                points_collected_1 += 1

        index += 1

    #acc
    standard_accuracy = sum([1 for i, j in zip(original_labels, original_preds) if i == j]) / len(original_labels)
    adversarial_accuracy = sum([1 for i, j in zip(original_labels, perturbed_preds) if i == j]) / len(original_labels)
    print(f"Standard Accuracy: {standard_accuracy*100:.2f}%")
    print(f"Adversarial Accuracy: {adversarial_accuracy*100:.2f}%")
    acc_sum_std += standard_accuracy
    accs_std.append(standard_accuracy)
    acc_sum_adv += adversarial_accuracy
    accs_adv.append(adversarial_accuracy)


    # Plotting the 100 test cases 
    x_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    y_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    outputs,_ = model_adv(inputs)
    Z = outputs.detach().numpy().reshape(xx.shape)


    #Plot for standard data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="Spectral", alpha=0.6)

    original_plot = True#  
    if original_plot:
        ax.scatter([], [], color='blue', marker='o', s=100, label='Standard Data (Label 0)')
        ax.scatter([], [], color='red', marker='o', s=100, label='Standard Data (Label 1)')
    else:
        ax.scatter([], [], color='blue', marker='^', s=100, label='Adversarial Data (Label 0)')
        ax.scatter([], [], color='red', marker='^', s=100, label='Adversarial Data (Label 1)')

    marker_opacity = 1  # Adjust this value for desired transparency

    for op, pp in zip(original_points, perturbed_points):
        if y[original_points.index(op), 0].item() == 0:
            if original_plot:
                ax.scatter(op[0], op[1], color='blue', marker='o', s=100, alpha=marker_opacity)
            else:
                ax.scatter(pp[0], pp[1], color='blue', marker='^', s=100, alpha=marker_opacity)
        else:
            if original_plot:
                ax.scatter(op[0], op[1], color='red', marker='o', s=100, alpha=marker_opacity)
            else:
                ax.scatter(pp[0], pp[1], color='red', marker='^', s=100, alpha=marker_opacity)
    ax.legend()
    # Change the font size of the legend
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontsize(20)  # you can set a numeric value or string like 'x-small', 'medium', etc.
    #hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    if original_plot:
        fig.savefig(f"ARFL_balanced_model_moon_originalgamma" + str(gamma) + "-" + str(i) + ".jpg", format='jpeg', dpi=300)
    else:
        fig.savefig("ARFL_balanced_model_moon_perturbedgamma" + str(gamma) + "-" + str(i) + ".jpg", format='jpeg', dpi=300)
    plt.close(fig)

    #Plot for adversarial data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="Spectral", alpha=0.6)

    original_plot = False#  
    if original_plot:
        ax.scatter([], [], color='blue', marker='o', s=100, label='Standard Data (Label 0)')
        ax.scatter([], [], color='red', marker='o', s=100, label='Standard Data (Label 1)')
    else:
        ax.scatter([], [], color='blue', marker='^', s=100, label='Adversarial Data (Label 0)')
        ax.scatter([], [], color='red', marker='^', s=100, label='Adversarial Data (Label 1)')

    marker_opacity = 1  # Adjust this value for desired transparency

    for op, pp in zip(original_points, perturbed_points):
        if y[original_points.index(op), 0].item() == 0:
            if original_plot:
                ax.scatter(op[0], op[1], color='blue', marker='o', s=100, alpha=marker_opacity)
            else:
                ax.scatter(pp[0], pp[1], color='blue', marker='^', s=100, alpha=marker_opacity)
        else:
            if original_plot:
                ax.scatter(op[0], op[1], color='red', marker='o', s=100, alpha=marker_opacity)
            else:
                ax.scatter(pp[0], pp[1], color='red', marker='^', s=100, alpha=marker_opacity)
    ax.legend()
    # Change the font size of the legend
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontsize(20)  # you can set a numeric value or string like 'x-small', 'medium', etc.
    #hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    if original_plot:
        fig.savefig(f"ARFL_balanced_model_moon_originalgamma" + str(gamma) + "-" + str(i) + ".jpg", format='jpeg', dpi=300)
    else:
        fig.savefig("ARFL_balanced_model_moon_perturbedgamma" + str(gamma) + "-" + str(i) + ".jpg", format='jpeg', dpi=300)
    plt.close(fig)



print(f"gamma = {gamma} Average Standard Accuracy: {acc_sum_std*100/num_repeat:.2f}%")
print(f"gamma = {gamma} Standard Deviation: {np.std(accs_std)*100:.2f}%")
print(f"gamma = {gamma} Average Adversarial Accuracy: {acc_sum_adv*100/num_repeat:.2f}%")
print(f"gamma = {gamma} Standard Deviation: {np.std(accs_adv)*100:.2f}%")
