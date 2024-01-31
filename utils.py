'''
Calculation of the robust loss.
We want to encourage the correlation of each feature and label
i.e., let the model learn gamma robustly useful features
loss = y' * r, where y' = 2*y - 1 ({1,0} -> {1, -1})#so that positive r correlates with 1 and negative r correlates with -1

'''
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def calculate_robustness_img(Y, representation):
    batch_size, c, w, h = representation.size()
    # Expand the dimensions of Y to make it [batch_size, w, h]
    y_matrix = Y.view(batch_size, 1, 1).expand(-1, w, h)
    y_matrix2 = 2 * y_matrix - 1
    y_matrix2 = y_matrix2.to(device)
    representation = representation.to(device)
    cost_matrix = torch.abs(y_matrix2 * representation[:, 0, :, :])
    cost_increase = torch.sum(cost_matrix, dim=[1, 2])
    cost = cost_increase.sum()
    return cost.to(device)
 
def calculate_robustness(Y, representation):
    batch_size, n = representation.size()
    device = representation.device  # Assume device is the same for Y and representation
    Y_expanded = Y.view(batch_size, 1).expand_as(representation)
    y_matrix2 = (Y_expanded * 2 - 1).to(device)
    representation = representation.to(device)
    cost_matrix = torch.abs(y_matrix2 * representation)
    cost = torch.sum(cost_matrix) / (n * batch_size)
    return cost

def main():
    # Assume Y and representation are your input tensors
    Y = torch.tensor([0, 1, 1])
    representation = torch.rand(3, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Call your function
    cost = calculate_robustness(Y, representation)
    print(cost)

if __name__ == "__main__":
    main()
