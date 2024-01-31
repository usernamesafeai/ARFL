import torch
#We want to encourage the correlation of each feature and label
#i.e., let the model learn gamma robustly useful features
#loss = y' * f, where y' = 2*y - 1 ({1,0} -> {1, -1})#so that positive f correlates with 1 and negative r correlates with -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calculate_robustness(Y, representation):
    batch_size,c,w,h = representation.size()
    cost = torch.tensor(0.0, requires_grad=True).to(device)
    for batchi in range(batch_size):
        y_matrix = torch.full([w,h],Y[batchi])  
        y_matrix2 = torch.mul(y_matrix, 2) - 1
        y_matrix2 = y_matrix2.to(device)
        cost_matrix = torch.mul(y_matrix2, representation[batchi,0,:])  
        cost_matrix = torch.abs(cost_matrix)
        cost_increase = torch.sum(cost_matrix)
        cost += cost_increase
    return cost 
    
