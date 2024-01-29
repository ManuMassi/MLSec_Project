import warnings
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from autoattack import AutoAttack
import torch

warnings.filterwarnings('ignore')

# Device choice
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Testing accuracy and robustness of the following models from RobustBench
# model_id_numbers = [7, 13, 25, 29, 45]
model_ids = ['Gowal2021Improving_70_16_ddpm_100m', 
             'Xu2023Exploring_WRN-28-10',
             'Sridhar2021Robust_34_15',
             'Zhang2020Geometry',
             'Rebuffi2021Fixing_R18_ddpm']

# Loading the CIFAR10 dataset
x_test, y_test = load_cifar10(n_examples=50)

attacks_to_run=['apgd-ce', 'apgd-dlr', 'fab-t', 'square']

for model_id in model_ids:
    torch.cuda.empty_cache()

    print(f'Loading model {model_id}')
    model = load_model(model_name=model_id, threat_model='Linf')

    # Visualizing the number of parameters of the current model    
    number_of_params = (sum(param.numel() for param in model.parameters()))
    print(f'Number of parameters for model {model_id}: {number_of_params}\n\n')
    
    model.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    n_classes = 10
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=attacks_to_run, device=device)
    adversary.apgd.n_restarts = 1

    # Evaluating the overall accuracy of the model
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    print(f'Evaluated overall accuracy of the model {model_id}\n\n')
    
    # Evaluating accuracy for each class
    for i in range(n_classes):
        torch.cuda.empty_cache()

        class_idxs = (y_test == i)
        x_class = x_test[class_idxs].to(device)
        y_class = y_test[class_idxs].to(device)

        print (f"Robustness of class n.{i} of length: {len(y_class)}")
        
        # Output will show in the console automatically
        x_adv = adversary.run_standard_evaluation(x_class, y_class)
        print("-----/n")
    
    print(f"Model {model_id} evaluted for each class.\n\n")
