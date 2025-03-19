import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiments
import json


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


from nnunetv2.run.run_training import run_training
import os


def build_network_architecture(plans_manager: PlansManager,
                               dataset_json,
                               configuration_manager: ConfigurationManager,
                               num_input_channels,
                               enable_deep_supervision: bool = True) -> nn.Module:
    return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                  num_input_channels, deep_supervision=enable_deep_supervision)


if __name__ == "__main__":
    dataset_id = 511
    task_name = "roi"
    checkpoint_name = 'checkpoint_final.pth'
    configuration = '3d_fullres'
    fold = '0'
    patch_size = [96, 128, 192]

    dataset_name = convert_id_to_dataset_name(dataset_id)
    model_training_output_dir = os.path.join(nnUNet_results, dataset_name)
    checkpoint_file = os.path.join(nnUNet_results, dataset_name, f'fold_{fold}', checkpoint_name)
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    dataset_json_file = os.path.join(model_training_output_dir, 'dataset.json')
    dataset_fingerprint_file = os.path.join(model_training_output_dir, 'dataset_fingerprint.json')
    plan_file = os.path.join(model_training_output_dir, 'plans.json')

    dataset_json = load_json(dataset_json_file)
    dataset_fingerprint = load_json(dataset_fingerprint_file)
    plan = load_json(plan_file)

    f = int(fold) if fold != 'all' else fold

    plans_manager = PlansManager(plan)

    configuration_manager = plans_manager.get_configuration(configuration)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager,
                                                      dataset_json)
    network = build_network_architecture(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels
    )

    net = network
    net.load_state_dict(checkpoint["network_weights"])
    net.eval()

    dummy_input = torch.randn(1, 1, *patch_size)  # .to("cuda")

    torch.onnx.export(
        net,
        dummy_input,
        os.path.join(nnUNet_results, dataset_name, f'fold_{fold}', f'{task_name}.onnx'),
        input_names=['input'],
        output_names=['output']
        # dynamic_axes = {'input': {0: 'batch_size'},'output': {0: 'batch_size'}}
    )



