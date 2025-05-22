

import os
import torch
import pandas as pd
import json
from openood.evaluation_api import Evaluator
from openood.networks import wrappermodel
import pickle



eda_models={
 'eda_ResNet50_cifar100_supcon':'resnet50_supcon_cifar100',
 'eda_ResNet34_supcon_cifar10':'resnet34_supcon_cifar10',
 'eda_ResNet34_cifar100_supcon':'resnet34_supcon_cifar100',
 'eda_ResNet50_supcon_cifar10':'resnet50_supcon_cifar10',
 'eda_resnet18_cifar10':'resnet18_cifar10',
 'eda_resnet34_cifar10':'resnet34_cifar10',
 'eda_resnet18_cifar100':'resnet18_cifar100',
 'eda_resnet34_cifar100':'resnet34_cifar100',
}


for model_name, timm_model_name in eda_models.items():
                import timm
                import detectors
                id_dset = 'cifar100' if 'cifar100' in model_name else 'cifar10'
                from timm.data.config import resolve_data_config
                from timm.data.transforms_factory import create_transform
                model = timm.create_model(timm_model_name, pretrained=True)
                cfg = resolve_data_config({}, model = model, use_test_size = True)
                preprocess = create_transform(**cfg, is_training = False)

                net = wrappermodel(num_classes=100 if id_dset=='cifar100' else 10)
                net.basemodel=model

                net.cuda()
                net.eval()
                net.save_path = os.path.join('./outputs/',id_dset,model_name)

                # List of postprocessors (OOD detection methods) to evaluate
                postprocessor_names = [
                    "mds",
                    "mdsn",
                    "rmdsn",
                    "rmds",  
                    #  "knn",
                    # "vim", 
                    # "dice",
                    # "react",
                    # "openmax", 
                    # "msp", 
                    # "temp_scaling", 
                    # "ebo",  "mls", 
                    # "klm", 
                    # 'neco',
                    # 'gmmn',
                    # 'nnguide',
                    # # 'gmm_',
                    # "odin",
                    #                       "rankfeat",
                    #  "ash",
                    #   "she",
                ]

                print('Using model', model_name)
                print('Preprocessor: ',preprocess)

                # Dictionary to store the evaluation results
                results_dict = {}
                # Loop through each postprocessor and evaluate
                for postprocessor_name in postprocessor_names:
                    print(f"Evaluating with postprocessor: {postprocessor_name}")
                    try:
                    # if True:
                        # Initialize evaluator with the current postprocessor
                        evaluator = Evaluator(
                            net,
                            id_name=id_dset,                     # the target ID dataset
                            data_root='./data',                    # change if necessary
                            config_root='./configs',               # see notes above
                            preprocessor=preprocess,                     # default preprocessing for the target ID dataset
                            postprocessor_name=postprocessor_name, # the postprocessor to use
                            postprocessor=None,                    # if you want to use your own postprocessor
                            batch_size=512,                        # batch size for certain methods
                            shuffle=False,
                            num_workers=2                          # could use more num_workers outside colab
                        )
                        
                        # Evaluate OOD detection
                        metrics = evaluator.eval_ood(fsood=False)

                        # Store the results in the dictionary
                        results_dict[postprocessor_name] = {
                            "metrics": evaluator.metrics,  # This is a dictionary, not a DataFrame
                            "scores": evaluator.scores    # This is a dictionary containing predictions and confidences
                        }

                        # Extract the 'ood' part of the metrics dictionary (which is a DataFrame)
                        ood_metrics = evaluator.metrics.get('ood', None)

                        # Create the "results" directory if it doesn't exist
                        results_dirs = ["results"]
                        results_dirs = [os.path.join(r,id_dset) for r in results_dirs]
                        for results_dir in results_dirs:
                            if not os.path.exists(results_dir):
                                os.makedirs(results_dir)

                            # Create a subfolder for the model inside the "results" directory
                            model_results_dir = os.path.join(results_dir, model_name)
                            if not os.path.exists(model_results_dir):
                                os.makedirs(model_results_dir)

                            # Create a subfolder for each postprocessor under the model's folder
                            postprocessor_results_dir = os.path.join(model_results_dir, postprocessor_name)
                            if not os.path.exists(postprocessor_results_dir):
                                os.makedirs(postprocessor_results_dir)

                            # Check if ood_metrics exists and is a DataFrame
                            if isinstance(ood_metrics, pd.DataFrame):
                                ood_metrics['id_dset']=id_dset
                                # Save the OOD metrics to CSV file in the postprocessor's subfolder
                                metrics_csv_path = os.path.join(postprocessor_results_dir, f"{postprocessor_name}_metrics.csv")
                                ood_metrics.to_csv(metrics_csv_path, index=True)  # Save the DataFrame to CSV directly
                            else:
                                print("No OOD metrics available to save.")

                            score_save_root = os.path.join(model_results_dir, 'scores')
                            if not os.path.exists(score_save_root):
                                os.makedirs(score_save_root)
                            with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
                                    'wb') as f:
                                pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)
                    except Exception as error:
                        # handle the exception
                        print(f"An exception occurred for {postprocessor_name}:\n", error) # An exception occurred: division by zero
