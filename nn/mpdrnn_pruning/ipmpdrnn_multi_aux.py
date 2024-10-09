import colorama
import os
import logging

from tqdm import tqdm

from config.dataset_config import drnn_paths_config
from nn.mpdrnn_pruning.base_class_ipmpdrnn import BaseIPMPDRNN
from nn.models.model_selector import ModelFactory
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         reorder_metrics_lists, get_num_of_neurons)


class IPMPDRNN(BaseIPMPDRNN):
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

        # Create timestamp as the program begins to execute.
        timestamp = create_timestamp()

        # Setup logger and colour
        setup_logger()
        colorama.init()

        penalty_term = self.cfg.get('penalty')
        rcond = self.cfg.get("rcond").get(self.method)
        num_aux_models = self.cfg.get('num_aux_net')
        sp = self.cfg.get('subset_percentage')
        neurons = get_num_of_neurons(self.cfg, self.method)

        drnn_config = drnn_paths_config(self.dataset_name)

        self.hyperparam_config = {
            "rcond": rcond,
            "penalty_term": penalty_term,
            "subset_percentage": sp,
            "num_aux_net": num_aux_models,
            "neurons": neurons
        }

        # Save path
        self.save_filename = (
            os.path.join(
                drnn_config.get("ipmpdrnn").get("path_to_results"),
                f"{timestamp}_ipmpdrnn_{self.dataset_name}_{self.method}_sp_{sp}_rcond_{rcond}.xlsx")
        )

    def main(self):
        """

        Returns:

        """

        # Load data
        training_time = []

        for i in tqdm(range(self.cfg.get('number_of_tests')), desc=colorama.Fore.CYAN + "Process"):
            # Create model
            net_cfg = self.get_network_config("MultiPhaseDeepRandomizedNeuralNetworkBase", self.hyperparam_config)
            self.initial_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkBase", net_cfg)

            # Train and evaluate the model
            self.initial_model, initial_model_training_metrics, initial_model_testing_metrics = (
                self.model_training_and_evaluation(
                    model=self.initial_model,
                    eval_set=self.test_loader,
                    weights=self.initial_model.beta_weights,
                    num_hidden_layers=1,
                    verbose=True
                )
            )
            training_time.append(self.initial_model.train_ith_layer.execution_time)

            # Pruning
            least_important_prune_indices = (
                self.prune_initial_model(
                    self.initial_model, set_weights_to_zero=False, config=self.hyperparam_config
                )
            )

            # Create aux model 1
            self.initial_model = (
                self.create_train_prune_initial_aux_model(model=self.initial_model,
                                                          model_type="MultiPhaseDeepRandomizedNeuralNetworkBase",
                                                          least_important_prune_indices=least_important_prune_indices,
                                                          config=self.hyperparam_config)
            )
            training_time.append(self.initial_model.train_ith_layer.execution_time)

            (self.initial_model,
             initial_model_subs_weights_training_metrics,
             initial_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=self.initial_model,
                    eval_set=self.test_loader,
                    weights=self.initial_model.beta_weights,
                    num_hidden_layers=1,
                    verbose=True
                )
            )
            training_time.append(self.initial_model.train_ith_layer.execution_time)

            # Subsequent Model

            net_cfg = (
                self.get_network_config(
                    "MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
                    self.hyperparam_config
                )
            )
            self.subsequent_model = (
                ModelFactory.create(
                    "MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
                    net_cfg)
            )

            (self.subsequent_model,
             subsequent_model_subs_weights_training_metrics,
             subsequent_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=self.subsequent_model,
                    eval_set=self.test_loader,
                    weights=[self.subsequent_model.extended_beta_weights,
                             self.subsequent_model.gamma_weights],
                    num_hidden_layers=2,
                    verbose=True
                )
            )
            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            least_important_prune_indices = self.prune_subsequent_model(
                model=self.subsequent_model,
                set_weights_to_zero=False,
                config=self.hyperparam_config
            )

            self.subsequent_model = (
                self.create_train_prune_subsequent_aux_model(
                    model=self.subsequent_model,
                    model_type="MultiPhaseDeepRandomizedNeuralNetworkSubsequent",
                    least_important_prune_indices=least_important_prune_indices,
                    config=self.hyperparam_config
                )
            )
            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            (self.subsequent_model,
             subsequent_model_model_subs_weights_training_metrics,
             subsequent_model_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=self.subsequent_model,
                    eval_set=self.test_loader,
                    weights=[self.subsequent_model.extended_beta_weights,
                             self.subsequent_model.gamma_weights],
                    num_hidden_layers=2,
                    verbose=True
                )
            )
            training_time.append(self.subsequent_model.train_ith_layer.execution_time)

            # Final Model

            net_cfg = (
                self.get_network_config(
                    network_type="MultiPhaseDeepRandomizedNeuralNetworkFinal",
                    config=self.hyperparam_config
                )
            )
            final_model = ModelFactory.create("MultiPhaseDeepRandomizedNeuralNetworkFinal", net_cfg)

            (final_model,
             final_model_subs_weights_training_metrics,
             final_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=final_model,
                    eval_set=self.test_loader,
                    weights=[final_model.extended_beta_weights,
                             final_model.extended_gamma_weights,
                             final_model.delta_weights],
                    num_hidden_layers=3,
                    verbose=True
                )
            )
            training_time.append(final_model.train_ith_layer.execution_time)

            least_important_prune_indices = self.prune_final_model(
                model=final_model,
                set_weights_to_zero=False,
                config=self.hyperparam_config
            )

            final_model = (
                self.create_train_prune_final_aux_model(
                    model=final_model,
                    model_type="MultiPhaseDeepRandomizedNeuralNetworkFinal",
                    least_important_prune_indices=least_important_prune_indices,
                    config=self.hyperparam_config
                )
            )
            training_time.append(final_model.train_ith_layer.execution_time)

            (final_model,
             final_model_model_subs_weights_training_metrics,
             final_model_model_subs_weights_testing_metrics) = (
                self.model_training_and_evaluation(
                    model=final_model,
                    eval_set=self.test_loader,
                    weights=[final_model.extended_beta_weights,
                             final_model.extended_gamma_weights,
                             final_model.delta_weights],
                    num_hidden_layers=3,
                    verbose=True
                )
            )
            training_time.append(final_model.train_ith_layer.execution_time)

            # Excel
            best = (
                final_model_subs_weights_testing_metrics) if (final_model_subs_weights_testing_metrics[0] >
                                                              final_model_model_subs_weights_testing_metrics[0]) else (
                final_model_model_subs_weights_testing_metrics
            )

            metrics = reorder_metrics_lists(train_metrics=final_model_model_subs_weights_training_metrics,
                                            test_metrics=best,
                                            training_time_list=training_time)
            insert_data_to_excel(self.save_filename, self.cfg.get("dataset_name"), i + 2, metrics)

            training_time.clear()

        average_columns_in_excel(self.save_filename)


if __name__ == "__main__":
    try:
        ipmpdrnn = IPMPDRNN()
        ipmpdrnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f"Keyboard interrupt received: {kie}")
