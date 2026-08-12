import colorama
import os
import logging
from typing import Tuple, List
from tqdm import tqdm
from config.dataset_config import drnn_paths_config
from nn.mpdrnn_pruning.base_class_ipmpdrnn import BaseIPMPDRNN
from nn.models.model_selector import ModelFactory
from utils.utils import (average_columns_in_excel, create_timestamp, insert_data_to_excel, setup_logger,
                         reorder_metrics_lists, get_num_of_neurons, plot_confusion_matrix_mpdrnn, extract_float)


class IPMPDRNN(BaseIPMPDRNN):
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, override_cfg: dict = None, celery_task=None):
        super().__init__(override_cfg=override_cfg, celery_task=celery_task)
        # Create timestamp as the program begins to execute.
        self.timestamp = create_timestamp()
        # Setup logger and colour
        setup_logger()
        colorama.init()
        penalty_term = self.cfg.get('penalty')
        rcond_cfg = self.cfg.get("rcond")
        rcond = rcond_cfg.get(self.method) if isinstance(rcond_cfg, dict) else rcond_cfg
        num_aux_models = self.cfg.get('num_aux_net', 1)
        sp = self.cfg.get('subset_percentage', 0.1)
        neurons = self.cfg.get("exp_neurons") or get_num_of_neurons(self.cfg, self.method)
        drnn_config = drnn_paths_config(self.dataset_name)

        base_results_dir = drnn_config.get("ipmpdrnn").get("path_to_results")
        self.excel_dir = os.path.join(base_results_dir, "excel")
        self.cm_dir = os.path.join(base_results_dir, "confusion_matrix")
        os.makedirs(self.excel_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)

        self.hyperparam_config = {
            "rcond": rcond,
            "penalty_term": penalty_term,
            "subset_percentage": sp,
            "num_aux_net": num_aux_models,
            "neurons": neurons
        }
        # Save path
        rcond_str = f"{rcond:.4f}" if rcond is not None else "none"
        self.save_filename = (
            os.path.join(
                self.excel_dir,
                f"{self.timestamp}_ipmpdrnn_{self.dataset_name}_{self.method}_sp_{sp}_rcond_{rcond_str}.xlsx")
        )
        self.all_run_metrics = []

    def main(self) -> Tuple[str, List[float]]:
        """
        Returns:
        """
        # Load data
        training_time = []
        num_tests = self.cfg.get('number_of_tests', 1)
        for i in tqdm(range(num_tests), desc=colorama.Fore.CYAN + "Process"):
            if self.celery_task:
                is_aborted = self.celery_task.backend.client.get(f"ipmpdrnn:abort:{self.celery_task.request.id}")
                if is_aborted:
                    logging.info("IPMPDRNN testing series abort signal detected mid-cycle. Breaking loop.")
                    break
                current_test = i + 1
                progress_percent = round((current_test / num_tests) * 100, 1)
                self.celery_task.update_state(
                    state="PROGRESS",
                    meta={
                        "status": f"Running cycle {current_test}/{num_tests}",
                        "telemetry": {
                            "current_cycle": current_test,
                            "total_cycles": num_tests,
                            "progress_percent": progress_percent
                        }
                    }
                )

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

            train_cms = [
                initial_model_subs_weights_training_metrics[4],
                subsequent_model_model_subs_weights_training_metrics[4],
                final_model_model_subs_weights_training_metrics[4]
            ]
            test_cms = [
                initial_model_subs_weights_testing_metrics[4],
                subsequent_model_model_subs_weights_testing_metrics[4],
                best[4]
            ]
            path_to_plots = self.cm_dir
            class_labels = self.gen_ds_cfg.get("class_labels")
            file_prefix = f"{self.timestamp}_cycle_{i}_"

            plot_confusion_matrix_mpdrnn(
                cm=train_cms,
                path_to_plot=path_to_plots,
                name_of_dataset=self.dataset_name,
                operation="train",
                method=self.method,
                labels=class_labels,
                prefix=file_prefix
            )
            plot_confusion_matrix_mpdrnn(
                cm=test_cms,
                path_to_plot=path_to_plots,
                name_of_dataset=self.dataset_name,
                operation="test",
                method=self.method,
                labels=class_labels,
                prefix=file_prefix
            )

            clean_metrics = [extract_float(m) for m in metrics[0]]
            self.all_run_metrics.append(clean_metrics)

            training_time.clear()

        if self.all_run_metrics:
            num_runs = len(self.all_run_metrics)
            num_metrics = len(self.all_run_metrics[0])
            self.averaged_metrics = [
                sum(run[j] for run in self.all_run_metrics) / num_runs
                for j in range(num_metrics)
            ]
        else:
            self.averaged_metrics = []

        if os.path.exists(self.save_filename):
            average_columns_in_excel(self.save_filename)

        return self.save_filename, self.averaged_metrics


if __name__ == "__main__":
    try:
        ipmpdrnn = IPMPDRNN()
        ipmpdrnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f"Keyboard interrupt received: {kie}")