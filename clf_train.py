import pickle
import warnings
import yaml
import json
from typing import Dict, List, Tuple

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, classification_report, \
    average_precision_score
from transformers import EvalPrediction

warnings.filterwarnings('ignore')

from clf_dataset import MultiLabelDataset
from preprocess import extract_title
from preprocess import extract_labels
from preprocess import get_encoded_labels
from clf_custom_trainer import WeightedMultiLabelClassifierTrainer

with open('genre_clf_train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# path = 'train.json'
# model_id = 'distilbert-base-uncased'
# threshold = 0.1
# num_trainable_layers = 1
# min_examples_threshold = 350
# min_class_weight = 0.1
# max_class_weight = 1
# train_split_ratio = 0.8
# top_n = 3



    def convert_prediction_to_one_hot(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> np.ndarray:
        """
        Converts the predicted probabilities into one-hot encoded predictions.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted probabilities.
            n (int): The number of top predictions to consider. Defaults to 3.

        Returns:
            np.ndarray: The one-hot encoded predictions.

        """

        sigmoid = torch.nn.Sigmoid()
        y_pred = sigmoid(torch.Tensor(y_pred)).numpy()
        top_n_pred = np.argsort(y_pred, axis=1)[:, :-(n + 1):-1]

        top_n_pred_binary = np.zeros_like(y_true)
        for i, j in enumerate(top_n_pred):
            top_n_pred_binary[i, j] = 1
        return top_n_pred_binary


    def top_n_metrics(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> Dict[str, float]:
        """
        Calculates various evaluation metrics for the top N predictions.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted probabilities.
            n (int): The number of top predictions to consider. Defaults to 3.

        Returns:
            Dict[str, float]: A dictionary of evaluation metric names and their corresponding values.

        """
        prediction_binary = convert_prediction_to_one_hot(y_true, y_pred, n=n)
        metrics_top3 = {
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
            'jaccard_score': jaccard_score
        }
        at_least_one_true = np.any(np.logical_and(y_true, prediction_binary), axis=1)
        percentage_top_n_true = np.mean(at_least_one_true)
        results_top_n = {}

        for name, metric in metrics_top3.items():
            results_top_n[name] = metric(y_true, prediction_binary, average='weighted')
        average_precision_val = average_precision_score(y_true, y_pred, average='weighted')
        results_top_n['average_precision_score'] = average_precision_val
        results_top_n['percentage_examples_at_least_1_true'] = percentage_top_n_true
        return results_top_n


    def metric_per_label(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], col_to_filter: List[int] = [],
                         n: int = 3
                         ) -> pd.DataFrame:
        """
        Calculates evaluation metrics per label and generates a report.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted probabilities.
            class_names (List[str]): List of class names.
            col_to_filter (List[int]): Optional. List of column indices to filter. Defaults to an empty list.
            n (int): The number of top predictions to consider. Defaults to 3.

        Returns:
            pd.DataFrame: A report containing evaluation metrics per label.

        """
        if col_to_filter:
            y_pred = y_pred[:, col_to_filter]
            y_true = y_true[:, col_to_filter]

        y_pred = convert_prediction_to_one_hot(y_true, y_pred, n=n)
        report = pd.DataFrame(
            classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
            columns=class_names).transpose().sort_values('support', ascending=False)
        return report


    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        """
        Computes evaluation metrics using the predicted probabilities and true labels.

        Args:
            p (EvalPrediction): The evaluation prediction object from the transformers library.

        Returns:
            Dict[str, float]: The computed evaluation metrics.

        """

        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions

        result = top_n_metrics(
            y_pred=preds,
            y_true=p.label_ids)
        return result


    def create_datasets(text: List[str], labels: List[List[int]], tokenizer: AutoTokenizer) -> \
            Tuple[MultiLabelDataset, MultiLabelDataset]:
        """
        Creates train and test datasets for multilabel classification.

        Args:
            text (List[str]): List of texts.
            labels (List[List[int]]): List of label lists.
            tokenizer (PreTrainedTokenizer): Tokenizer object for tokenizing the texts.

        Returns:
            Tuple[Dataset, Dataset]: Train and test datasets.

        """
        dataset = MultiLabelDataset(texts=text, tokenizer=tokenizer, labels=labels)
        train_size = int(config['data']['train_split_ratio'] * len(dataset))
        val_size = len(dataset) - train_size
        train, test = random_split(dataset, [train_size, val_size])
        return train, test


    def get_class_weights(encoder: MultiLabelBinarizer, labels: List[List[int]],
                          label_order: List[str]
                          ) -> np.ndarray:
        """
        Computes class weights based on label frequencies and predefined threshold of minimum examples.

        Args:
            encoder (LabelEncoder): The label encoder.
            labels (List[List[int]]): The labels.
            label_order (List[str]): The order of labels.

        Returns:
            np.ndarray: The computed class weights.

        """
        classes_freq = pd.DataFrame(pd.DataFrame(labels, columns=encoder.classes_).sum(),
                                    columns=['count']).sort_values('count', ascending=False)
        classes_freq_srs = classes_freq['count']
        classes_freq_srs.index = pd.Categorical(classes_freq_srs.index, categories=label_order, ordered=True)

        sorted_series = classes_freq_srs.sort_index()
        class_frq_arr = np.array(sorted_series)
        class_weights = np.where(class_frq_arr > config['metrics']['min_examples_threshold'],
                                 config['model']['max_class_weight'], config['model']['min_class_weight'])
        return class_weights


    def freeze_model_layers(model: AutoModelForSequenceClassification) -> AutoModelForSequenceClassification:
        """
        Freezes specified layers of a model by setting their requires_grad attribute to False.

        Args:
            model (PreTrainedModel): The model to freeze layers.

        Returns:
            PreTrainedModel: The model with frozen layers.

        """

        num_layers = model.config.num_hidden_layers
        for name, param in model.named_parameters():
            if any(f"transformer.layer.{i}." in name for i in
                   range(0, num_layers - config['model']['num_trainable_layers'])):
                param.requires_grad = False
        return model


def main():
    """
    Main function for the program.

    """
    with open(config['data']['path'], 'r') as f:
        data = [json.loads(line) for line in f]
    # title_list = [extract_title(exmp) for exmp in data]
    summary_plots = [exmp['plot_summary'] for exmp in data]
    labels_list = [extract_labels(exmp) for exmp in data]
    encoded_labels, label_encoder = get_encoded_labels(labels_list)
    num_classes = len(label_encoder.classes_)
    label_order = list(label_encoder.classes_)
    class_weights = get_class_weights(label_encoder, encoded_labels, label_order)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_id'])

    model = AutoModelForSequenceClassification.from_pretrained(config['model']['model_id'], num_labels=num_classes)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    train_dataset, val_dataset = create_datasets(text=summary_plots[:100], labels=encoded_labels[:100],
                                                 tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="./fine-tuned-model",
        overwrite_output_dir=True,
        num_train_epochs=config['model']['epochs'],
        per_device_train_batch_size=config['model']['batch_size'],
        per_device_eval_batch_size=config['model']['batch_size'],
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    model = freeze_model_layers(model)
    trainer = WeightedMultiLabelClassifierTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights=class_weights,
        compute_metrics=compute_metrics
    )
    trainer.train()
    predictions = trainer.predict(val_dataset)
    logits = torch.tensor(predictions.predictions)

    class_weight_srs = pd.Series(class_weights)
    high_freq_classes = class_weight_srs[class_weight_srs == 1].index.tolist()
    class_names = pd.Series(label_order)[high_freq_classes].values
    report = metric_per_label(predictions.label_ids, logits, class_names, col_to_filter=high_freq_classes,
                              n=config['model']['top_n'])
    tokenizer.save_pretrained(config['model']['fined_tune_tokenizer_path'])
    trainer.save_model(config['model']['fined_tune_model_path'])
    with open(config['model']['label_encoder_path'], 'wb') as f:
        pickle.dump(label_encoder, f)
    print(report)
    print(trainer.evaluate(val_dataset))


if __name__ == '__main__':
    main()
