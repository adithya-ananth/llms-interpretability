import os
import torch
import logging
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Setup logging to file
log_file = "probe_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Write to file, overwrite each run
        # logging.StreamHandler()  # Also print to console
    ]
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load Dataset
def load_sst2_dataset(keep_samples: int = 100) -> Tuple[Dataset, Dataset, Dataset]:
    '''
    Load the SST-2 dataset from Hugging Face.
    The dataset contains short sentences and their sentiment labels (positive or negative).
    Label 1 = positive, Label 0 = negative.

    :param keep_samples: Number of samples to keep for faster training.
    :return: Tuple of train, dev, and test datasets. Each supports indexing and has 'text' and 'label' keys.
    '''
    dataset = load_dataset("stanfordnlp/sst2", download_mode="force_redownload")

    # Keep only a subset of the data for quick prototyping
    train_dataset = Dataset.from_dict(dataset['train'].shuffle(seed=42)[:keep_samples])
    dev_dataset = Dataset.from_dict(dataset['validation'].shuffle(seed=42)[:keep_samples])
    test_dataset = Dataset.from_dict(dataset['test'].shuffle(seed=42)[keep_samples:2*keep_samples])

    logging.info(f'Loaded SST-2 dataset: {len(train_dataset)} training, {len(dev_dataset)} dev, {len(test_dataset)} test samples.')
    return train_dataset, dev_dataset, test_dataset

# Split the dataset into train, dev, and test sets
train_dataset, dev_dataset, test_dataset = load_sst2_dataset(keep_samples=50)

# Load Model and Tokenizer
def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    '''
    Load a model from huggingface.
    :param model_name: Check huggingface for acceptable model names.
    :return: Model and tokenizer.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    model.config.max_position_embeddings = 128  # Reducing from default 512 to 128 for computational efficiency

    logging.info(f'Loaded model and tokenizer: {model_name} with {model.config.num_hidden_layers} layers, '
                 f'hidden size {model.config.hidden_size} and sequence length {model.config.max_position_embeddings}.')
    return model, tokenizer

model, tokenizer = load_model("distilbert-base-uncased-finetuned-sst-2-english")

# Get model architecture details
num_layers = model.config.num_hidden_layers
print(f'The model has {num_layers} layers.')

## Setting up the probe with embeddings from a specific layer
def get_embeddings_from_model(model: AutoModel, tokenizer: AutoTokenizer, layer_num: int, data: list[str], batch_size: int) -> torch.Tensor:
    '''
    Get fixed-size mean-pooled embeddings from a specific layer of a transformer model (e.g., DistilBERT).

    :param model: The transformer model (e.g., DistilBERT).
    :param tokenizer: Corresponding tokenizer for the model.
    :param layer_num: Which hidden layer to extract embeddings from (0-indexed).
    :param data: List of input sentences (strings).
    :param batch_size: Batch size for processing inputs.
    :return: A tensor of shape [num_samples, hidden_dim] representing pooled embeddings.
    '''
    logging.info(f'Getting embeddings from layer {layer_num} for {len(data)} samples...')

    model.eval()
    all_embeddings = []
    batch_num = 1

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            logging.debug(f'Getting embeddings for batch {batch_num}...')
            batch_num += 1

            # Tokenize the batch
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)

            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)

            # Extract hidden states for the given layer
            hidden_states = outputs.hidden_states[layer_num]  # Shape: [batch_size, seq_len, hidden_dim]

            # Apply mean pooling (mask out padding tokens)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)               # Shape: [batch_size, seq_len, 1]
            masked_hidden = hidden_states * attention_mask                        # Zero out embeddings of padding tokens
            sum_hidden = masked_hidden.sum(dim=1)                                 # Sum along sequence length
            lengths = attention_mask.sum(dim=1)                                   # Count non-padded tokens
            embeddings = sum_hidden / lengths                                     # Final shape: [batch_size, hidden_dim]

            all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)  # Final shape: [num_samples, hidden_dim]
    logging.info(f'Got embeddings for {len(data)} samples from layer {layer_num}. Final shape: {all_embeddings.shape}')
    return all_embeddings

def visualize_embeddings(embeddings: torch.Tensor, labels: list, layer_num: int, visualization_method: str = 't-SNE', save_plot: bool = False) -> None:
    '''
    Visualize the embeddings using t-SNE.
    :param embeddings: The embeddings to visualize. Shape is N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
    :param labels: The labels for the embeddings. A list of integers.
    :return: None
    '''

    # Since we are working with sentiment analysis, which is sentence based task, we can use sentence embeddings.
    # The sentence embeddings are simply the mean of the token embeddings of that sentence.
    sentence_embeddings = torch.mean(embeddings, dim=1)  # N, D

    # Convert to numpy
    sentence_embeddings = sentence_embeddings.detach().numpy()
    labels = np.array(labels)

    assert visualization_method in ['t-SNE', 'PCA'], "visualization_method must be one of 't-SNE' or 'PCA'"

    # Visualize the embeddings
    if visualization_method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=0)
        embeddings_2d = tsne.fit_transform(sentence_embeddings)
        xlabel = 't-SNE dimension 1'
        ylabel = 't-SNE dimension 2'
    if visualization_method == 'PCA':
        pca = PCA(n_components=2, random_state=0)
        embeddings_2d = pca.fit_transform(sentence_embeddings)
        xlabel = 'First Principal Component'
        ylabel = 'Second Principal Component'

    negative_points = embeddings_2d[labels == 0]
    positive_points = embeddings_2d[labels == 1]

    # Plot the embeddings. We want to colour the datapoints by label.
    fig, ax = plt.subplots()
    ax.scatter(negative_points[:, 0], negative_points[:, 1], label='Negative', color='red', marker='o', s=10, alpha=0.7)
    ax.scatter(positive_points[:, 0], positive_points[:, 1], label='Positive', color='blue', marker='o', s=10, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{visualization_method} of Sentence Embeddings - Layer{layer_num}')
    plt.legend()

    # Save the plot if needed, then display it
    if save_plot:
        plt.savefig(f'{visualization_method}_layer_{layer_num}.png')
    plt.show()

    logging.info(f'Visualized embeddings using {visualization_method}.')

# Probe Class
class Probe():
    def __init__(self, hidden_dim: int = 768, class_size: int = 2)  -> None:
        '''
        Initialize the probe.
        :param hidden_dim: The dimensionality of the hidden layer of the probe.
        :param num_layers: The number of layers in the probe.
        :return: None
        '''

        # The probe is a simple linear classifier, with a hidden layer and an output layer.
        # The input to the probe is the embeddings from the model, and the output is the predicted class.

        # Exercise: Try playing around with the hidden_dim and num_layers to see how it affects the probe's performance.
        # But watch out: if a complex probe performs well on the task, we don't know if the performance
        # is because of the model embeddings, or the probe itself learning the task!

        self.probe = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, class_size),

            # Add more layers here if needed

            # Sigmoid is used to convert the hidden states into a probability distribution over the classes
            torch.nn.Sigmoid()
        )


    def train(self, data_embeddings: torch.Tensor, labels: torch.Tensor, num_epochs: int = 10,
              learning_rate: float = 0.001, batch_size: int = 32) -> None:
        '''
        Train the probe on the embeddings of data from the model.
        :param data_embeddings: A tensor of shape N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape N, where N is the number of samples. Each element is the label for the corresponding sample.
        :param num_epochs: The number of epochs to train the probe for. An epoch is one pass through the entire dataset.
        :param learning_rate: How fast the probe learns. A hyperparameter.
        :param batch_size: Used to batch the data for computational efficiency. A hyperparameter.
        :return:
        '''

        # Setup the loss function (training objective) for the training process.
        # The cross-entropy loss is used for multi-class classification, and represents the negative log likelihood of the true class.
        criterion = torch.nn.CrossEntropyLoss()

        # Setup the optimization algorithm to update the probe's parameters during training.
        # The Adam optimizer is an extension to stochastic gradient descent, and is a popular choice.
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=learning_rate)

        # Train the probe
        logging.info('Training the probe...')
        for epoch in range(num_epochs):  # Pass over the data num_epochs times

            for i in range(0, len(data_embeddings), batch_size):

                # Iterate through one batch of data at a time
                batch_embeddings = data_embeddings[i:i+batch_size].detach()
                batch_labels = labels[i:i+batch_size]

                # Convert to sentence embeddings, since we are performing a sentence classification task
                # batch_embeddings = torch.mean(batch_embeddings, dim=1)  # N, D

                # Get the probe's predictions, given the embeddings from the model
                outputs = self.probe(batch_embeddings)

                # Calculate the loss of the predictions, against the true labels
                loss = criterion(outputs, batch_labels)

                # Backward pass - update the probe's parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        logging.info('Done.')


    def predict(self, data_embeddings: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        '''
        Get the probe's predictions on the embeddings from the model, for unseen data.
        :param data_embeddings: A tensor of shape N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
        :param batch_size: Used to batch the data for computational efficiency.
        :return: A tensor of shape N, where N is the number of samples. Each element is the predicted class for the corresponding sample.
        '''

        # Iterate through batches
        for i in range(0, len(data_embeddings), batch_size):

            # Iterate through one batch of data at a time
            batch_embeddings = data_embeddings[i:i+batch_size]

            # Convert to sentence embeddings, since we are performing a sentence classification task
            # batch_embeddings = torch.mean(batch_embeddings, dim=1)  # N, D

            # Get the probe's predictions
            outputs = self.probe(batch_embeddings)

            # Get the predicted class for each sample
            _, predicted = torch.max(outputs, 1)

            # Concatenate the predictions from each batch
            if i == 0:
                all_predicted = predicted
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)

        return all_predicted


    def evaluate(self, data_embeddings: torch.tensor, labels: torch.tensor, batch_size: int = 32) -> float:
        '''
        Evaluate the probe's performance by testing it on unseen data.
        :param data_embeddings: A tensor of shape N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape N, where N is the number of samples. Each element is the label for the corresponding sample.
        :return: The accuracy of the probe on the unseen data.
        '''

        # Iterate through batches
        for i in range(0, len(data_embeddings), batch_size):

            # Iterate through one batch of data at a time
            batch_embeddings = data_embeddings[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Convert to sentence embeddings, since we are performing a sentence classification task
            # batch_embeddings = torch.mean(batch_embeddings, dim=1)  # N, D

            # Get the probe's predictions
            with torch.no_grad():
                outputs = self.probe(batch_embeddings)

            # Get the predicted class for each sample
            _, predicted = torch.max(outputs, dim=-1)

            # Concatenate the predictions from each batch
            if i == 0:
                all_predicted = predicted
                all_labels = batch_labels
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)
                all_labels = torch.cat([all_labels, batch_labels], dim=0)

        # Calculate the accuracy of the probe
        correct = (all_predicted == all_labels).sum().item()
        accuracy = correct / all_labels.shape[0]
        logging.info(f'Probe accuracy: {accuracy:.2f}')

        return accuracy
    
# Probe object
probe = Probe()

# Analyze the embeddings from each layer
layer_wise_accuracies = []
best_probe, best_layer, best_accuracy = None, -1, 0
batch_size = 32

for layer_num in range(num_layers):
    logging.info(f'Evaluating representations of layer {layer_num}:\n')

    train_embeddings = get_embeddings_from_model(model, tokenizer, layer_num=layer_num, data=train_dataset['sentence'], batch_size=batch_size)
    dev_embeddings = get_embeddings_from_model(model, tokenizer, layer_num=layer_num, data=dev_dataset['sentence'], batch_size=batch_size)
    train_labels, dev_labels = torch.tensor(train_dataset['label'],  dtype=torch.long), torch.tensor(dev_dataset['label'],  dtype=torch.long)

    # Now, let's train the probe on the embeddings from the model.
    # Feel free to play around with the training hyperparameters, and see what works best for your probe.
    probe = Probe()
    probe.train(data_embeddings=train_embeddings, labels=train_labels,
                num_epochs=5, learning_rate=0.001, batch_size=8)

    # Let's see how well our probe does on a held out dev set
    accuracy = probe.evaluate(data_embeddings=dev_embeddings, labels=dev_labels)
    layer_wise_accuracies.append(accuracy)

    # Keep track of the best probe
    if accuracy > best_accuracy:
        best_probe, best_layer, best_accuracy = probe, layer_num, accuracy

logging.info(f'DONE.\n Best accuracy of {best_accuracy*100}% from layer {best_layer}.')

print(f'DONE.\n Best accuracy of {best_accuracy*100}% from layer {best_layer}.')

# Generate plot
plt.plot(layer_wise_accuracies)
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.title('Probe Accuracy by Layer')
plt.grid(alpha=0.3)
plt.show()
