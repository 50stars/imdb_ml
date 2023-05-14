import json
import os
import yaml
from typing import Dict, List, Tuple
import pickle

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from preprocess import extract_title
from preprocess import extract_labels

# Load movie data (assuming you have a dataset with 'title' and 'plot' columns)
with open('similarity_config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class CustomCache:
    def __init__(self, maxsize=config['model']['max_cache']):
        self.cache = {}
        self.order = []
        self.maxsize = maxsize

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        if len(self.order) >= self.maxsize:
            oldest_key = self.order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.order.append(key)


cache = CustomCache()


class MovieEmbeddings:
    """
    A class used to generate and manipulate movie embeddings.
    """

    def __init__(self, embedding_config: Dict):
        """
        Constructs all the necessary attributes for the MovieEmbeddings object.

        Args:
            embedding_config (dict): The configuration dictionary containing data path and model details.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.config = embedding_config
        self.model = SentenceTransformer(embedding_config['model']['model_id'])

        with open(self.config['data']['path'], 'r') as f:
            data = [json.loads(line) for line in f]

        title_arr = [extract_title(example) for example in data]
        summary_plots = [example['plot_summary'] for example in data]
        generes_arr = [extract_labels(example) for example in data]

        summary_plots = [plot.lower() for plot in summary_plots]

        self.df = pd.DataFrame({'plots': summary_plots, 'generes': generes_arr, 'title': title_arr})

        if os.path.isfile(config['model']['embedding_path']):
            with open(config['model']['embedding_path'], 'rb') as f:
                self.plot_embeddings = pickle.load(f)
        else:
            self.plot_embeddings = self.model.encode(summary_plots, convert_to_tensor=True)
            with open(config['model']['embedding_path'], 'wb') as f:
                pickle.dump(self.plot_embeddings, f)

        float_embeddings = self.plot_embeddings.numpy().astype(np.float32)

        self.index = faiss.IndexFlatL2(float_embeddings.shape[1])
        self.index.add(float_embeddings)

    def encode_new_embedding(self):
        """
        Encodes new embeddings and saves them.

        Returns:
            None
        """
        self.plot_embeddings = self.model.encode(self.df['plots'], convert_to_tensor=True)
        with open(config['model']['embedding_path'], 'wb') as f:
            pickle.dump(self.plot_embeddings, f)

    def find_proximate_movie_indices(self, query: str, n: int = config['model']['top_n']) -> np.ndarray:
        """
        Finds the indices of n most similar movies to the query.

        Args:
            query (str): The query string.
            n (int, optional): The number of most similar movies to return (default is 10).

        Returns:
            np.ndarray: The indices of n most similar movies.
        """

        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        _, most_similar_indices = self.index.search(query_embedding, n)
        return most_similar_indices[0]

    def get_similar_titles(self, query: str, n: int = 10) -> List:
        """
        Returns titles of n most similar movies to the query.

        Args:
            query (str): The query string.
            n (int, optional): The number of most similar movies to return (default is 10).

        Returns:
            List[str]: The titles of n most similar movies.
        """
        cached_result = cache.get(query)
        if cached_result is not None:
            return cached_result
        similar_indices = self.find_proximate_movie_indices(query, n=n)
        titles = [self.df['title'][i] for i in similar_indices]
        cache.set(query, titles)
        return titles

    def calculate_stats(self, current_ind: int, results_ind: List) -> Tuple:
        """
        Calculates the IoU and the hit rate for the results.

        Args:
            current_ind (int): The index of the current movie.
            results_ind (List[int]): The indices of the result movies.

        Returns:
            Tuple[float, float]: The IoU and the hit rate.
        """

        genere_input_set = set(self.df['generes'][current_ind])
        intersections = [len(genere_input_set & set(self.df['generes'][movie_ind])) for movie_ind in results_ind]
        unions = [len(genere_input_set | set(self.df['generes'][movie_ind])) for movie_ind in results_ind]

        iou = np.mean(np.nan_to_num(np.array(intersections) / np.array(unions)))
        clipped_list = np.mean(np.minimum(intersections, 1))
        return iou, clipped_list

    def evaluate(self) -> Tuple:
        """
        Evaluates the model by calculating mean IoU and hit rate.

        Returns:
            Tuple[float, float]: The mean IoU and hit rate.
        """
        iou_results = []
        at_one_hit = []
        top_k = config['model']['top_n'] + 1  # the model will return the input so we need to increase by one
        for curr_ind, curr_plot in enumerate(self.df['plots']):
            result = self.find_proximate_movie_indices(curr_plot, n=top_k)
            if curr_ind in result:  # remove the extra output
                result = result[result != curr_ind]
            else:
                result = result[:-1]

            curr_iou, curr_at_one_hit = self.calculate_stats(curr_ind, result)
            iou_results.append(curr_iou)
            at_one_hit.append(curr_at_one_hit)

        mean_iou = np.mean(iou_results)
        stdev_iou = np.std(iou_results)
        mean_min_one_hit = np.mean(at_one_hit)
        stdev_min_one_hit = np.std(mean_min_one_hit)

        print(f'Mean of Intersection over union over all examples is: {mean_iou}')
        print(f'Standard deviation of Intersection over union over all examples is: {stdev_iou}')
        print(
            f'Mean of Intersection of at least one genere intersected between query and suggested movie over'
            f' all examples is: {mean_min_one_hit}')
        print(f'Standard deviation of Intersection of at least one genere intersected between query and '
              f'suggested movie over'
              f' all examples is: {stdev_min_one_hit}')
        return mean_iou, mean_min_one_hit


def main():
    movie_embeddings = MovieEmbeddings(config)
    movie_embeddings.evaluate()


if __name__ == '__main__':
    main()
