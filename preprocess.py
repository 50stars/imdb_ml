from typing import Dict, List

from sklearn.preprocessing import MultiLabelBinarizer


def extract_labels(movie: Dict[str, any]) -> List[str]:
    """
    Extracts labels from a movie dictionary.

    Args:
        movie (Dict[str, any]): The movie dictionary.

    Returns:
        List[str]: The extracted labels as a list.

    """
    if 'genres' in movie.keys():
        label = movie['genres']
        arr_label = list(label.values())
    else:
        return ['none']

    return arr_label


def extract_title(movie: Dict[str, any]) -> str:
    """
    Extracts the title from a movie dictionary.

    Args:
        movie (Dict[str, any]): The movie dictionary.

    Returns:
        str: The extracted title as a string, or 'none' if 'title' is not present in the movie dictionary.

    """
    if 'title' in movie.keys():
        title = movie['title']
    else:
        return 'none'

    return title


def get_encoded_labels(labels):
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    encoded_labels = mlb.transform(labels)
    encoded_labels = encoded_labels.tolist()
    return encoded_labels, mlb
