from typing import Any, List, Optional

from braintrust_core.score import Score, Scorer
from Levenshtein import distance
from numpy import array, clip, ndarray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from .oai import arun_cached_request, run_cached_request


class Levenshtein(Scorer):
    """
    A simple scorer that uses the Levenshtein distance to compare two strings.
    """

    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("LevenshteinScorer requires an expected value")

        output, expected = str(output), str(expected)
        max_len = max(len(x) for x in [output, expected])

        score = 1
        if max_len > 0:
            score = 1 - (distance(output, expected) / max_len)

        return Score(name=self._name(), score=score)


LevenshteinScorer = Levenshtein  # backcompat


def cosine_similarity(list1, list2):
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(list1, list2))

    # Calculate the magnitude of each list
    magnitude_list1 = sum(a**2 for a in list1) ** 0.5
    magnitude_list2 = sum(b**2 for b in list2) ** 0.5

    # Calculate cosine similarity
    if magnitude_list1 * magnitude_list2 == 0:
        # Avoid division by zero
        return 0
    else:
        # Sometimes, rounding errors cause the dot product to be slightly > 1
        return min(dot_product / (magnitude_list1 * magnitude_list2), 1)


class EmbeddingSimilarity(Scorer):
    """
    A simple scorer that uses cosine similarity to compare two strings.
    """

    MODEL = "text-embedding-ada-002"

    def __init__(self, prefix="", model=MODEL, expected_min=0.7, api_key=None, base_url=None):
        """
        Create a new EmbeddingSimilarity scorer.

        :param prefix: A prefix to prepend to the prompt. This is useful for specifying the domain of the inputs.
        :param model: The model to use for the embedding distance. Defaults to "text-embedding-ada-002".
        :param expected_min: The minimum expected score. Defaults to 0.7. Values below this will be scored as 0, and
        values between this and 1 will be scaled linearly.
        """
        self.prefix = prefix
        self.expected_min = expected_min

        self.extra_args = {"model": model}
        if api_key:
            self.extra_args["api_key"] = api_key
        if base_url:
            self.extra_args["base_url"] = base_url

    async def _run_eval_async(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("EmbeddingSimilarity requires an expected value")

        output_embedding_p = arun_cached_request(input=f"{self.prefix}{output}", **self.extra_args)
        expected_embedding_p = arun_cached_request(input=f"{self.prefix}{expected}", **self.extra_args)

        output_result, expected_result = await output_embedding_p, await expected_embedding_p
        return Score(
            name=self._name(),
            score=self.scale_score(
                cosine_similarity(output_result["data"][0]["embedding"], expected_result["data"][0]["embedding"]),
                self.expected_min,
            ),
        )

    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("EmbeddingSimilarity requires an expected value")

        output_result = run_cached_request("embed", input=f"{self.prefix}{output}", **self.extra_args)
        expected_result = run_cached_request("embed", input=f"{self.prefix}{expected}", **self.extra_args)

        return Score(
            name=self._name(),
            score=self.scale_score(
                cosine_similarity(output_result["data"][0]["embedding"], expected_result["data"][0]["embedding"]),
                self.expected_min,
            ),
        )

    @staticmethod
    def scale_score(score, expected_min):
        return max((score - expected_min) / (1 - expected_min), 0)


class StringListSimilarity(Scorer):

    MODEL = "text-embedding-ada-002"

    def __init__(self, min_distance: float = 0.65, embedding_template: str = "{}", model: str = MODEL):
        """
        Create a new StringListSimilarity scorer.

        :param prefix: A prefix to prepend to the prompt. This is useful for specifying the domain of the inputs.
        :param min_distance: The minimum distance between the two lists. Defaults to 0.65.
        :param embedding_template: The template to use for the embedding. Defaults to "{}".
        """
        self.min_distance = min_distance
        self.embedding_template = embedding_template

        self.extra_args = {"model": model}

    def __compute_embedding_array(self, input_array: List[str]) -> List[List[float]]:
        output = []

        for item in input_array:
            embedding_response = run_cached_request(
                "embed", input=self.embedding_template.format(item), **self.extra_args
            )
            output.append(embedding_response["data"][0]["embedding"])

        return output

    def _run_eval_sync(self, output: List[str], expected: Optional[List[str]] = None, **kwargs):
        if expected is None:
            raise ValueError(f"{self._name()} requires an expected value")

        if output == expected:
            return Score(name=self._name(), score=1.0)

        # if len(output) != len(expected):
        #     return Score(
        #         name=self._name(),
        #         score=0.0,
        #     )

        # @TODO: Confirm the following scenarios
        # 1. Output length differs from expected length
        # 2. Repeated items (Either in output and/or expected)
        metadata: Any = {
            "total_similarity": 0.0,
            "base_score": 0.0,
            "missing_unexpected_penalty": 0.0,
            "final_score": 0.0,
            "best_pairings": [],
            "missing_expected": [],
            "unexpected_output": [],
        }

        output_embeddings = self.__compute_embedding_array(output)
        expected_embeddings = self.__compute_embedding_array(expected)

        # Compute similarity matrix
        distances: ndarray = pairwise_distances(
            X=array(output_embeddings),
            Y=array(expected_embeddings),
            # metric=cosine_similarity,
            metric="cosine",
        )

        # Squashing the distances because totally different items still get like 0.7 similarity
        distances = 1.0 - distances
        distances = (distances - self.min_distance) / (1.0 - self.min_distance)
        distances = clip(distances, 0.0, 1.0)
        distances = abs(1.0 - distances)

        # Solve the assignment problem to find the best pairings
        output_ind, expected_ind = linear_sum_assignment(distances)

        paired_output = set()
        paired_expected = set()
        for output_i, expected_j in zip(output_ind, expected_ind):
            metadata["best_pairings"].append(
                {
                    "output": output[output_i],
                    "expected": expected[expected_j],
                    "similarity": 1.0 - distances[output_i, expected_j],
                }
            )
            paired_output.add(output[output_i])
            paired_expected.add(expected[expected_j])

        metadata["missing_expected"] = list(set(expected) - paired_expected)
        metadata["unexpected_output"] = list(set(output) - paired_output)

        # The score is the sum of similarities for the best pairings
        similarities = 1.0 - distances[output_ind, expected_ind]

        # Sum similarities
        total_similarity = similarities.sum()
        metadata["total_similarity"] = total_similarity

        # Normalize the score by the number of items
        final_score = total_similarity / max(len(output), len(expected))
        metadata["final_score"] = final_score

        return Score(name=self._name(), score=min(1.0, max(0.0, final_score)), metadata=metadata)


__all__ = ["LevenshteinScorer", "Levenshtein", "EmbeddingSimilarity", "StringListSimilarity"]
