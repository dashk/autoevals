from autoevals import EmbeddingSimilarity
from autoevals.string import StringListSimilarity

SYNONYMS = [
    ("water", ["water", "H2O", "agua"]),
    ("fire", ["fire", "flame"]),
    ("earth", ["earth", "Planet Earth"]),
]

UNRELATED = ["water", "The quick brown fox jumps over the lazy dog", "I like to eat apples"]


def test_embeddings():
    evaluator = EmbeddingSimilarity(prefix="resource type: ")
    for word, synonyms in SYNONYMS:
        for synonym in synonyms:
            result = evaluator(word, synonym)
            print(f"[{word}]", f"[{synonym}]", result)
            assert result.score > 0.66

    for i in range(len(UNRELATED)):
        for j in range(len(UNRELATED)):
            if i == j:
                continue

            word1 = UNRELATED[i]
            word2 = UNRELATED[j]
            result = evaluator(word1, word2)
            print(f"[{word1}]", f"[{word2}]", result)
            assert result.score < 0.5


def test_list_diff_similarity():
    related_pairs = [
        (
            ["Review the minutes from our last meeting.", "Talk about how far we've come with our ongoing projects."],
            ["Go over the minutes from the previous meeting.", "Discuss current project progress."],
        ),
        (
            ["Set new objectives for the team.", "Distribute assignments among team members."],
            ["Establish fresh goals for our group.", "Assign tasks to team members."],
        ),
        (
            ["Address any issues or concerns.", "Plan for the next team meeting."],
            ["Talk about any problems or areas of concern.", "Schedule our next meeting."],
        ),
        (
            ["Provide updates on key performance indicators.", "Come up with ideas to tackle obstacles."],
            ["Give an update on important metrics and figures.", "Brainstorm solutions to challenges."],
        ),
        (
            ["Review the current budget and financial status.", "Assess how the team is performing."],
            ["Go over our budget and financial situation.", "Evaluate our team's performance."],
        ),
    ]

    evaluator = StringListSimilarity()
    for a, b in related_pairs:
        actual = evaluator.eval(a, b)
        print(f"[{a}]", f"[{b}]", actual)
        assert actual.error is None
        assert actual.score >= 0.66
