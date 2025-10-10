def get_values(dataset: str):

    '''
    Input: dataset name
    Returns: the label name, positive prediction, privileged group, unprivileged group
    '''

    if "adult" in dataset:
        return ("income", 1, {'sex': 1}, {'sex': 0})
    if "bank" in dataset:
        return ("loan", 1, {'age': 1}, {'age': 0})
    if "compas" in dataset:
        return ("two_year_recid", 0, {'race': 1}, {'race': 0})
    if "german" in dataset:
        return ("credit", 1, {'age': 1}, {'age': 0})
    if "crime" in dataset:
        return ("ViolentCrimesClass", 100, {'black_people': 1}, {'black_people': 0})
    if "drug" in dataset:
        return ("y", 0, {'race': 1}, {'race': 0})
    if "law" in dataset:
        return ("gpa", 2, {'race': 1}, {'race': 0})
    if "obesity" in dataset:
        return ("y", 0, {'Age': 1}, {'Age': 0})
    if "park" in dataset:
        return ("score_cut", 0, {'sex': 1}, {'sex': 0})
    if "wine" in dataset:
        return ("quality", 6, {'type': 1}, {'type': 0})