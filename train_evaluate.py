
def score_model(predictions, true_values):
    """
    Calculate the accuracy score for the given predictions and true values.

    Parameters:
    predictions (array-like): The predicted values.
    true_values (array-like): The actual values.

    Returns:
    float: The accuracy score.
    """
    return accuracy_score(true_values, predictions)
