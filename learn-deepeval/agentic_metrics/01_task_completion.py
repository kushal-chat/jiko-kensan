"""
# The task completion metric uses LLM-as-a-judge to evaluate how effectively an LLM agent accomplishes a task. 
# Task Completion is a self-explaining LLM-Eval, meaning it outputs a reason for its metric score.

There are SEVEN optional parameters when creating a TaskCompletionMetric:

[Optional] threshold: a float representing the minimum passing threshold, defaulted to 0.5.
[Optional] task: a string representing the task to be completed. If no task is supplied, it is automatically inferred from the trace. Defaulted to the None
[Optional] model: a string specifying which of OpenAI's GPT models to use, OR any custom LLM model of type DeepEvalBaseLLM. Defaulted to 'gpt-4o'.
[Optional] include_reason: a boolean which when set to True, will include a reason for its evaluation score. Defaulted to True.
[Optional] strict_mode: a boolean which when set to True, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to False.
[Optional] async_mode: a boolean which when set to True, enables concurrent execution within the measure() method. Defaulted to True.
[Optional] verbose_mode: a boolean which when set to True, prints the intermediate steps used to calculate said metric to the console, as outlined in the How Is It Calculated section. Defaulted to False.
"""

from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric

@observe()
def trip_planner_agent(input):
    destination = "Paris"
    days = 2

    @observe()
    def restaurant_finder(city):
        output = ["Le Jules Verne", "Angelina Paris", "Septime"]
        return output

    @observe()
    def itinerary_generator(destination, days):
        output = ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]
        return output

    itinerary = itinerary_generator(destination, days)
    restaurants = restaurant_finder(destination)

    return itinerary + restaurants


# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="This is a test query")])

# Initialize metric
# Note that the task is automatically inferred from the trace
task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset
for golden in dataset.evals_iterator(metrics=[task_completion]):
    trip_planner_agent(golden.input)