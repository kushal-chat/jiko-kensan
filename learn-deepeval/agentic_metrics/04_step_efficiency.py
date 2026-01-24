"""
# The Step Efficiency metric is an agentic metric that extracts the task from your agent's trace and evaluates the efficiency of your agent's execution steps in completing that task. 
# It is a self-explaining eval, which means it outputs a reason for its metric score.

"""

from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric

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
metric = StepEfficiencyMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset
for golden in dataset.evals_iterator(metrics=[metric]):
    trip_planner_agent(golden.input)