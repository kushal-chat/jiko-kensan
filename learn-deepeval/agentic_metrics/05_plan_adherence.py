"""
# The Plan Adherence metric is an agentic metric that extracts the task and plan from your agent's trace which are then used to evaluate how well your agent has adhered to the plan in completing the task. 
# It is a self-explaining eval, which means it outputs a reason for its metric score.

"""

from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanAdherenceMetric

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
metric = PlanAdherenceMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset
for golden in dataset.evals_iterator(metrics=[metric]):
    trip_planner_agent(golden.input)