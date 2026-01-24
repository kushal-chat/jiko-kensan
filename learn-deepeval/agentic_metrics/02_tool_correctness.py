"""
# The tool correctness metric is an agentic LLM metric that assesses your LLM agent's function/tool calling ability. 
# It is calculated by comparing whether every tool that is expected to be used was indeed called,
# and if the selection of the tools made by the LLM agent were the most optimal.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    tools_called=[ToolCall(name="WebSearch"), ToolCall(name="ToolQuery")],
    expected_tools=[ToolCall(name="WebSearch")],
)
"""

# To use the ToolCorrectnessMetric, you'll have to provide the following arguments when creating an LLMTestCase:
# input,
# actual_output, NOTE: actually not needed in practice.
# tools_called,
# expected_tools.

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric

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

    update_current_trace(
        test_case=LLMTestCase(
            input=input,
            tools_called=[ToolCall(name="ItineraryGenerator"), ToolCall(name="RestaurantFinder")],
            expected_tools=[ToolCall(name="ItineraryGenerator")]
        )
    )

    return itinerary + restaurants


# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="This is a test query")])

# Initialize metric
tool_correctness = ToolCorrectnessMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset
for golden in dataset.evals_iterator(metrics=[tool_correctness]):
    trip_planner_agent(golden.input)