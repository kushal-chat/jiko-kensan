from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe
from deepeval.metrics import PlanQualityMetric

@observe()
def x():
    print("Hi")

    @observe(metrics=[PlanQualityMetric()])
    def tool():

        print("Hi")
    
    return tool()

dataset = EvaluationDataset(goldens = [Golden(input = "hello")])

# ~~~~~~~~~~~~~

from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

@observe()
def agent(g: str):

    @observe(metrics=[AnswerRelevancyMetric(include_reason=True, verbose_mode=True)])
    def tool():
        update_current_span(test_case=LLMTestCase(input="What is the city where I was born", actual_output="You mean why is the sky blue?"))
        return "yooo"
    
    return(tool() + f" {g}")

eval_dataset = EvaluationDataset(goldens = [Golden(input = "hello")])
for golden in eval_dataset.evals_iterator():
    agent(golden.input)

# ~~~~~~~~~~~~~~


from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    # Replace this with the tools that was actually used by your LLM agent
    tools_called=[ToolCall(name="WebSearch"), ToolCall(name="ToolQuery")],
    expected_tools=[ToolCall(name="WebSearch")],
)
metric = ToolCorrectnessMetric()

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])