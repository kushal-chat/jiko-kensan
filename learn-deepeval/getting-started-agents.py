# Author: Kushal Chattopadhyay

from abc import ABC, abstractmethod
from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import EvaluationDataset, Golden

### DeepEval Abstract Agent
class DeepEvalAgent(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def tool(self):
        pass

    @abstractmethod
    def agent(self,):
        pass

### Python Agent
from argparse import ArgumentParser
from deepeval.tracing import observe

class PythonAgent(DeepEvalAgent):

    @observe
    def tool(self, city):
        return city

    @observe(metrics=[TaskCompletionMetric])
    def agent(self, city):
        print("Calling tool...")
        return self.tool(city)

    def __call__(self):
        self.agent(city="Tokyo")

### LangGraph Agent
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import EvaluationDataset, Golden

class LangGraphAgent(DeepEvalAgent):

    class State(TypedDict):
        city: str
        weather: str

    @staticmethod
    def tool(state: "LangGraphAgent.State") -> "LangGraphAgent.State":
        city = state["city"]
        return {
            "city": city,
            "weather": f"It's always sunny in {city}!"
        }

    def agent(self, city: str = "Tokyo"):
        builder = StateGraph(self.State)

        builder.add_node("get_weather", self.tool)
        builder.add_edge(START, "get_weather")
        builder.add_edge("get_weather", END)

        graph = builder.compile()
        print(graph.get_graph().draw_ascii())

        input_state = {"city": city, "weather": ""}
        result = graph.invoke(
            input=input_state,
            config={"callbacks": [CallbackHandler(metrics=[task_completion_metric])]},
        )

        return result

if __name__ == "__main__":

    task_completion_metric = TaskCompletionMetric(model="gpt-4.1")
    dataset = EvaluationDataset(goldens=[Golden(input="Tokyo")])

    deepeval_agent = LangGraphAgent()
    for golden in dataset.evals_iterator():
        output = deepeval_agent.agent(city=golden.input)
        print(output)