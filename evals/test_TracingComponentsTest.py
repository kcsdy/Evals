import sys, os

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import StepEfficiencyMetric, TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from agent_instrumented import support_agent as _support_agent

#Trace have expected values and actual values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace( expected_tools=golden.expected_tools )
        if golden.expected_output:
            update_current_trace( expected_output=golden.expected_output )

    return _support_agent( user_input )




task_completion = TaskCompletionMetric(threshold=0.7,model="gpt-4o")

tool_correctness = ToolCorrectnessMetric(
    available_tools=[ToolCall(name="get_order_status"), ToolCall(name="get_refund_policy")],
)

step_efficiency = StepEfficiencyMetric()

dataSet = EvaluationDataset(goldens= [
    Golden(input = "Where is my order ORD-1042?",
           expected_tools=[ToolCall(name="get_order_status")]),
    Golden(input = "What is refund policy for electronics?",
           expected_tools=[ToolCall(name="get_refund_policy")]),
])


display = DisplayConfig(
    truncate_passing_cases=False,   # show score + reason for passing cases too
)

for golden in dataSet.evals_iterator(metrics =[task_completion,tool_correctness,step_efficiency],
                                     display_config=display):
    support_agent(golden.input)

