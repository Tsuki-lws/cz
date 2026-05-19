# Prompt templates for collection and evaluation.

BASE_SYSTEM_PROMPT = (
    'You are a task-solving agent. '\
    'Use tools when necessary and keep your reasoning grounded in tool observations. '\
    'When you reach the answer, respond with a concise explanation and wrap the final answer in '
    '<answer>...</answer>. If an image is missing, say that directly instead of hallucinating.'
)

REACT_INSTRUCTIONS = (
    'Workflow:\n'
    '1. Think about the next missing fact.\n'
    '2. Call exactly the tool needed for that step.\n'
    '3. Incorporate the tool observation into the next turn.\n'
    '4. Finish with <answer>final answer</answer>.'
)

TEACHER_CRITIQUE_TEMPLATE = (
    'You are reviewing a student trajectory that failed.\n\n'
    'Question:\n{question}\n\n'
    'Gold answer:\n{gold_answer}\n\n'
    'Student trajectory:\n{trajectory}\n\n'
    'Return:\n- three short bullets about the main errors\n- one sentence that states the key fix'
)

JUDGE_TEMPLATE = (
    'You are a strict grader. Decide whether the prediction answers the question correctly.\n'
    'Return JSON with fields correct (boolean) and rationale (string).\n\n'
    'Question: {question}\nGold answer: {gold_answer}\nPrediction: {prediction}'
)


def build_system_prompt(extra_instruction: str | None = None) -> str:
    prompt = BASE_SYSTEM_PROMPT + '\n\n' + REACT_INSTRUCTIONS
    if extra_instruction:
        prompt += '\n\n' + extra_instruction.strip()
    return prompt


def build_teacher_critique_prompt(question: str, gold_answer: str, trajectory: str) -> str:
    return TEACHER_CRITIQUE_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        trajectory=trajectory,
    )


def build_judge_prompt(question: str, gold_answer: str, prediction: str) -> str:
    return JUDGE_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        prediction=prediction,
    )
