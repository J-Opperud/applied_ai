import ollama


MODEL = "llama3.2"


def ask_ollama(prompt):
    """Send a prompt to the local Ollama model."""
    response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)


    return response["message"]["content"]


# ============================================================
# Task 1 — Code Explanation
# ============================================================

bad_prompt_1 = """
What does st.session_state do?
"""

good_prompt_1 = """
You are a Python and Streamlit instructor.

Explain what st.session_state does in Streamlit.

Explain that Streamlit reruns the script when users interact
with widgets, and that st.session_state allows values to persist
between those reruns during the user's current session.

Give a simple counter example using a button.

Important:

    Do not claim that session state persists after the user closes
    the session or restarts the application.

    Explain the concept in beginner-friendly language.

    Keep the answer under 150 words.
    """


print("\n" + "=" * 60)
print("TASK 1 — CODE EXPLANATION")
print("=" * 60)

print("\nBAD PROMPT:")
print(ask_ollama(bad_prompt_1))

print("\nGOOD PROMPT:")
print(ask_ollama(good_prompt_1))


# ============================================================
# Task 2 — Data Formatting
# ============================================================

tasks = """
I need to finish my Python homework, review Streamlit,
and study for my AI quiz. Python homework is high priority.
The Streamlit review is medium priority. The AI quiz is high
priority and is not started yet.
"""

bad_prompt_2 = f"""
Turn these tasks into JSON:

{tasks}
"""

good_prompt_2 = f"""
You are a data formatting assistant.

Convert the following task list into a JSON array.

Each task must contain exactly these fields:
- title
- priority
- status

Use only these priority values:
"high", "medium", "low"

Use only these status values:
"not_started", "in_progress", "completed"

Return valid JSON only. Do not include Markdown
or explanations.

Tasks:
{tasks}
"""

print("\n" + "=" * 60)
print("TASK 2 — DATA FORMATTING")
print("=" * 60)

print("\nBAD PROMPT:")
print(ask_ollama(bad_prompt_2))

print("\nGOOD PROMPT:")
print(ask_ollama(good_prompt_2))


# ============================================================
# Task 3 — System Prompt Design
# ============================================================

system_prompt = """
You are a Course Study Assistant.

Answer questions using only the provided course context.

Rules:
1. Do not use information that is not contained in the context.
2. If the answer cannot be found in the context, say:
   "I don't know based on the provided course notes."
3. Keep every answer under 150 words.
4. Include the source document used for the answer.
5. If multiple sources were used, list all relevant sources.

Course context:
{context}

Student question:
{question}
"""

print("\n" + "=" * 60)
print("TASK 3 — SYSTEM PROMPT")
print("=" * 60)

print(system_prompt)
"""
OUTPUTS:

ed121@Ed1217795 MINGW64 ~/documents/A.I/applied_ai
$ python  prompt_workshop.py

============================================================
TASK 1 — CODE EXPLANATION
============================================================

BAD PROMPT:
St.session_state is a function in the Scikit-learn library that is used to create a dictionary-like object that can be used to store and retrieve session-specific data.

When you call `from_state()` or `to_state()` with St.session_state, you create a dictionary-like object that can be used to store and retrieve session-specific data. This is useful for several reasons:

1.  **Session persistence**: When you use `from_state()` or `to_state()` with St.session_state, you can persist your session data across multiple function calls, which is useful for complex models that require multiple iterations.
2.  **Reproducibility**: By storing your session data in `St.session_state`, you can ensure that your results are reproducible, which is essential in scientific computing and data science.
3.  **Easy debugging**: With `St.session_state`, you can easily inspect and debug your code by printing the contents of `session_state` at any point in your code.

Here's an example of how you can use `St.session_state`:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
model = LinearRegression()

# Store the model in session_state
St.session_state.model = model

# Create some data
X = np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1)

# Fit the model
model.fit(X, y)

# Store the data in session_state
St.session_state.X = X
St.session_state.y = y

# Now you can access the data and model in session_state
print(St.session_state.model)
print(St.session_state.X)
print(St.session_state.y)

# You can also use from_state() to load the data and model
model = St.session_state.model
X = St.session_state.X
y = St.session_state.y
```

In summary, `St.session_state` is a powerful tool that allows you to store and retrieve session-specific data, ensuring reproducibility and persistence across multiple function calls.

GOOD PROMPT:
In Streamlit, `st.session_state` is a dictionary that allows you to store values that persist between user interactions. When a user interacts with a widget (like a button or text input), Streamlit reruns the script to update the new input values. This is called "re-runs" or "server-side rendering".

`st.session_state` helps you to "save" these values between re-runs, so they're available when the user interacts again. This allows you to build interactive apps where the user's input affects the output.

Here's a simple example:
```python
import streamlit as st

def counter():
    if 'count' not in st.session_state:
        st.session_state.count = 0

    st.session_state.count += 1
    return st.session_state.count

st.title("Counter App")
if st.button("Click me!"):
    st.write(counter())
```
In this example, when the user clicks the button, `counter()` is called, and the `count` value is incremented and returned. Because of `st.session_state`, the `count` value persists between button clicks, so the app keeps counting!

============================================================
TASK 2 — DATA FORMATTING
============================================================

BAD PROMPT:
Here is the JSON representation of the tasks:

```
{
    "tasks": [
        {
            "name": "Finish Python homework",
            "priority": "high"
        },
        {
            "name": "Review Streamlit",
            "priority": "medium"
        },
        {
            "name": "Study for AI quiz",
            "priority": "high",
            "status": "not started"
        }
    ]
}
```

Note: I added a `status` field to the AI quiz task to indicate that it has not been started yet.

GOOD PROMPT:
 [{"title": "Finish Python homework", "priority": "high", "status": "not_started"}, {"title": "Review Streamlit", "priority": "medium", "status": "in_progress"}, {"title": "Study for AI quiz", "priority": "high", "status": "not_started"}]

============================================================
TASK 3 — SYSTEM PROMPT
============================================================

You are a Course Study Assistant.

Answer questions using only the provided course context.

Rules:
1. Do not use information that is not contained in the context.
2. If the answer cannot be found in the context, say:
   "I don't know based on the provided course notes."
3. Keep every answer under 150 words.
4. Include the source document used for the answer.
5. If multiple sources were used, list all relevant sources.

Course context:
{context}

Student question:
{question}

(venv)
ed121@Ed1217795 MINGW64 ~/documents/A.I/applied_ai

"""