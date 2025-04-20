# AI Prompter

A prompt management library using Jinja2 templates to build complex prompts easily. Supports raw text or file-based templates and integrates with LangChain.

## Features

- Define prompts as Jinja templates.
- Load default templates from `src/ai_prompter/prompts`.
- Override templates via `PROMPT_PATH` environment variable.
- Render prompts with arbitrary data or Pydantic models.
- Export to LangChain `ChatPromptTemplate`.

## Installation

1. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the package:
   ```bash
   pip install .
   ```
### Extras

To enable LangChain integration:

```bash
pip install .[langchain]
# or
uv add langchain_core
```

## Configuration

Configure a custom template path by creating a `.env` file in the project root:

```dotenv
PROMPT_PATH=path/to/custom/templates
```

## Usage

### Raw text template

```python
from ai_prompter import Prompter

template = """Write an article about {{ topic }}."""
prompter = Prompter(prompt_text=template)
prompt = prompter.render({"topic": "AI"})
print(prompt)  # Write an article about AI.
```

### Using File-based Templates

You can store your templates in files and reference them by name (without the `.jinja` extension). The library looks for templates in the `prompts` directory by default, or you can set a custom directory with the `PROMPT_PATH` environment variable.

```python
from ai_prompter import Prompter

prompter = Prompter(prompt_template="greet")
prompt = prompter.render({"who": "Tester"})
print(prompt)  # GREET Tester
```

### Including Other Templates

You can include other template files within a template using Jinja2's `{% include %}` directive. This allows you to build modular templates.

```jinja
# outer.jinja
This is the outer file

{% include 'inner.jinja' %}

This is the end of the outer file
```

```jinja
# inner.jinja
This is the inner file

{% if type == 'a' %}
    You selected A
{% else %}
    You didn't select A
{% endif %}
```

```python
from ai_prompter import Prompter

prompter = Prompter(prompt_template="outer")
prompt = prompter.render(dict(type="a"))
print(prompt)
# This is the outer file
# 
# This is the inner file
# 
#     You selected A
# 
# 
# This is the end of the outer file
```

### Using Variables

Templates can use variables that you pass in through the `render()` method. You can use Jinja2 filters and conditionals to control the output based on your data.

```python
from ai_prompter import Prompter

prompter = Prompter(prompt_text="Hello {{name|default('Guest')}}!")
prompt = prompter.render()  # No data provided, uses default
print(prompt)  # Hello Guest!
prompt = prompter.render({"name": "Alice"})  # Data provided
print(prompt)  # Hello Alice!
```

The library also automatically provides a `current_time` variable with the current timestamp in format "YYYY-MM-DD HH:MM:SS".

```python
from ai_prompter import Prompter

prompter = Prompter(prompt_text="Current time: {{current_time}}")
prompt = prompter.render()
print(prompt)  # Current time: 2025-04-19 23:28:00
```

### File-based template

Place a Jinja file (e.g., `article.jinja`) in the default prompts directory (`src/ai_prompter/prompts`) or your custom path:

```jinja
Write an article about {{ topic }}.
```

```python
from ai_prompter import Prompter

prompter = Prompter(prompt_template="article")
prompt = prompter.render({"topic": "AI"})
print(prompt)
```

### LangChain integration

```python
from ai_prompter import Prompter

prompter = Prompter(prompt_template="article")
lc_template = prompter.to_langchain()
# use lc_template in LangChain chains
```

### Jupyter Notebook

See `notebooks/prompter_usage.ipynb` for interactive examples.

## Project Structure

```
ai-prompter/
├── src/ai_prompter
│   ├── __init__.py
│   └── prompts/
│       └── *.jinja
├── notebooks/
│   ├── prompter_usage.ipynb
│   └── prompts/
├── pyproject.toml
├── README.md
└── .env (optional)
```

## Testing

Run tests with:

```bash
uv run pytest -v
```

## Contributing

Contributions welcome! Please open issues or PRs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.