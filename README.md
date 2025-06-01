# AI Prompter

A prompt management library using Jinja2 templates to build complex prompts easily. Supports raw text or file-based templates and integrates with LangChain.

## Features

- Define prompts as Jinja templates.
- Load default templates from `src/ai_prompter/prompts`.
- Override templates via `PROMPTS_PATH` environment variable.
- Automatic project root detection for prompt templates.
- Render prompts with arbitrary data or Pydantic models.
- Export to LangChain `ChatPromptTemplate`.
- Automatic output parser integration for structured outputs.

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
PROMPTS_PATH=path/to/custom/templates
```

## Usage

### Basic Usage

```python
from ai_prompter import Prompter

# Initialize with a template name
prompter = Prompter('my_template')

# Render a prompt with variables
prompt = prompter.render({'variable': 'value'})
print(prompt)
```

### Custom Prompt Directory

You can specify a custom directory for your prompt templates using the `prompt_dir` parameter:

```python
prompter = Prompter(template_text='Hello {{ name }}!', prompt_dir='/path/to/your/prompts')
```

### Using Environment Variable for Prompt Path

Set the `PROMPTS_PATH` environment variable to point to your custom prompts directory:

```bash
export PROMPTS_PATH=/path/to/your/prompts
```

You can specify multiple directories separated by `:` (colon):

```bash
export PROMPTS_PATH=/path/to/templates1:/path/to/templates2
```

### Template Search Order

The `Prompter` class searches for templates in the following locations (in order of priority):

1. **Custom directory** - If you provide `prompt_dir` parameter when initializing Prompter
2. **Environment variable paths** - Directories specified in `PROMPTS_PATH` (colon-separated)
3. **Current directory prompts** - `./prompts` subfolder in your current working directory
4. **Project root prompts** - Automatically detects your Python project root (by looking for `pyproject.toml`, `setup.py`, `setup.cfg`, or `.git`) and checks for a `prompts` folder there
5. **Home directory** - `~/ai-prompter` folder
6. **Package defaults** - Built-in templates at `src/ai_prompter/prompts`

This allows you to organize your project with prompts at the root level, regardless of your package structure:
```
my-project/
├── prompts/           # <- Templates here will be found automatically
│   └── my_template.jinja
├── src/
│   └── my_package/
│       └── main.py
└── pyproject.toml
```

### Using File-based Templates

You can store your templates in files and reference them by name (without the `.jinja` extension). The library will search through all configured paths (see Template Search Order above) until a matching template is found.

```python
from ai_prompter import Prompter

# Will search for 'greet.jinja' in all configured paths
prompter = Prompter(prompt_template="greet")
result = prompter.render({"name": "World"})
print(result)  # Output depends on the content of greet.jinja
```

You can also specify multiple search paths via environment variable:

```python
import os
from ai_prompter import Prompter

# Set multiple search paths
os.environ["PROMPTS_PATH"] = "/path/to/templates1:/path/to/templates2"

prompter = Prompter(prompt_template="greet")
result = prompter.render({"name": "World"})
print(result)  # Uses greet.jinja from the first path where it's found
```

### Raw text template

```python
from ai_prompter import Prompter

template = """Write an article about {{ topic }}."""
prompter = Prompter(template_text=template)
prompt = prompter.render({"topic": "AI"})
print(prompt)  # Write an article about AI.
```

### Using Raw Text Templates

Alternatively, you can provide the template content directly as raw text using the `template_text` parameter or the `from_text` class method.

```python
from ai_prompter import Prompter

# Using template_text parameter
prompter = Prompter(template_text="Hello, {{ name }}!")
result = prompter.render({"name": "World"})
print(result)  # Output: Hello, World!

# Using from_text class method
prompter = Prompter.from_text("Hi, {{ person }}!", model="gpt-4")
result = prompter.render({"person": "Alice"})
print(result)  # Output: Hi, Alice!
```

### LangChain Integration

You can convert your prompts to LangChain's `ChatPromptTemplate` format for use in LangChain workflows. This works for both text-based and file-based templates.

```python
from ai_prompter import Prompter

# With text-based template
text_prompter = Prompter(template_text="Hello, {{ name }}!")
lc_text_prompt = text_prompter.to_langchain()

# With file-based template
file_prompter = Prompter(prompt_template="greet")
lc_file_prompt = file_prompter.to_langchain()
```

**Note**: LangChain integration requires the `langchain-core` package. Install it with `pip install .[langchain]`.

### Using Output Parsers

The Prompter class supports LangChain output parsers to automatically inject formatting instructions into your prompts. When you provide a parser, it will call the parser's `get_format_instructions()` method and make the result available as `{{ format_instructions }}` in your template.

```python
from ai_prompter import Prompter
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define your output model
class Article(BaseModel):
    title: str = Field(description="Article title")
    summary: str = Field(description="Brief summary")
    tags: list[str] = Field(description="Relevant tags")

# Create a parser
parser = PydanticOutputParser(pydantic_object=Article)

# Create a prompter with the parser
prompter = Prompter(
    template_text="""Write an article about {{ topic }}.

{{ format_instructions }}""",
    parser=parser
)

# Render the prompt - format instructions are automatically included
prompt = prompter.render({"topic": "AI Safety"})
print(prompt)
# Output will include the topic AND the parser's format instructions
```

This works with file-based templates too:

```jinja
# article_structured.jinja
Write an article about {{ topic }}.

Please format your response according to these instructions:
{{ format_instructions }}
```

```python
prompter = Prompter(
    prompt_template="article_structured",
    parser=parser
)
```

The parser integration supports any LangChain output parser that implements `get_format_instructions()`, including:
- `PydanticOutputParser` - For structured Pydantic model outputs
- `OutputFixingParser` - For fixing malformed outputs
- `RetryOutputParser` - For retrying failed parsing attempts
- `StructuredOutputParser` - For dictionary-based structured outputs

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

prompter = Prompter(template_text="Current time: {{current_time}}")
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