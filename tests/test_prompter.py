import os
import pytest
from pydantic import BaseModel

from ai_prompter import Prompter


def test_raw_text_template():
    template = "Hello {{name}}!"
    p = Prompter(prompt_text=template)
    assert p.render({"name": "World"}) == "Hello World!"


def test_missing_both_raises():
    with pytest.raises(ValueError):
        Prompter()


def test_base_model_data():
    class DataModel(BaseModel):
        foo: str

    template = "Value is {{foo}}"
    p = Prompter(prompt_text=template)
    result = p.render(DataModel(foo="BAR"))
    assert "Value is BAR" in result


def test_to_langchain_import_error():
    p = Prompter(prompt_text="X")
    with pytest.raises(ImportError):
        p.to_langchain()


def test_file_template():
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    os.environ["PROMPT_PATH"] = prompt_dir

    p = Prompter(prompt_template="greet")
    result = p.render({"who": "Tester"})
    assert result == "GREET Tester"


def test_missing_template_file():
    # Test when a template file does not exist
    with pytest.raises(ValueError, match="Template nonexistent not found"):
        Prompter(prompt_template="nonexistent")


def test_custom_prompt_path_not_found():
    # Test when custom PROMPT_PATH is set but doesn't exist
    os.environ["PROMPT_PATH"] = "/nonexistent/path"
    with pytest.raises(ValueError, match="Template greet not found"):
        Prompter(prompt_template="greet")


def test_empty_template_file_name():
    # Test with empty template name
    with pytest.raises(ValueError, match="Template name cannot be empty"):
        Prompter(prompt_template="")


def test_empty_text_template():
    # Test with empty text template
    p = Prompter(prompt_text="")
    result = p.render({"key": "value"})
    assert result == ""


def test_template_with_no_variables():
    # Test template with no variables
    p = Prompter(prompt_text="Static content")
    result = p.render({"key": "value"})
    assert result == "Static content"


def test_render_with_no_data():
    # Test rendering with no data provided
    p = Prompter(prompt_text="Hello {{name|default('Guest')}}!")
    result = p.render()
    assert result == "Hello Guest!"


def test_current_time_in_render():
    # Test if current_time is available in render data
    p = Prompter(prompt_text="Time: {{current_time}}")
    result = p.render()
    assert "Time: " in result
    assert len(result) > len("Time: ")
