import os
import tempfile

import jinja2
import pytest
from pydantic import BaseModel

from ai_prompter import Prompter


def test_raw_text_template():
    template = "Hello {{name}}!"
    p = Prompter(template_text=template)
    assert p.template is not None
    result = p.render({"name": "World"})
    assert result == "Hello World!"


def test_missing_both_raises():
    with pytest.raises(ValueError):
        Prompter()


def test_base_model_data():
    class DataModel(BaseModel):
        foo: str

    template = "Value is {{foo}}"
    p = Prompter(template_text=template)
    data = DataModel(foo="BAR")
    result = p.render(data)
    assert "Value is BAR" in result


def test_to_langchain_conversion():
    # Test successful conversion for text-based template
    template_text = "Hello, {{ name }}!"
    p = Prompter(template_text=template_text)
    lc_prompt = p.to_langchain()
    assert lc_prompt is not None, "LangChain conversion should succeed with text template"
    assert hasattr(lc_prompt, "invoke"), "Converted object should have invoke method"

    # Test successful conversion for file-based template
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create main template
        template_path = os.path.join(temp_dir, "main.jinja")
        with open(template_path, "w") as f:
            f.write("Hello, {{ name }}!\n{% include 'included.jinja' %}")
        
        # Create included template
        included_path = os.path.join(temp_dir, "included.jinja")
        with open(included_path, "w") as f:
            f.write("Included content for {{ name }}")
            
        p = Prompter(prompt_template="main", prompt_dir=temp_dir)
        lc_prompt = p.to_langchain()
        assert lc_prompt is not None, "LangChain conversion should succeed with file-based template"
        assert hasattr(lc_prompt, "invoke"), "Converted object should have invoke method"
        
        # Test invoking the template
        result = lc_prompt.invoke({"name": "Test"})
        assert "Hello, Test!" in str(result), "Rendered output should contain the substituted variable"
        assert "Included content for Test" in str(result), "Rendered output should include content from included template with substituted variable"


def test_to_langchain_no_template():
    # Test for ValueError when neither template_text nor prompt_template is provided
    with pytest.raises(ValueError):
        p = Prompter()
        p.to_langchain()


def test_file_template():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = os.path.join(temp_dir, "greet.jinja")
        with open(template_path, "w") as f:
            f.write("Hello, {{ name }}!")
        p = Prompter(prompt_template="greet", prompt_dir=temp_dir)
        assert p.template is not None
        result = p.render({"name": "World"})
        assert result == "Hello, World!"


def test_missing_template_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(jinja2.exceptions.TemplateNotFound):
            Prompter(prompt_template="nonexistent", prompt_dir=temp_dir)


def test_custom_prompt_path_not_found():
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(jinja2.exceptions.TemplateNotFound):
            Prompter(prompt_template="greet", prompt_dir=temp_dir)


def test_empty_template_file_name():
    # Test with empty template name
    with pytest.raises(ValueError, match="Template name cannot be empty"):
        Prompter(prompt_template="")


def test_empty_text_template():
    # Test with empty text template
    p = Prompter(template_text="")
    result = p.render({"key": "value"})
    assert result == ""


def test_template_with_no_variables():
    # Test template with no variables
    p = Prompter(template_text="Static content")
    result = p.render({"key": "value"})
    assert result == "Static content"


def test_render_with_no_data():
    # Test rendering with no data provided
    p = Prompter(template_text="Hello {{name|default('Guest')}}!")
    result = p.render()
    assert result == "Hello Guest!"


def test_current_time_in_render():
    # Test if current_time is available in render data
    p = Prompter(template_text="Time: {{current_time}}")
    result = p.render()
    assert "Time: " in result
    assert len(result) > len("Time: ")


def test_prompter_init_with_custom_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = os.path.join(temp_dir, "test_template.jinja")
        with open(template_path, "w") as f:
            f.write("Hello {{ name }}")
        prompter = Prompter("test_template", prompt_dir=temp_dir)
        assert prompter.template is not None
        result = prompter.render({"name": "World"})
        assert result == "Hello World"


def test_prompter_init_with_prompts_path_env():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = os.path.join(temp_dir, "test_template.jinja")
        with open(template_path, "w") as f:
            f.write("Hello {{ name }}")
        os.environ["PROMPTS_PATH"] = temp_dir
        prompter = Prompter("test_template")
        assert prompter.template is not None
        result = prompter.render({"name": "World"})
        assert result == "Hello World"
        del os.environ["PROMPTS_PATH"]


def test_from_text_class_method():
    # Test the from_text class method for initializing with raw text
    model = "gpt-4"
    p = Prompter.from_text("Simple text {{ data }}", model)
    assert p.model == model
    assert p.template_text == "Simple text {{ data }}"
    result = p.render({"data": "test"})
    assert result == "Simple text test"


def test_multiple_prompts_path():
    # Test handling of multiple directories in PROMPTS_PATH
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            template_path1 = os.path.join(temp_dir1, "multi.jinja")
            with open(template_path1, "w") as f:
                f.write("First {{ name }}!")
            template_path2 = os.path.join(temp_dir2, "multi.jinja")
            with open(template_path2, "w") as f:
                f.write("Second {{ name }}!")
            original_path = os.getenv("PROMPTS_PATH", "")
            os.environ["PROMPTS_PATH"] = f"{temp_dir1}:{temp_dir2}"
            try:
                p = Prompter(prompt_template="multi")
                result = p.render({"name": "Tester"})
                assert (
                    result == "First Tester!"
                ), "Should pick the first available template in PROMPTS_PATH"
            finally:
                os.environ["PROMPTS_PATH"] = original_path
