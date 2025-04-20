"""
A prompt management module using Jinja to generate complex prompts with simple templates.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

prompt_path_default = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts"
)
prompt_path_custom = os.getenv("PROMPT_PATH")

env_default = Environment(loader=FileSystemLoader(prompt_path_default))


@dataclass
class Prompter:
    """
    A class for managing and rendering prompt templates.

    Attributes:
        prompt_template (str, optional): The name of the prompt template file.
        prompt_variation (str, optional): The variation of the prompt template.
        prompt_text (str, optional): The raw prompt text.
        template (Union[str, Template], optional): The Jinja2 template object.
    """

    prompt_template: Optional[str] = None
    prompt_variation: Optional[str] = "default"
    prompt_text: Optional[str] = None
    template: Optional[Union[str, Template]] = None
    parser: Optional[Any] = None

    def __init__(self, prompt_template=None, prompt_text=None, parser=None):
        """
        Initialize the Prompter with either a template file or raw text.

        Args:
            prompt_template (str, optional): The name of the prompt template file.
            prompt_text (str, optional): The raw prompt text.
        """
        self.prompt_template = prompt_template
        self.prompt_text = prompt_text
        self.parser = parser
        self.setup()

    def setup(self):
        """
        Set up the Jinja2 template based on the provided template file or text.
        Raises:
            ValueError: If neither prompt_template nor prompt_text is provided, or if template name is empty.
        """
        if self.prompt_template is not None:
            if not self.prompt_template:
                raise ValueError("Template name cannot be empty")
            # attempt to load from custom path at runtime
            custom_path = os.getenv("PROMPT_PATH")
            if custom_path and os.path.exists(custom_path):
                try:
                    env = Environment(loader=FileSystemLoader(custom_path))
                    self.template = env.get_template(f"{self.prompt_template}.jinja")
                    return
                except Exception:
                    pass
            # fallback to default path
            try:
                env = Environment(loader=FileSystemLoader(prompt_path_default))
                self.template = env.get_template(f"{self.prompt_template}.jinja")
            except Exception as e:
                raise ValueError(f"Template {self.prompt_template} not found in default folder: {e}")
        elif self.prompt_text is not None:
            self.template = Template(self.prompt_text)
        else:
            raise ValueError("Prompter must have a prompt_template or prompt_text")

        # Removed assertion as it's redundant with the checks above
        # assert self.prompt_template or self.prompt_text, "Prompt is required"

    def to_langchain(self):
        # only file-based templates supported for LangChain
        if self.prompt_text is not None:
            raise ImportError(
                "langchain-core integration only supports file-based templates; install with `pip install .[langchain]`"
            )
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            raise ImportError(
                "langchain-core is required for to_langchain; install with `pip install .[langchain]`"
            )
        if isinstance(self.template, str):
            template_text = self.template
        elif isinstance(self.template, Template):
            # raw Jinja2 template object
            template_text = self.prompt_text
        else:
            # file-based template
            prompt_dir = (
                prompt_path_custom
                if prompt_path_custom and os.path.exists(prompt_path_custom)
                else prompt_path_default
            )
            template_file = os.path.join(prompt_dir, f"{self.prompt_template}.jinja")
            with open(template_file, "r") as f:
                template_text = f.read()
        return ChatPromptTemplate.from_template(template_text, template_format="jinja2")

    @classmethod
    def from_text(cls, text: str):
        """
        Create a Prompter instance from raw text, which can contain Jinja code.

        Args:
            text (str): The raw prompt text.

        Returns:
            Prompter: A new Prompter instance.
        """

        return cls(prompt_text=text)

    def render(self, data: Optional[Union[Dict, BaseModel]] = None) -> str:
        """
        Render the prompt template with the given data.

        Args:
            data (Union[Dict, BaseModel]): The data to be used in rendering the template.
                Can be either a dictionary or a Pydantic BaseModel.

        Returns:
            str: The rendered prompt text.

        Raises:
            AssertionError: If the template is not defined or not a Jinja2 Template.
        """
        if isinstance(data, BaseModel):
            data_dict = data.model_dump()
        elif isinstance(data, dict):
            data_dict = data
        else:
            data_dict = {}
        render_data = dict(data_dict)
        render_data["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.parser:
            render_data["format_instructions"] = self.parser.get_format_instructions()
        assert self.template, "Prompter template is not defined"
        assert isinstance(
            self.template, Template
        ), "Prompter template is not a Jinja2 Template"
        return self.template.render(render_data)
