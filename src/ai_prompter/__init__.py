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
prompt_path_custom = os.getenv("PROMPTS_PATH")

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
    template_text: Optional[str] = None
    parser: Optional[Any] = None

    def __init__(
        self,
        prompt_template: Optional[str] = None,
        model: Optional[Union[str, Any]] = None,
        template_text: Optional[str] = None,
        prompt_dir: Optional[str] = None,
    ) -> None:
        """Initialize the Prompter with a template name, model, and optional custom directory.

        Args:
            prompt_template (str, optional): The name of the prompt template (without .jinja extension).
            model (Union[str, Any], optional): The model to use for generation.
            template_text (str, optional): The raw text of the template.
            prompt_dir (str, optional): Custom directory to search for templates.
        """
        self.prompt_template = prompt_template
        self.template = None
        self.template_text = template_text
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self._setup_template(template_text, prompt_dir)

    def _setup_template(
        self, template_text: Optional[str] = None, prompt_dir: Optional[str] = None
    ) -> None:
        """Set up the Jinja2 template based on the provided template file or text.

        Args:
            template_text (str, optional): The raw text of the template.
            prompt_dir (str, optional): Custom directory to search for templates.
        """
        if template_text is None:
            if self.prompt_template is None:
                raise ValueError(
                    "Either prompt_template or template_text must be provided"
                )
            if not self.prompt_template:
                raise ValueError("Template name cannot be empty")
            prompt_dirs = []
            if prompt_dir:
                prompt_dirs.append(prompt_dir)
            prompts_path = os.getenv("PROMPTS_PATH")
            if prompts_path is not None:
                prompt_dirs.extend(prompts_path.split(":"))
            # Fallback to local folder and ~/ai-prompter
            prompt_dirs.extend([os.getcwd(), os.path.expanduser("~/ai-prompter")])
            # Default package prompts folder
            if os.path.exists(prompt_path_default):
                prompt_dirs.append(prompt_path_default)
            env = Environment(loader=FileSystemLoader(prompt_dirs))
            self.template = env.get_template(f"{self.prompt_template}.jinja")
        else:
            self.template_text = template_text
            self.template = Template(template_text)

    def to_langchain(self):
        # Support for both text-based and file-based templates with LangChain
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            raise ImportError(
                "langchain-core is required for to_langchain; install with `pip install .[langchain]`"
            )
        if self.template_text is not None:
            template_content = self.template_text
        elif self.prompt_template is not None and self.template is not None:
            # For file-based templates, we need to get the raw string content
            if isinstance(self.template, Template):
                # If we have a Jinja2 Template object, we might not have the raw string
                # Try to read the file from the environment's loader search path
                template_content = None
                for searchpath in self.template.environment.loader.searchpath:
                    template_file = os.path.join(
                        searchpath, f"{self.prompt_template}.jinja"
                    )
                    if os.path.exists(template_file):
                        with open(template_file, "r") as f:
                            template_content = f.read()
                        break
                if template_content is None:
                    raise ValueError(
                        "Could not load file-based template content for LangChain conversion"
                    )
            else:
                raise ValueError(
                    "Template is not properly initialized for LangChain conversion"
                )
        else:
            raise ValueError(
                "Either prompt_template with a valid template or template_text must be provided for LangChain conversion"
            )
        return ChatPromptTemplate.from_template(
            template_content, template_format="jinja2"
        )

    @classmethod
    def from_text(
        cls, text: str, model: Optional[Union[str, Any]] = None
    ) -> "Prompter":
        """Create a Prompter instance from raw text, which can contain Jinja code.

        Args:
            text (str): The raw template text.
            model (Union[str, Any], optional): The model to use for generation.

        Returns:
            Prompter: A new Prompter instance.
        """
        return cls(template_text=text, model=model)

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
