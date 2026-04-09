# Changelog

## [0.4.0] - 2026-04-09

### Security

- Use Jinja2 `SandboxedEnvironment` for all template rendering to prevent Server-Side Template Injection (SSTI) attacks (CVSS 9.2 Critical). Previously, user-provided template text could execute arbitrary Python code via crafted Jinja2 expressions.

### Changed

- All `jinja2.Environment` instances replaced with `jinja2.sandbox.SandboxedEnvironment`. This blocks access to dangerous attributes (`__globals__`, `__subclasses__`, etc.) and prevents execution of arbitrary code within templates.

## [0.3.2] - 2025-03-20

### Added

- Support for `.jinja` extension in template names
- `template_location()` method

## [0.3.1] - 2025-03-18

### Added

- Support for `{% include %}` in `template_text` mode
- Nested include resolution

## [0.3.0] - 2025-03-15

### Added

- Jinja2 extensions support
- LangChain integration via `to_langchain()`
- `from_text()` class method
- Multiple prompt directory search paths
- `PROMPTS_PATH` environment variable support
