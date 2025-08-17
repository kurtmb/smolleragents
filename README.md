# SmolAgents

A lightweight ReACT agent framework with OpenAI API integration, designed for serverless and edge deployments.

**SmolAgents** is a streamlined, production-ready fork of the original [SmolAgents](https://github.com/huggingface/smolagents) by HuggingFace, optimized for simplicity and deployment efficiency.

## Features

- **Lightweight**: Minimal dependencies, optimized for serverless environments
- **OpenAI Integration**: Built-in support for OpenAI API with streaming
- **ReACT Framework**: Step-by-step reasoning with tool support
- **Memory Management**: Conversation history and context tracking
- **Type Safety**: Full type hints and validation with Pydantic
- **Error Handling**: Robust error handling and recovery
- **Tool Ecosystem**: Extensible tool system with built-in utilities

## Production Ready

SmolAgents is designed for production use with configurable output levels:

### Clean Output for Production
```python
# Minimal output - only errors and warnings
agent = create_tool_calling_agent(
    model=model,
    tools=tools,
    verbosity_level=LogLevel.WARNING  # Production recommended
)
```

### Development vs Production
- **Development**: Use `LogLevel.DEBUG` for detailed troubleshooting
- **Production**: Use `LogLevel.WARNING` for clean, minimal output
- **Default**: `LogLevel.INFO` provides good balance for most use cases

### What's Included
- ✅ **No debug prints**: Clean console output
- ✅ **Configurable verbosity**: Control output level per agent
- ✅ **Error handling**: Robust error management
- ✅ **Type safety**: Full type hints and validation
- ✅ **Lightweight**: Minimal dependencies for serverless deployment

## Installation

### Development Installation (Recommended)

For local development and testing:

```bash
# Clone the repository
git clone <your-repo-url>
cd smolleragents

# Install in development mode
pip install -e .
```

### Production Installation

```bash
pip install smolleragents
```

## Quick Start

### Choose Your Agent

**ToolCallingAgent** (Recommended for most tasks):
- Simple, direct tool execution
- Fast and lightweight
- Perfect for calculations, API calls, and straightforward tasks

**MultiStepAgent** (For complex reasoning):
- Multi-step planning and reasoning
- Better for research, analysis, and complex problem solving
- More verbose output with planning steps

### Basic Usage

```python
from smolleragents import create_tool_calling_agent, OpenAIServerModel, PythonInterpreterTool

# Create a model
model = OpenAIServerModel("gpt-4o-mini")

# Create an agent with tools
agent = create_tool_calling_agent(
    model=model,
    tools=[PythonInterpreterTool()]
)

# Run a task
result = agent.run("What is 2^10?")
print(result)  # 1024
```

### Controlling Output Verbosity

SmolAgents provides configurable output levels to suit different use cases:

```python
from smolleragents import LogLevel

# Quiet mode - minimal output (production recommended)
agent = create_tool_calling_agent(
    model=model,
    tools=[PythonInterpreterTool()],
    verbosity_level=LogLevel.WARNING
)

# Normal mode - standard user feedback
agent = create_tool_calling_agent(
    model=model,
    tools=[PythonInterpreterTool()],
    verbosity_level=LogLevel.INFO  # Default
)

# Debug mode - detailed output for development
agent = create_tool_calling_agent(
    model=model,
    tools=[PythonInterpreterTool()],
    verbosity_level=LogLevel.DEBUG
)
```

**Verbosity Levels:**
- `LogLevel.WARNING`: Only warnings and errors (cleanest for production)
- `LogLevel.INFO`: Standard feedback including tool calls and results (default)
- `LogLevel.DEBUG`: Detailed output including planning steps and internal state

**What each level shows:**
- **WARNING**: Only errors and warnings
- **INFO**: Tool calls, results, and final answers
- **DEBUG**: Everything + planning steps, observations, and detailed state

## Agent Types Comparison

| Feature | ToolCallingAgent | MultiStepAgent |
|---------|------------------|----------------|
| **Use Case** | Simple, single-step tasks | Complex, multi-step reasoning |
| **Planning** | No planning (planning_interval=None) | Planning every step (planning_interval=1) |
| **Performance** | Fast, efficient | More thorough, slower |
| **Memory** | Minimal state tracking | Full conversation history |
| **Examples** | Calculations, API calls, simple queries | Research, analysis, complex workflows |
| **Best For** | Quick answers, direct tool usage | Problem-solving, exploration, detailed analysis |

## When to Use Each Agent

### Use ToolCallingAgent when:
- ✅ You need a quick answer to a simple question
- ✅ Performing basic calculations or API calls
- ✅ Direct tool execution is sufficient
- ✅ Speed and efficiency are priorities
- ✅ No complex reasoning is required

**Examples:**
- "What is 2 + 2?"
- "Convert 100 USD to EUR"
- "Get the current weather in San Francisco"
- "Calculate the area of a circle with radius 5"

### Use MultiStepAgent when:
- ✅ You need complex reasoning and planning
- ✅ The task requires multiple steps and tools
- ✅ Research or analysis is involved
- ✅ You want explicit planning and strategy
- ✅ Need to maintain state across multiple steps

**Examples:**
- "Research the latest AI trends and summarize key findings"
- "Analyze this dataset and identify patterns"
- "Solve this complex math problem step by step"
- "Compare different approaches to a problem"

## Usage Examples

### Basic ToolCallingAgent with Custom Tools

```python
from smolleragents import create_tool_calling_agent, OpenAIServerModel, tool

# Create a custom tool
@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

# Create agent with custom tool
model = OpenAIServerModel("gpt-3.5-turbo")
agent = create_tool_calling_agent(
    model=model,
    tools=[calculate_area]
)

# Use the agent
result = agent.run("Calculate the area of a rectangle with length 5 and width 3")
print(result)  # Output: 15.0
```

### MultiStepAgent with Multiple Tools

```python
from smolleragents import create_multi_step_agent, OpenAIServerModel, PythonInterpreterTool

# Create agent with multiple tools
model = OpenAIServerModel("gpt-4")
agent = create_multi_step_agent(
    model=model,
    tools=[PythonInterpreterTool()],
    max_steps=15
)

# Complex task requiring multiple steps
result = agent.run("""
    Solve this problem step by step:
    1. Calculate the derivative of f(x) = x^3 + 2x^2 + x
    2. Find the critical points
    3. Determine if each critical point is a maximum, minimum, or saddle point
    4. Provide a summary of your findings
""")
print(result)
```

### Streaming Mode

Both agent types support streaming for real-time output:

```python
from smolleragents import create_tool_calling_agent, OpenAIServerModel

model = OpenAIServerModel("gpt-3.5-turbo")
agent = create_tool_calling_agent(model=model, tools=[PythonInterpreterTool()])

# Run in streaming mode
for step in agent.run("Calculate 2 + 2", stream=True):
    print(f"Step: {step}")
```

## Advanced Configuration

### Custom Planning Intervals

```python
from smolleragents import MultiStepAgent, OpenAIServerModel

# Create MultiStepAgent with custom planning interval
model = OpenAIServerModel("gpt-4")
agent = MultiStepAgent(
    model=model,
    tools=[PythonInterpreterTool()],
    planning_interval=3,  # Plan every 3 steps
    max_steps=20
)
```

### Verbosity Control

```python
from smolleragents import create_tool_calling_agent, LogLevel

agent = create_tool_calling_agent(
    model=model,
    tools=[PythonInterpreterTool()],
    verbosity_level=LogLevel.DEBUG  # More detailed logging
)
```

## More Examples

For comprehensive working examples, see the `examples.py` file in the package root:

```bash
# Run examples
python examples.py
```

The examples include:
- Simple calculations with ToolCallingAgent
- Complex reasoning with MultiStepAgent
- Custom tools with both agents
- Streaming mode examples
- Direct comparison of both agent types

## Dependencies

- `openai>=1.0.0` - OpenAI API client
- `pydantic>=2.0.0` - Data validation
- `requests>=2.25.0` - HTTP requests
- `duckduckgo-search>=4.0.0` - Web search
- `beautifulsoup4>=4.9.0` - HTML parsing
- `lxml>=4.6.0` - XML/HTML processing
- `jinja2>=3.0.0` - Template engine
- `pyyaml>=6.0.0` - YAML processing

## License

Apache License 2.0

## Attribution

This project is based on the original [SmolAgents](https://github.com/huggingface/smolagents) by HuggingFace Inc. The original work is licensed under the Apache License 2.0, and this fork maintains the same license while providing additional optimizations and improvements for production deployment.

### Original Authors
- HuggingFace Inc. team - Original SmolAgents framework

### Current Maintainer
- Kurt Boden - Streamlined fork and production optimizations
- Contact: smolleragents@gmail.com 