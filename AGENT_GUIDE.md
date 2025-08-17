# SmolAgents Decision Guide

## Quick Decision Tree

**Which agent should you use?**

```
Start Here
    ↓
Is your task simple and straightforward?
    ↓
YES → Use ToolCallingAgent
    - Calculations (2+2, derivatives, etc.)
    - API calls (weather, currency conversion)
    - Simple tool execution
    - Quick answers needed
    
NO → Use MultiStepAgent
    - Research and analysis
    - Multi-step problem solving
    - Complex reasoning required
    - Planning and strategy needed
```

## Detailed Comparison

### ToolCallingAgent
**Best for:** Simple, single-step tasks

**Characteristics:**
- ⚡ **Fast**: No planning overhead
- 🎯 **Direct**: One tool call per step
- 💾 **Lightweight**: Minimal memory usage
- 🔧 **Simple**: JSON-based tool calls

**Perfect for:**
- Mathematical calculations
- API calls and data fetching
- Simple text processing
- Quick answers and conversions

### MultiStepAgent
**Best for:** Complex, multi-step reasoning

**Characteristics:**
- 🧠 **Intelligent**: Planning and reasoning capabilities
- 🔄 **Iterative**: Multi-step problem solving
- 📝 **Detailed**: Comprehensive planning output
- 🎯 **Strategic**: Can break down complex tasks

**Perfect for:**
- Research and analysis
- Complex problem solving
- Multi-step workflows
- Tasks requiring explanation

## Output Verbosity Control

Both agent types support configurable output levels to match your use case:

### Production Use (Recommended)
```python
from smolleragents import LogLevel

# Clean, minimal output
agent = create_tool_calling_agent(
    model=model,
    tools=tools,
    verbosity_level=LogLevel.WARNING  # Only errors and warnings
)
```

### Development/Debugging
```python
# Detailed output for troubleshooting
agent = create_tool_calling_agent(
    model=model,
    tools=tools,
    verbosity_level=LogLevel.DEBUG  # Everything including planning steps
)
```

### Verbosity Levels Explained

| Level | Output | Use Case |
|-------|--------|----------|
| `WARNING` | Errors and warnings only | Production, clean logs |
| `INFO` | Tool calls, results, final answers | Default, good balance |
| `DEBUG` | Everything + planning, observations | Development, troubleshooting |

**What you'll see at each level:**

**WARNING (Production):**
```
❌ Error: Tool not found
```

**INFO (Default):**
```
Calling tool: 'python_interpreter' with arguments: {'code': '2**10'}
Final answer: 1024
```

**DEBUG (Development):**
```
Planning: I need to calculate 2^10 using Python
Calling tool: 'python_interpreter' with arguments: {'code': '2**10'}
Observations: 1024
Final answer: 1024
```

## Code Examples

### ToolCallingAgent Setup
```python
from smolleragents import create_tool_calling_agent, OpenAIServerModel

model = OpenAIServerModel("gpt-3.5-turbo")
agent = create_tool_calling_agent(
    model=model,
    tools=[PythonInterpreterTool()],
    max_steps=3  # Usually sufficient for simple tasks
)
```

### MultiStepAgent Setup
```python
from smolleragents import create_multi_step_agent, OpenAIServerModel

model = OpenAIServerModel("gpt-4")
agent = create_multi_step_agent(
    model=model,
    tools=[PythonInterpreterTool()],
    max_steps=10  # More steps for complex reasoning
)
```

## Performance Considerations

| Aspect | ToolCallingAgent | MultiStepAgent |
|--------|------------------|----------------|
| **Speed** | ⚡ Fast | 🐌 Slower (planning overhead) |
| **Token Usage** | 💰 Lower | 💰 Higher (planning steps) |
| **Accuracy** | ✅ Good for simple tasks | ✅ Better for complex tasks |
| **Memory** | 💾 Minimal | 💾 Full conversation history |

## Migration Guide

**Starting with ToolCallingAgent?**
- If you need more reasoning → Switch to MultiStepAgent
- If you need planning → Switch to MultiStepAgent
- If you need state management → Switch to MultiStepAgent

**Starting with MultiStepAgent?**
- If it's too slow for simple tasks → Switch to ToolCallingAgent
- If you don't need planning → Switch to ToolCallingAgent
- If you want to reduce token usage → Switch to ToolCallingAgent

## Pro Tips

1. **Start Simple**: Begin with ToolCallingAgent for most tasks
2. **Upgrade When Needed**: Switch to MultiStepAgent if you need more reasoning
3. **Monitor Performance**: Use streaming mode to see how agents work
4. **Customize Planning**: Adjust `planning_interval` for MultiStepAgent
5. **Use Helper Functions**: `create_tool_calling_agent()` and `create_multi_step_agent()` are pre-configured

## Still Unsure?

**Default Recommendation:** Start with `ToolCallingAgent`. It's fast, efficient, and handles most common use cases. You can always upgrade to `MultiStepAgent` if you need more sophisticated reasoning.

```python
# When in doubt, start here:
from smolleragents import create_tool_calling_agent, OpenAIServerModel

model = OpenAIServerModel("gpt-3.5-turbo")
agent = create_tool_calling_agent(model=model, tools=[PythonInterpreterTool()])
result = agent.run("Your task here")
``` 