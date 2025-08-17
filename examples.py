#!/usr/bin/env python
# coding=utf-8

"""
SmolAgents Examples
==================

This file contains working examples of how to use both agent types.

Quick Start Examples:
1. Simple calculation with ToolCallingAgent
2. Complex reasoning with MultiStepAgent
3. Custom tools with both agents
4. Streaming mode examples
5. Verbosity control examples

Run these examples to see how each agent type works.
"""

import os
from smolleragents import (
    create_tool_calling_agent,
    create_multi_step_agent,
    OpenAIServerModel,
    PythonInterpreterTool,
    LogLevel,
    tool
)


def example_1_simple_calculation():
    """
    Example 1: Simple calculation using ToolCallingAgent
    
    Use ToolCallingAgent for straightforward calculations and API calls.
    """
    print("=== Example 1: Simple Calculation with ToolCallingAgent ===")
    
    model = OpenAIServerModel("gpt-4o-mini")
    agent = create_tool_calling_agent(
        model=model,
        tools=[PythonInterpreterTool()]
    )
    
    result = agent.run("What is 2^10?")
    print(f"Result: {result}")
    print()


def example_2_complex_reasoning():
    """
    Example 2: Complex reasoning using MultiStepAgent
    
    Use MultiStepAgent for tasks that require planning and multi-step reasoning.
    """
    print("=== Example 2: Complex Reasoning with MultiStepAgent ===")
    
    model = OpenAIServerModel("gpt-4o-mini")
    agent = create_multi_step_agent(
        model=model,
        tools=[PythonInterpreterTool()],
        max_steps=5
    )
    
    result = agent.run("Calculate the derivative of x^2 + 3x + 1 and explain each step")
    print(f"Result: {result}")
    print()


def example_3_verbosity_control():
    """
    Example 3: Controlling output verbosity
    
    Demonstrates how to control the amount of output for different use cases.
    """
    print("=== Example 3: Verbosity Control ===")
    
    model = OpenAIServerModel("gpt-4o-mini")
    
    # Production mode - minimal output
    print("--- Production Mode (WARNING level) ---")
    agent_quiet = create_tool_calling_agent(
        model=model,
        tools=[PythonInterpreterTool()],
        verbosity_level=LogLevel.WARNING
    )
    result_quiet = agent_quiet.run("What is 5 * 7?")
    print(f"Result: {result_quiet}")
    print()
    
    # Normal mode - standard feedback
    print("--- Normal Mode (INFO level) ---")
    agent_normal = create_tool_calling_agent(
        model=model,
        tools=[PythonInterpreterTool()],
        verbosity_level=LogLevel.INFO
    )
    result_normal = agent_normal.run("What is 5 * 7?")
    print(f"Result: {result_normal}")
    print()
    
    # Debug mode - detailed output
    print("--- Debug Mode (DEBUG level) ---")
    agent_debug = create_tool_calling_agent(
        model=model,
        tools=[PythonInterpreterTool()],
        verbosity_level=LogLevel.DEBUG
    )
    result_debug = agent_debug.run("What is 5 * 7?")
    print(f"Result: {result_debug}")
    print()


def example_4_custom_tools():
    """
    Example 4: Using custom tools
    
    Shows how to create and use custom tools with both agent types.
    """
    print("=== Example 4: Custom Tools ===")
    
    @tool
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}! Nice to meet you!"
    
    model = OpenAIServerModel("gpt-4o-mini")
    
    # ToolCallingAgent with custom tools
    print("--- ToolCallingAgent with custom tools ---")
    agent_simple = create_tool_calling_agent(
        model=model,
        tools=[greet]
    )
    result = agent_simple.run("Greet Alice")
    print(f"Result: {result}")
    print()
    
    # MultiStepAgent with custom tools
    print("--- MultiStepAgent with custom tools ---")
    agent_complex = create_multi_step_agent(
        model=model,
        tools=[greet],
        max_steps=3
    )
    result = agent_complex.run("Greet Bob and explain why you're greeting them")
    print(f"Result: {result}")
    print()


def example_5_streaming_mode():
    """
    Example 5: Streaming mode
    
    Shows how to use streaming mode to get real-time updates.
    """
    print("=== Example 5: Streaming Mode ===")
    
    model = OpenAIServerModel("gpt-4o-mini")
    agent = create_tool_calling_agent(
        model=model,
        tools=[PythonInterpreterTool()]
    )
    
    print("--- ToolCallingAgent Streaming ---")
    print("Streaming steps:")
    for step in agent.run("What is 3^4?", stream=True):
        print(f"  Step: {type(step).__name__}")
    print()


def example_6_agent_comparison():
    """
    Example 6: Comparing agent types
    
    Shows the difference between ToolCallingAgent and MultiStepAgent.
    """
    print("=== Example 6: Agent Comparison ===")
    
    model = OpenAIServerModel("gpt-4o-mini")
    
    # Simple task - both agents should work similarly
    print("--- ToolCallingAgent (Simple) ---")
    agent_simple = create_tool_calling_agent(
        model=model,
        tools=[PythonInterpreterTool()],
        verbosity_level=LogLevel.INFO
    )
    result_simple = agent_simple.run("Calculate 2 + 2")
    print(f"Result: {result_simple}")
    print()
    
    # Complex task - MultiStepAgent will show planning
    print("--- MultiStepAgent (With Planning) ---")
    agent_complex = create_multi_step_agent(
        model=model,
        tools=[PythonInterpreterTool()],
        max_steps=3,
        verbosity_level=LogLevel.INFO
    )
    result_complex = agent_complex.run("Calculate 2 + 2 and explain why this is a fundamental operation")
    print(f"Result: {result_complex}")
    print()


if __name__ == "__main__":
    print("SmolAgents Examples")
    print("==================")
    print()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print()
        print("Examples will show the structure but won't run without an API key.")
    else:
        print("✅ OpenAI API key found")
        print()
        
        try:
            # Run examples
            example_1_simple_calculation()
            example_2_complex_reasoning()
            example_3_verbosity_control()
            example_4_custom_tools()
            example_5_streaming_mode()
            example_6_agent_comparison()
            
            print("🎉 All examples completed successfully!")
        except Exception as e:
            print(f"❌ Example failed with error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc() 